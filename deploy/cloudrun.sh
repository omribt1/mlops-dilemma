#!/usr/bin/env bash
# Deploy `dilemma ui` to Cloud Run as a private service behind Identity-Aware Proxy (IAP).
#
#   ./deploy/cloudrun.sh --project my-gcp-project --user-email you@example.com [--region us-central1] [--dry-run]
#
# Idempotent: re-running skips what already exists and rolls out a new revision with a freshly built image.
# Creates: the APIs below, an Artifact Registry repo, a Cloud Storage bucket for sessions (mounted at /data),
# a runtime service account with least-privilege roles, the Cloud Run service, IAP access for USER_EMAIL.
# No credentials are baked into the image: the app runs as the runtime service account (metadata server).
#
# Flags / env (flags win):
#   --project   PROJECT     Google Cloud project ID (default: gcloud config project)
#   --region    REGION      Cloud Run + Artifact Registry + bucket region (default: us-central1)
#   --service   SERVICE     Cloud Run service name (default: mlops-dilemma)
#   --bucket    BUCKET      sessions bucket (default: <project>-mlops-dilemma-sessions)
#   --user-email USER_EMAIL who gets IAP access (default: gcloud config account)
#   --member    MEMBER      more IAP access, repeatable: user:a@example.com, group:ai-ops@example.com or
#                           domain:example.com (everyone in it acts as the app's service account: trusted users only)
#   --repo      REPO        Artifact Registry repository (default: mlops-dilemma)
#   --tag       TAG         image tag (default: UTC timestamp)
#   --grant-act-as          also grant the runtime SA roles/iam.serviceAccountUser on itself (see below)
#   --hf-token-secret SECRET  optional Secret Manager secret holding a read-only Hugging Face token, exposed to the
#                           app as HF_TOKEN (gated models such as Llama: sizes, context, tokenizer). Create it first:
#                           printf %s "hf_..." | gcloud secrets create SECRET --data-file=- --project PROJECT
#   --dry-run               print the commands instead of running them (read-only lookups still run)
set -euo pipefail

PROJECT="${PROJECT:-}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-mlops-dilemma}"
BUCKET="${BUCKET:-}"
USER_EMAIL="${USER_EMAIL:-}"
REPO="${REPO:-mlops-dilemma}"
TAG="${TAG:-$(date -u +%Y%m%d-%H%M%S)}"
GRANT_ACT_AS="${GRANT_ACT_AS:-0}"
DRY_RUN="${DRY_RUN:-0}"
HF_TOKEN_SECRET="${HF_TOKEN_SECRET:-}"

MEMBERS=()

usage() { sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project) PROJECT="$2"; shift 2 ;;
    --region) REGION="$2"; shift 2 ;;
    --service) SERVICE="$2"; shift 2 ;;
    --bucket) BUCKET="$2"; shift 2 ;;
    --user-email) USER_EMAIL="$2"; shift 2 ;;
    --member)
      [[ "$2" =~ ^(user|group|domain|serviceAccount):[^[:space:]]+$ ]] || { echo "--member needs user:, group:, domain: or serviceAccount: ($2)" >&2; exit 2; }
      MEMBERS+=("$2"); shift 2 ;;
    --repo) REPO="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --grant-act-as) GRANT_ACT_AS=1; shift ;;
    --hf-token-secret) HF_TOKEN_SECRET="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

command -v gcloud >/dev/null || { echo "gcloud CLI not found" >&2; exit 1; }
gval() { local v; v="$(gcloud config get-value "$1" 2>/dev/null || true)"; [[ "$v" == "(unset)" ]] && v=""; echo "$v"; }
PROJECT="${PROJECT:-$(gval project)}"
USER_EMAIL="${USER_EMAIL:-$(gval account)}"
[[ -n "$PROJECT" ]] || { echo "set --project (or PROJECT)" >&2; exit 2; }
[[ -n "$USER_EMAIL" ]] || { echo "set --user-email (or USER_EMAIL)" >&2; exit 2; }
BUCKET="${BUCKET:-${PROJECT}-mlops-dilemma-sessions}"

SA_NAME="${SERVICE}-run"
SA_EMAIL="${SA_NAME}@${PROJECT}.iam.gserviceaccount.com"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/${SERVICE}:${TAG}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_UID=10001   # the non-root user in the Dockerfile; the bucket is mounted as this uid/gid

run() {   # print, then run unless --dry-run
  printf '+'; printf ' %q' "$@"; printf '\n'
  [[ "$DRY_RUN" == 1 ]] || "$@"
}
exists() {   # read-only existence check; in --dry-run a failed lookup is reported as "would create"
  "$@" >/dev/null 2>&1
}
step() { printf '\n== %s\n' "$*"; }

# Read-only: the project number names the IAP service agent.
PROJECT_NUMBER="$(gcloud projects describe "$PROJECT" --format='value(projectNumber)' 2>/dev/null || true)"
if [[ -z "$PROJECT_NUMBER" ]]; then
  [[ "$DRY_RUN" == 1 ]] || { echo "can't read project $PROJECT (does it exist, and can you see it?)" >&2; exit 1; }
  PROJECT_NUMBER="PROJECT_NUMBER"
fi
IAP_AGENT="service-${PROJECT_NUMBER}@gcp-sa-iap.iam.gserviceaccount.com"

echo "project=$PROJECT ($PROJECT_NUMBER) region=$REGION service=$SERVICE bucket=gs://$BUCKET"
echo "runtime SA=$SA_EMAIL image=$IMAGE IAP user=$USER_EMAIL$([[ $DRY_RUN == 1 ]] && echo '  [dry run]')"

step "Enable APIs"
# run, artifactregistry, cloudbuild: build and host the service. aiplatform: endpoints, Model Garden, Gemini.
# cloudbilling: live prices (Cloud Billing Catalog). iap: the sign-in gate. storage: the sessions bucket.
# serviceusage: the app's API check. cloudresourcemanager: the project picker. cloudquotas: GPU quotas.
run gcloud services enable --project "$PROJECT" \
  run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com aiplatform.googleapis.com \
  cloudbilling.googleapis.com iap.googleapis.com storage.googleapis.com serviceusage.googleapis.com \
  cloudresourcemanager.googleapis.com monitoring.googleapis.com cloudquotas.googleapis.com

step "Artifact Registry repository"
if exists gcloud artifacts repositories describe "$REPO" --location "$REGION" --project "$PROJECT"; then
  echo "exists: $REPO"
else
  run gcloud artifacts repositories create "$REPO" --repository-format docker --location "$REGION" \
    --project "$PROJECT" --description "mlops-dilemma images"
fi

step "Build the image with Cloud Build (source upload honours .gcloudignore: no runs/, .venv or credentials)"
run gcloud builds submit "$ROOT" --tag "$IMAGE" --project "$PROJECT"

step "Sessions bucket (Cloud Storage FUSE volume at /data)"
if exists gcloud storage buckets describe "gs://$BUCKET" --project "$PROJECT"; then
  echo "exists: gs://$BUCKET"
else
  run gcloud storage buckets create "gs://$BUCKET" --project "$PROJECT" --location "$REGION" \
    --uniform-bucket-level-access --public-access-prevention
fi

step "Runtime service account"
if exists gcloud iam service-accounts describe "$SA_EMAIL" --project "$PROJECT"; then
  echo "exists: $SA_EMAIL"
else
  run gcloud iam service-accounts create "$SA_NAME" --project "$PROJECT" \
    --display-name "mlops-dilemma Cloud Run runtime"
  # IAM can take a few seconds to see a new service account before bindings succeed.
  [[ "$DRY_RUN" == 1 ]] || sleep 10
fi

step "Runtime service account roles (least privilege)"
# roles/aiplatform.user: create/deploy/undeploy/delete Agent Platform endpoints and models, list and deploy
#   Model Garden models, send load-test predictions, and call Gemini (describe-a-workload, Gemini prices check).
# roles/serviceusage.serviceUsageConsumer: bill API quota to the project (x-goog-user-project on Cloud Billing
#   Catalog calls) and read which APIs are enabled (the app's Agent Platform / Billing API check).
# roles/browser: resourcemanager.projects.get/list, so the project picker can show this project
#   (grant it on other projects too if the app should list and use them).
# roles/monitoring.viewer: read Cloud Monitoring time series ("Import from Cloud Monitoring" in the Traffic
#   section: model invocation / token counts and endpoint prediction counts).
# roles/cloudquotas.viewer: read the Agent Platform serving quotas shown in the hardware drawer.
# The Cloud Billing Catalog (prices) needs no role; it only needs cloudbilling.googleapis.com enabled.
for role in roles/aiplatform.user roles/serviceusage.serviceUsageConsumer roles/browser roles/monitoring.viewer \
    roles/cloudquotas.viewer; do
  run gcloud projects add-iam-policy-binding "$PROJECT" --member "serviceAccount:$SA_EMAIL" --role "$role" \
    --condition None --quiet --format none
done
# roles/storage.objectAdmin on the sessions bucket only (not the project): read and write runs/ and cache/.
run gcloud storage buckets add-iam-policy-binding "gs://$BUCKET" --member "serviceAccount:$SA_EMAIL" \
  --role roles/storage.objectAdmin --format none
# roles/iam.serviceAccountUser (actAs) is only needed when an Agent Platform deploy runs the model as a custom
# service account. This app's deploys don't set one (Agent Platform uses its own service agent), so it is
# opt-in (--grant-act-as) and scoped to the runtime SA itself, never project-wide.
if [[ "$GRANT_ACT_AS" == 1 ]]; then
  run gcloud iam service-accounts add-iam-policy-binding "$SA_EMAIL" --project "$PROJECT" \
    --member "serviceAccount:$SA_EMAIL" --role roles/iam.serviceAccountUser --format none
fi

# Optional Hugging Face token from Secret Manager (mapped to HF_TOKEN below). The env var wins over a token saved in
# the app's Settings, which on Cloud Run would only live in the container's filesystem (lost on restart). Only the
# runtime SA may read this one secret (roles/secretmanager.secretAccessor on the secret, not the project).
SECRET_FLAGS=()
if [[ -n "$HF_TOKEN_SECRET" ]]; then
  step "Hugging Face token secret ($HF_TOKEN_SECRET -> HF_TOKEN)"
  run gcloud services enable secretmanager.googleapis.com --project "$PROJECT"
  run gcloud secrets add-iam-policy-binding "$HF_TOKEN_SECRET" --project "$PROJECT" \
    --member "serviceAccount:$SA_EMAIL" --role roles/secretmanager.secretAccessor --format none
  SECRET_FLAGS=(--set-secrets "HF_TOKEN=${HF_TOKEN_SECRET}:latest")
fi

step "Deploy the Cloud Run service (private: no unauthenticated access)"
# One always-on instance: benchmark jobs run as detached processes inside it and must keep CPU between
# requests (--no-cpu-throttling, --min-instances 1). --max-instances 1 keeps a single writer on the bucket.
# --timeout 3600: the job progress stream (SSE) is one long request.
run gcloud run deploy "$SERVICE" --project "$PROJECT" --region "$REGION" --image "$IMAGE" \
  --service-account "$SA_EMAIL" --no-allow-unauthenticated \
  --cpu 2 --memory 4Gi --no-cpu-throttling --min-instances 1 --max-instances 1 --concurrency 80 \
  --timeout 3600 --execution-environment gen2 \
  --add-volume "name=sessions,type=cloud-storage,bucket=${BUCKET},mount-options=uid=${APP_UID};gid=${APP_UID};implicit-dirs" \
  --add-volume-mount "volume=sessions,mount-path=/data" \
  --set-env-vars "DILEMMA_RUNS_DIR=/data/runs,DILEMMA_CACHE_DIR=/data/cache,GOOGLE_CLOUD_PROJECT=${PROJECT},DILEMMA_CLOUD=1,DILEMMA_IAP_AUDIENCE=/projects/${PROJECT_NUMBER}/locations/${REGION}/services/${SERVICE}" \
  --labels app=mlops-dilemma ${SECRET_FLAGS[@]+"${SECRET_FLAGS[@]}"}

step "Identity-Aware Proxy"
run gcloud beta run services update "$SERVICE" --project "$PROJECT" --region "$REGION" --iap
# Make sure the IAP service agent exists, then let it (and only it) invoke the service.
run gcloud beta services identity create --service iap.googleapis.com --project "$PROJECT" --format none
run gcloud run services add-iam-policy-binding "$SERVICE" --project "$PROJECT" --region "$REGION" \
  --member "serviceAccount:$IAP_AGENT" --role roles/run.invoker --format none
# Who may pass IAP. Teammates: re-run this command with their user:/group: member.
run gcloud beta iap web add-iam-policy-binding --project "$PROJECT" --resource-type cloud-run \
  --service "$SERVICE" --region "$REGION" --member "user:$USER_EMAIL" \
  --role roles/iap.httpsResourceAccessor --format none
for m in ${MEMBERS[@]+"${MEMBERS[@]}"}; do
  run gcloud beta iap web add-iam-policy-binding --project "$PROJECT" --resource-type cloud-run \
    --service "$SERVICE" --region "$REGION" --member "$m" --role roles/iap.httpsResourceAccessor --format none
done

step "Done"
if [[ "$DRY_RUN" == 1 ]]; then
  echo "dry run: nothing was changed"
else
  URL="$(gcloud run services describe "$SERVICE" --project "$PROJECT" --region "$REGION" --format 'value(status.url)')"
  echo "Open $URL (sign in as $USER_EMAIL; IAP access can take a minute to propagate)"
fi
