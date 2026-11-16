#!/usr/bin/env bash
# Remove what deploy/cloudrun.sh created.
#
#   ./deploy/teardown-cloudrun.sh --project my-gcp-project [--region us-central1] [--all] [--dry-run]
#
# Default: deletes only the Cloud Run service (sessions in the bucket are kept).
# --all: also deletes the sessions bucket (ALL saved sessions), the runtime service account and its project
#        role bindings, and the Artifact Registry repository (all images).
# Check the app's Benchmark tab first: Agent Platform endpoints a benchmark left running are billed separately and
# are NOT removed here (`dilemma teardown -c <session>/workload.yaml`, or delete them in the console).
# Flags / env: --project PROJECT, --region REGION, --service SERVICE, --bucket BUCKET, --repo REPO, --all, --dry-run
set -euo pipefail

PROJECT="${PROJECT:-}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-mlops-dilemma}"
BUCKET="${BUCKET:-}"
REPO="${REPO:-mlops-dilemma}"
ALL=0
DRY_RUN="${DRY_RUN:-0}"

usage() { sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --project) PROJECT="$2"; shift 2 ;;
    --region) REGION="$2"; shift 2 ;;
    --service) SERVICE="$2"; shift 2 ;;
    --bucket) BUCKET="$2"; shift 2 ;;
    --repo) REPO="$2"; shift 2 ;;
    --all) ALL=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

command -v gcloud >/dev/null || { echo "gcloud CLI not found" >&2; exit 1; }
if [[ -z "$PROJECT" ]]; then
  PROJECT="$(gcloud config get-value project 2>/dev/null || true)"; [[ "$PROJECT" == "(unset)" ]] && PROJECT=""
fi
[[ -n "$PROJECT" ]] || { echo "set --project (or PROJECT)" >&2; exit 2; }
BUCKET="${BUCKET:-${PROJECT}-mlops-dilemma-sessions}"
SA_EMAIL="${SERVICE}-run@${PROJECT}.iam.gserviceaccount.com"

run() { printf '+'; printf ' %q' "$@"; printf '\n'; [[ "$DRY_RUN" == 1 ]] || "$@"; }
exists() { "$@" >/dev/null 2>&1; }

if exists gcloud run services describe "$SERVICE" --project "$PROJECT" --region "$REGION"; then
  run gcloud run services delete "$SERVICE" --project "$PROJECT" --region "$REGION" --quiet
else
  echo "no service $SERVICE in $REGION"
fi

if [[ "$ALL" == 1 ]]; then
  if exists gcloud storage buckets describe "gs://$BUCKET" --project "$PROJECT"; then
    run gcloud storage rm --recursive "gs://$BUCKET" --project "$PROJECT"
  else
    echo "no bucket gs://$BUCKET"
  fi
  for role in roles/aiplatform.user roles/serviceusage.serviceUsageConsumer roles/browser roles/monitoring.viewer \
      roles/cloudquotas.viewer; do
    run gcloud projects remove-iam-policy-binding "$PROJECT" --member "serviceAccount:$SA_EMAIL" --role "$role" \
      --condition None --quiet --format none || true
  done
  if exists gcloud iam service-accounts describe "$SA_EMAIL" --project "$PROJECT"; then
    run gcloud iam service-accounts delete "$SA_EMAIL" --project "$PROJECT" --quiet
  else
    echo "no service account $SA_EMAIL"
  fi
  if exists gcloud artifacts repositories describe "$REPO" --location "$REGION" --project "$PROJECT"; then
    run gcloud artifacts repositories delete "$REPO" --location "$REGION" --project "$PROJECT" --quiet
  else
    echo "no repository $REPO"
  fi
fi
[[ "$DRY_RUN" == 1 ]] && echo "dry run: nothing was changed"
exit 0
