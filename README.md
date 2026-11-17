# mlops-dilemma

**When does self-hosting an open model pay off vs a managed model API (MaaS)?**
This repo answers it for *your* traffic: it deploys the model on an Agent Platform endpoint, load-tests it
with your prompts (or synthetic ones shaped like them), and computes what a production fleet would
cost against pay-per-token MaaS — including the cases where MaaS can't serve your peak.

Companion to the blog post [*You Want to Self-Host Your LLM. Here's What the Napkin Math Leaves Out.*](https://medium.com/google-cloud/you-want-to-self-host-your-llm-heres-what-the-napkin-math-leaves-out-21006f7492fa)
(Google Cloud Community on Medium); its example workload and run are in `blog/example/`.
How every number is calculated: [docs/methodology.md](docs/methodology.md).

## What it does

```
dilemma ui        launch the interactive web application (wizard + live benchmark + Gemini advisor)
dilemma plan      size the model, list GPU/TPU configs that fit, rank by estimated $/token, pick one
dilemma prompts   your prompts (JSONL), custom system prompts, or synthetic ones from a short description
dilemma deploy    1-replica Agent Platform dedicated endpoint (Model Garden vLLM container)
dilemma load      closed-loop load test: pre-flight check, concurrency sweep, streaming, unique prompts
dilemma preflight the pre-flight check alone (one long answer + one long prompt) on a deployed endpoint
dilemma analyze   peak input tokens/sec per GPU/TPU within your latency budget
dilemma calibrate short-only run on the same endpoint -> per-class throughput for long prompts
dilemma report    cost model -> report.md + a single-file interactive dashboard.html
dilemma teardown  delete the endpoint
dilemma auto      all of the above, end to end (tears down unless --keep)
dilemma estimate  report without a benchmark, from an assumed throughput (what-if)
dilemma simulate  replay a week request by request: minimum fleet + monthly cost per scenario
dilemma run-job   benchmark steps as a durable background job (what the web app launches)
dilemma reference-entry  print a catalog.REFERENCE_BENCHMARKS entry from a reference run
```

The cost model assumes what most teams hit in practice: **GPU capacity is scarce, so you can't
autoscale** — the fleet is sized for your busiest hour and always on. A MaaS endpoint absorbs the top
of the peak up to its rate limit. Five pricing options are compared: Agent Platform on demand, Agent Platform on a
Compute Engine reservation with 1-/3-year CUDs, and self-managed GKE with 1-/3-year CUDs.

The **baseline** is honest about rate limits too: "MaaS only" means open-model MaaS up to its limit, plus the
traffic above it on a Gemini tier (`hybrid.failover_target`, Flash-Lite by default), because that is what you would
have to buy. Three hybrid levers (`hybrid` in the config, the rail's Routing section in the app):

- **Peak overflow** (`peak_overflow_target`): where the busiest hours spill. Default `open_maas`: the fleet is sized
  so the spill fits under the MaaS limit, no Gemini at peaks. When sending peaks to Gemini would be at least 3%
  cheaper, the app says so and offers "Allow Gemini at peaks" (fewer GPUs; the optimizer and the GPU slider use the
  same rule).
- **Long-prompt routing** (`route_long` + `route_long_target`): prompts over 32K go to MaaS / Gemini instead of the
  fleet.
- **Failover** (`failover_target`): an outage is served pay-per-token instead of by idle standby GPUs.

## Quick start

Prerequisites: a GCP project with Agent Platform enabled, GPU quota for prediction in your region,
`gcloud auth application-default login`, the Cloud Billing API enabled (for live prices),
and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/omribt1/mlops-dilemma.git && cd mlops-dilemma
uv sync

# Auto mode: model + a few traffic numbers -> everything else is chosen/generated for you
uv run dilemma init --auto -o my.yaml        # edit project, model, traffic, MaaS price
uv run dilemma auto -c my.yaml               # ~1-2 hours incl. deploy; endpoint deleted at the end

# Or step by step (manual mode)
uv run dilemma init -o my.yaml
uv run dilemma plan -c my.yaml
uv run dilemma prompts -c my.yaml
uv run dilemma deploy -c my.yaml
uv run dilemma load -c my.yaml               # starts with a ~2-minute pre-flight check
uv run dilemma analyze -c my.yaml
uv run dilemma report -c my.yaml             # runs/<name>/report.md and dashboard.html
uv run dilemma teardown -c my.yaml
```

> **Cost note.** A benchmark endpoint bills by the hour while it exists (e.g. ~$6/h for one
> RTX PRO 6000, ~$14/h for one H100 on Agent Platform). `auto` deletes it at the end; if you deploy by hand,
> run `teardown` when you're done.

## The web app

`uv run dilemma ui` opens the workload on a rail (Traffic, System prompt, Prompts, Pricing, Routing, Advanced) next
to the tabs Costs, Resilience, Long prompts, Simulation, Benchmark and Config. The cost model runs in the browser,
so every change updates the numbers instantly.

- **Sign-in and projects.** It uses Application Default Credentials and shows which identity it resolved to; "Sign
  in" completes a `gcloud` login in the browser (handy on a remote VM). The project picker lists the projects your
  account can see. The chosen project is the app's own: it is used as the resource and quota project of every API
  call, and your `gcloud` configuration is never changed. Switching accounts refreshes the catalogs and re-checks
  the project.
- **Durable benchmarks.** A benchmark runs as its own background process (`dilemma run-job`) and writes
  `runs/<session>/job.json` + `job_events.jsonl`: closing the browser or restarting the server doesn't stop it, and
  the live progress replays when you come back. Stop cancels at the next safe point and tears down the endpoint the
  job deployed; a job whose process died is marked interrupted, with an offer to tear down a leftover endpoint.
- **Estimates are labelled.** Throughput you haven't measured carries its source (reference, scaled or rough, see
  below), and a first-use notice explains why a benchmark with your prompts is what to decide on.

## Describing your workload

See [`examples/workload.yaml`](examples/workload.yaml) (every field, commented) and
[`examples/auto.yaml`](examples/auto.yaml) (the minimum).

| Section | What you provide |
|---|---|
| `model` | Model Garden id, Hugging Face id (for size + tokenizer), precision (`bf16` / `fp8` / `int4` / `mxfp4`; only the ones the model has a checkpoint for) |
| `deployment` | `auto` (planner picks) or `manual` machine / accelerator / count |
| `traffic` | an hourly CSV for a week, **or** peak / average / floor + busy hours + weekend factor, **or** a named shape (`preset`: `us_core_hours`, `us_europe`, `follow_the_sun`, `steady`) at your weekly average |
| `requests` | avg input & output tokens, % of requests > 32K / > 128K, fixed system prompt size, cache hit, thinking share; optional `cached_input_tokens` (cached tokens per request, e.g. resent chat history; overrides system prompt × hit rate) |
| `maas` | the managed API you compare against: $/1M input, output, cached; rate limit (req/s) |
| `pricing` | `tier`: the fleet pricing for the headline (`auto` = cheapest, or e.g. `vertex_cud_1y`); optional $/GPU-hour overrides |
| `hybrid` | peak overflow, long-prompt routing and failover targets: `open_maas`, a role alias (`gemini_flash_lite`, `gemini_flash`, `gemini_pro`, `gemini_4`) or any live Gemini model id |
| `constraints` | latency budget, burst headroom, resilience multiplier |
| `throughput` | optional measured per-chip input tok/s for 32K–128K and > 128K prompts (`medium_tps`, `large_tps`); a calibrated benchmark fills them in |
| `prompts` | `file` (your prompts, see below) or `synthetic` (from a one-paragraph domain description) |

**Import real traffic.** In the app, upload traffic exports with timestamps, for example the CSVs Cloud
Monitoring Metrics Explorer downloads ("Model requests per second (QPS)", "Token count by model"), `timestamp,value`
CSVs (ISO, RFC 3339, epoch or `YYYY-MM-DD HH:MM`, read as UTC) or JSON records. The importer guesses what each
series measures, resamples it by time into hourly means, builds a typical week (≥ 7 days: mean per weekday-hour;
1–6 days: weekday and weekend curves; under a day: you choose to use it as the daily shape or only for its level),
and combines files: requests/s + tokens/s give the tokens per request, input + output give the output ratio, a
token count without an input/output split is treated as input + output. A review sheet shows the mapping,
coverage, the resulting week against the current one and the derived values; nothing applies until you confirm.

**Or import straight from Cloud Monitoring.** "Import from Cloud Monitoring" in the Traffic section lists the
models (Gemini, partner models such as Claude, open models as a service) and dedicated endpoints that served
requests in the last 7, 14 or 28 days and reads their hourly requests and input / output tokens
(`aiplatform.googleapis.com/publisher/online_serving/model_invocation_count` and `token_count`, or
`prediction/online/prediction_count` for endpoints). Prompt-cache reads and writes count as input tokens. The
signed-in identity needs **Monitoring Viewer (`roles/monitoring.viewer`)** on the project.

**Use real prompts if you can.** Throughput depends on input size, output length and how much of
each prompt is a shared, cacheable prefix. See "Bring your own prompts" below.

### Bring your own prompts

In the app, open **Prompts → Prompts for the benchmark → My prompts** and upload files (several at once) or paste
text. The set is stored in the session's folder (`runs/<session>/uploaded_prompts.jsonl`) and sent only to your
benchmark endpoint; `prompts.source: file` + `prompts.file: uploaded_prompts.jsonl` point the benchmark at it.
"Remove (back to synthetic)" deletes it. Example sessions are read-only: uploading to one saves a copy first.

| Format | What is read |
|---|---|
| `.jsonl` | OpenAI chat records (`messages`, also batch lines with a `body`), or objects with `prompt` / `input` / `text` / `question` (+ optional `system`, `max_tokens`, `response_format`, `chat_template_kwargs`, `input_tokens` / `output_tokens`) |
| `.json` | an array of strings, an array of such objects, or `{"prompts": [...]}` |
| `.csv` / `.tsv` | a `prompt` / `text` / `input` column (+ optional `system`, `max_tokens`, token-count columns) |
| `.txt` / `.md` | prompts separated by lines holding only `---` (or blank-line separated blocks, or one per line) |

```jsonl
{"messages": [{"role": "system", "content": "You are a playful storyteller."}, {"role": "user", "content": "Write a limerick about a robot baker."}], "max_tokens": 300}
{"prompt": "List three ways to keep a sourdough starter alive.", "system": "Answer briefly.", "output_tokens": 120}
```

```text
Summarize the plot of a novel about a lighthouse keeper who adopts a seagull.
---
Translate 'the purple giraffe sings at dawn' into formal Latin.
```

Tokens are counted with the model's tokenizer when it loads (Hugging Face; gated models need the token in
Settings), else estimated at ~3.5 characters per token (labelled). Empty and duplicate prompts are dropped; limits
50 MB and 20,000 prompts. The summary card shows the count, input sizes (median / p95 / max, share over 32K and
128K), whether answer lengths came with the file, how many prompts carry their own system prompt, a preview and
warnings (prompts repeat across load-test levels when there are fewer prompts than requests, with the repeat factor;
prompts longer than the model's context). A prompt without `max_tokens` or an output count gets the workload's
"Output tokens (answer)"; the configured thinking share of prompts (deterministic) runs with thinking on and
"Thinking tokens" more. "Update request sizes from my prompts" offers the measured average input, shares over
32K / 128K, medium / large means (and average output when most prompts brought one) for review; nothing changes
until you apply.

The benchmark then uses your set end to end: the prompts step imports it (custom system prompts are added to
prompts without one), every load-test level gets the same size-class / thinking mix of your prompts (repeats
reported), the per-class calibration runs a short-only pass on your prompts of 32K tokens or less (skipped with a
note when none of yours is over 32K), and the Benchmark tab and `report.md` say "Prompts: your N prompts (uploaded
<date>)" or "synthetic". API: `POST /api/sessions/{name}/prompts` (multipart `files` + `text_mode`, or JSON
`{text, mode}`), `GET` (summary), `DELETE` (back to synthetic).

**One uploader per section.** Traffic takes time series and request logs (reviewed before applying) and workload
descriptions (PDF / DOCX / TXT / MD, read by Gemini); System prompt takes system-prompt files; Prompts takes your
prompt set. A file of the wrong kind gets a message pointing to the right section, and a file that holds several
kinds (e.g. a request log with timestamps, prompts and token counts) offers its other parts ("also has prompts:
use them for the benchmark?").

## Where the data comes from

The tool reads live sources wherever one exists and keeps its own tables only as a labelled fallback:

| Data | Live source | Fallback (labelled in the app) |
|---|---|---|
| Which open models are deployable, their Hugging Face ids, verified machine shapes | Model Garden (publisher model listing, deploy recipes) | the last listing on disk, then a small seed catalog |
| Model size, MoE active parameters, context window, native precision (BF16 / FP8 / INT4 / NVFP4 / MXFP4), MXFP4 weight size (per-dtype safetensors counts), release month, license | Hugging Face Hub: model API (safetensors parameter counts, createdAt, tags, card) + `config.json` | the rule table in `modelinfo.py` ("built-in estimate") |
| Which accelerators, machine shapes and counts exist in your region | Compute Engine API (`acceleratorTypes`, `machineTypes`; TPUs are machine types too) | `catalog.FALLBACK_MACHINES` |
| Per-chip memory, TFLOPS, bandwidth, interconnect | none (no API publishes them): `catalog.CHIP_SPECS`, from vendor data sheets | accelerators the API lists but the table lacks are shown as "no performance data" and not estimated |
| GPU / TPU prices | Cloud Billing Catalog | `catalog.FALLBACK_PRICES` |
| Your project's GPU / TPU serving quota per region ("Custom model serving … per region") | Cloud Quotas API (needs `cloudquotas.googleapis.com` enabled and `cloudquotas.quotas.get`, e.g. Viewer) | none: the hardware drawer says why it can't show quotas |
| Which Gemini models exist, their prices (input / output / cached, >200K tier, batch, regional) and launch stage | Cloud Billing Catalog (every Gemini text SKU) + Model Garden publisher models (ids, GA / preview / deprecated) | `catalog.GEMINI_MODELS` (the four role aliases) |

**Gemini models are discovered, not listed by hand.** Every Gemini text model the Cloud Billing Catalog prices per
token appears in the app (e.g. Gemini 4 Argon, Carbon and Hydrogen the day their SKUs show up), joined with Model
Garden's launch stage: previews are labelled, deprecated models hidden unless you picked one. Routing, peak overflow
and failover accept any model or a role alias that follows the newest model of its kind: "Gemini Flash-Lite" (newest
GA Flash-Lite), "Gemini Flash", "Gemini Pro" (a preview of a newer generation beats an older GA) and "Gemini 4
flagship" (the newest generation's highest-priced model). If a model you pinned disappears, the app switches to its
role alias and says so. The Gemini list is cached for a day, refreshed in the background, and has a Refresh button in
the rail's Pricing → Gemini on Agent Platform section.

Hugging Face metadata is cached per model for 7 days and refreshed in the background (no token needed, even for
most gated models; a Hugging Face token also unlocks gated `config.json` and `tokenizer.json` files such as Llama's,
see [Hugging Face token](#hugging-face-token)). The Compute Engine listing
is cached per region for a day. The model drawer says where each model's figures come from ("Data: Hugging Face,
updated <date>" or "built-in estimate", per field on hover), and the hardware drawer says which accelerators your
region doesn't offer. Active parameters of mixture-of-experts models come from one formula over the config
(total − inactive experts − multi-token-prediction layers), checked against published sizes such as Kimi K2 (32B),
DeepSeek-V3 (37B), Qwen3-30B-A3B (3B), Mixtral 8x7B (13B), gpt-oss-120b (5B) and Gemma 4 26B-A4B (4B); when it
disagrees with a size published in the model's name by more than 15%, the published size is used.

### Hugging Face token

Gated repos (Llama, some Gemma variants) answer the Hub's model API without a token but refuse `config.json` and
`tokenizer.json`, so their context length and active parameters fall back to built-in values and prompts are counted
with a character heuristic. Add a read-only token (<https://huggingface.co/settings/tokens>) under **Settings** (the
gear in the header): the app checks it with Hugging Face (`whoami-v2`), stores it on the machine running the app in
`~/.config/aiops-dilemma/hf_token` (mode 0600, directory 0700; `$XDG_CONFIG_HOME` or `DILEMMA_CONFIG_DIR` moves it),
then refetches the metadata that failed for want of a token. It is sent only to huggingface.co (model metadata,
`config.json`, tokenizers), never written to sessions, `workload.yaml`, caches or exports, and the API only ever
returns a hint (`hf_…abcd`). Precedence: the `HF_TOKEN` environment variable, then `HUGGING_FACE_HUB_TOKEN`, then
the saved file; a token set in the environment is shown read-only in Settings. On a read-only filesystem a token
saved in Settings is kept in memory until the server restarts.

## How auto mode chooses a GPU

1. **Size**: total parameters from Hugging Face (or `params_b`); 2 bytes each at BF16, 1 at FP8, ~0.6 at INT4;
   MXFP4 (OpenAI gpt-oss, published only in MXFP4) from the checkpoint's per-dtype counts on Hugging Face: 4.25 bits
   per expert parameter plus the layers kept in BF16 (gpt-oss-120b ≈ 65 GB, gpt-oss-20b ≈ 14 GB), else ~0.55 byte
   per parameter, labelled "estimated". MXFP4 runs on GPUs only: Model Garden's verified gpt-oss recipes (live)
   and the GPUs vLLM documents (H100, H200, B200; A100 with Triton attention and Marlin kernels) are offered as
   supported, others (L4) are labelled "untested". vLLM reads the checkpoint's quantization config (no
   `--quantization` flag).
2. **Fit**: weights ≤ 75% of usable GPU memory (90% of VRAM minus ~2 GB/GPU), smallest GPU count per type, among
   the machine shapes the Compute Engine API lists for your region.
3. **Price**: live per-GPU-hour prices from the Cloud Billing catalog for your region (GPU + its share of
   vCPU/RAM + Agent Platform management fee).
4. **Rank**: estimated $/token = price ÷ a speed proxy blending dense TFLOPS (input processing) and memory
   bandwidth (output generation), with tensor-parallel efficiency. Model Garden verified configs first.
5. **Benchmark** the top candidate — the measurement replaces the estimate.

### Throughput estimates and reference benchmarks

> **Throughput figures you haven't measured are estimates.** They may not represent your workload: prompt sizes,
> thinking and output length, long prompts and caching change throughput by several times (unbudgeted thinking cut
> it about 4× in our runs). For a decision, run a benchmark with your own prompts and traffic.

Until you benchmark, each accelerator's per-chip throughput (a short-prompt rate) comes from one of three tiers,
shown as a tag next to every figure:

| Tag | When | How |
|---|---|---|
| **Reference est.** | a reference benchmark exists for this GPU × count, model and precision | its measured rate, adjusted to your output length and prompt size |
| **Scaled est.** | a reference benchmark exists for this GPU with another model, precision or count | its rate × the spec formula's ratio (your model over the reference model on that GPU), same workload adjustment |
| **Rough estimate** | no reference benchmark for this GPU | the spec formula (TFLOPS + memory bandwidth); can be off by several times |

The workload adjustment is the formula's ratio at your workload over the reference workload on your model: mostly
the decode term (output + thinking tokens per input token), plus prompt size and system-prompt cache share.

**Reference benchmarks** (`catalog.REFERENCE_BENCHMARKS`) are measured by this project on one fixed workload,
[`examples/reference-workload.yaml`](examples/reference-workload.yaml) ("reference-v1": Gemma 4 26B-A4B FP8,
~5,000-token prompts incl. a 1,800-token system prompt, ~500-token JSON answers, thinking off, nothing over 32K, the
pinned serving image). They let GPUs be compared with each other; they are not a promise for your workload. To add one:

```bash
mkdir -p runs/ref-h100 && cp examples/reference-workload.yaml runs/ref-h100/workload.yaml
# set name: ref-h100 and deployment.accelerator_type / accelerator_count / machine_type
# (80-96 GB GPUs: loadtest.concurrency [16, 32, 64, 128])
uv run dilemma auto -c runs/ref-h100/workload.yaml             # deploy, pre-flight, sweep, analyze, tear down
uv run dilemma reference-entry -c runs/ref-h100/workload.yaml  # prints the entry; warns if the peak sat at the top level
```

Paste the printed entry into `catalog.REFERENCE_BENCHMARKS` (or point `DILEMMA_REFERENCE_BENCHMARKS` at a JSON list
of entries to try it first). When the chosen GPU has no Model Garden recipe for the model (e.g. L4 or A100 for
Gemma 4), the deploy reuses the model's recipe from another machine with the tensor-parallel degree adjusted, so
every GPU runs the same weights and flags.

**Quota.** The hardware drawer's Quota column shows your project's Agent Platform serving quota for each
accelerator in the chosen region (Cloud Quotas API, cached 30 minutes): whether it covers the benchmark (one replica)
and the planned fleet, and how many more to request when it doesn't. It is a limit, not what is free: running
endpoints count against it. TPU quotas count cores (one per v5e / v6e chip); a fleet on GKE uses Compute Engine GPU
quota instead.

**Level comparability.** Every concurrency level of a sweep gets its own prompt sample with the same mix (shares of
prompts ≤ 32K / 32K–128K / > 128K, thinking on/off, and size quartiles), without reusing prompts across levels while
the pool lasts. The report and the Benchmark tab show each level's mix.

### Long prompts: per-class calibration

A load test measures one blended rate for your mix, but long prompts cost far more GPU time per token than short
ones. When the workload has prompts over 32K, the benchmark adds a **calibrate** step: a short-only run on the same
endpoint gives the short-prompt rate S, and the mixed run's peak level (its token shares by size and blended rate)
gives one long-prompt factor that scales the default slowdowns (3× for 32K–128K, 4× for > 128K; about 3× and 4.1×
measured on an RTX PRO 6000). The result is stored as `bucket_tps` in `measured.json` and used by the cost model, so
routing long prompts away is priced with measured rates. Without a calibrated run, `throughput.medium_tps` /
`large_tps` override the defaults, else the defaults apply. The sweep chart labels each point by run:
`c16 · mixed`, `c16 · short`, `c16 · long`.

## What the report tells you

- **Verdict**: cheapest option, GPUs, monthly cost, % of MaaS-only, average GPU utilization.
- **Break-even utilization** per pricing option — how busy a GPU must be to beat MaaS on the same tokens.
- **Sensitivity**: off-peak growth (busiest hour fixed), overall growth, MaaS rate limit, throughput error, resilience.
- **Dashboard**: one HTML file with sliders for all of the above — send it to stakeholders as-is.

### Serving image and pre-flight check

Benchmark endpoints run vLLM in a Model Garden container. By default the tool uses the image from Model Garden's
verified recipe for your model and machine, but that image changes over time, and some versions cut long requests:
streams drop after about 30 s ("peer closed connection"), the server then fails its health checks
("503 no healthy upstream"), even when the endpoint was created with a long inference timeout. A sweep against
such an endpoint fails for every long request while you pay for the GPU.

- **Choose the image** in the web app (Advanced → Serving container → Serving image: the verified recipe, the pinned
  image, or a custom Artifact Registry, Container Registry or Docker Hub URI), or in the config:
  ```yaml
  deployment:
    image: us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/pytorch-vllm-serve:gemma4   # null = verified recipe
  ```
  The verified recipe's arguments are kept; only the image changes. The pinned image (`catalog.SERVING_IMAGES`)
  is one this tool's maintainers ran long requests on, with the date of that test.
- **Pre-flight check.** Before the first concurrency level, every benchmark sends one streamed request forced to
  generate for 45–90 s (half your latency budget; `min_tokens` + `ignore_eos`) and one long prompt (~100K tokens
  if your workload has prompts over 32K, else your largest prompt). If either is cut, times out or gets a 5xx, the
  run stops with a message that names the image, and the endpoint is torn down: about two minutes of GPU time
  instead of a full sweep. Turn it off with `loadtest.preflight: false`.
- **Provenance.** `endpoint.json`, `measured.json` and the report record the serving image, the final container
  arguments and the endpoint's inference timeout ("Measured with `pytorch-vllm-serve:gemma4`, vLLM args …").
- **Re-test an image** (e.g. a newer pinned candidate): set `deployment.image`, then
  ```bash
  uv run dilemma deploy -c my.yaml
  uv run dilemma preflight -c my.yaml        # the pre-flight check alone, against runs/<name>/endpoint.json
  uv run dilemma teardown -c my.yaml
  ```

### Custom serving containers and tuned configs

Bring your own serving setup and measure it on your traffic: your image, command, vLLM arguments, environment,
routes, shared memory, and a Python patch that runs when the server starts (for example a tuned fused-MoE tile
schedule or custom Triton kernels). In the web app: **Advanced → Serving container**. In the config, all optional:

```yaml
deployment:
  image: docker.io/vllm/vllm-openai:v0.31.0   # or Artifact Registry / Container Registry
  command: [vllm, serve]                      # empty = the image's entrypoint
  args_mode: replace                          # append (default): extra_args merge into the recipe's arguments, and a
                                              # flag given here replaces the recipe's; replace: the full argument list
  extra_args:
    - google/gemma-4-26B-A4B-it
    - --max-model-len=262144
    - --max-num-seqs=256
    - '--speculative-config={"method":"ngram_gpu","num_speculative_tokens":2}'
  env: {HF_HOME: /dev/shm/hf_cache}
  routes: {predict: /v1/chat/completions, health: /health, port: 8080}   # replace mode defaults to these
  shared_memory_mb: null                      # empty: 16 GB, or ~43% of the machine's memory when HF_HOME is in /dev/shm
  pass_hf_token: true                         # gated weights from Hugging Face (not available on the Cloud Run deployment)
  sitecustomize: null                         # Python run at server start; needs `command`
```

- **Docker Hub images.** Agent Platform only runs images from Artifact Registry. For a `docker.io/...` image the first
  deploy creates an Artifact Registry remote repository, `dilemma-dockerhub`, in the endpoint's region (a pull-through
  cache that deletes cached versions after 14 days), pulls the image's manifest through it, and deploys from there.
  Without permission to create repositories, the error gives the one `gcloud` command to run.
- **How the benchmark reaches it.** Agent Platform's chat-completions route only works for Model Garden containers
  (a custom one answers 404 `UNIMPLEMENTED`). A container that serves vLLM's OpenAI API itself (`args_mode: replace`
  or a `command`, predict route `/v1/chat/completions`) is load-tested through `:streamRawPredict`, which passes each
  request to the predict route and streams the answer back, with the served model name (`--served-model-name`, else
  the model ID) instead of `default`.
- **Weights from Hugging Face.** With `args_mode: replace` and a Hugging Face model ID, vLLM downloads the weights.
  The container disk holds about 50 GB, so larger checkpoints go to memory (`HF_HOME=/dev/shm/hf_cache`, shared memory
  sized automatically). `pass_hf_token` sets your token as `HF_TOKEN` on the benchmark model: anyone who can read
  Agent Platform models in the project can see it until the run deletes the model.
- **Import a tuned config.** `alpha_inference` (Google Cloud AlphaEvolve tuning vLLM for a GPU, model and workload)
  writes its winner as `best_candidate.py`. Import it with **Import tuned config…** in the app, or:
  ```bash
  uv run dilemma import-tuned -c my.yaml path/to/best_candidate.py [--image us-docker.pkg.dev/...] [--dry-run]
  ```
  The file is parsed, never run: `get_candidate_config()` must return a literal dict. Its `vllm_args` become the full
  argument list, `env_vars` the environment, and `moe_tuned_config` plus `custom_kernel_code` a `sitecustomize`
  patch (the same `register_vllm_patch` hook its kernels use). Speculative decoding (`--speculative-config`) is
  left out: on vLLM 0.31 it fails under load together with structured JSON output ("grammar rejected tokens"); add it
  back under vLLM arguments if your prompts don't use a JSON schema. The import warns when the tuned config serves another
  model, precision or GPU count, includes kernel code, or caps `--max-model-len` below what your longest prompts
  need. A config tuned on someone else's traffic says little about yours: tune on your own traffic, then benchmark
  the result here to price it.
- **Provenance.** `endpoint.json`, `measured.json` and the report record the command, environment (never the token),
  routes, shared memory, the patch's size and sha256, and the tuned config's title. `dilemma reference-entry` warns
  that such a run isn't a reference benchmark.

## Lessons this tool encodes

- **Peak-to-average drives cost more than volume.** A fleet sized for the peak runs at 25–35% average
  utilization; growth in off-peak hours (other time zones) is almost free on GPUs and linear on MaaS.
- **Long prompts can eat most of your GPU time.** A few percent of requests above 32K tokens can take
  over half of the GPU time — routing them to MaaS can cut the fleet by more than half.
- **FP8 gains less than you'd expect when outputs are short** (~1.1–1.4× in our runs): it mostly speeds up
  output generation, which short-output workloads don't spend much time on.
- **Reasoning models without a thinking budget spend the whole output cap.** In a sample run of Gemma 4 with
  thinking on for 30% of requests, 89% of those requests used all ~3,250 output tokens and returned truncated JSON.
  The load test reports these per level ("answered but unusable"); budget output tokens from measurements.
- **Check the serving image before measuring.** A Model Garden verified image cut every request at ~30 s;
  every benchmark now starts with a pre-flight check (one long answer, one long prompt).
- **Fail over to pay-per-token, not to idle GPUs.** Always-on zone redundancy adds ~50% to the GPU bill;
  pre-contracted per-token endpoints cost nothing until an incident.

## Deploy to Cloud Run (private, behind IAP)

Run the web app as a private Cloud Run service your team opens in the browser. Sign-in is handled by Identity-Aware Proxy (IAP); only the accounts you grant can reach it.

> **Permissive by design: grant access only to trusted AI-Ops users.** The deployment is gated by IAP, but once
> inside, every user acts as the app's service account, which can deploy and delete Agent Platform endpoints, call
> Gemini and read Cloud Monitoring in the project, and can see and change every saved session. There are no per-user
> permissions or private sessions. Grant IAP access (users, groups or a whole domain) only to the AI-Ops engineers
> you'd trust with GPU spend in this project, and set a budget alert on it. It's built for team demos and
> evaluations, not as a multi-tenant or production service.

```bash
./deploy/cloudrun.sh --project my-gcp-project --user-email you@example.com --dry-run   # print every command first
./deploy/cloudrun.sh --project my-gcp-project --user-email you@example.com             # deploy (idempotent)
```

What it creates:

- **The Cloud Run service** `mlops-dilemma`: 2 vCPU, 4 GiB, always-on CPU, exactly one instance (benchmark jobs run inside it and need a single writer), 60-minute request timeout for live progress streams.
- **A runtime service account** `mlops-dilemma-run`, with only the roles the app uses:
  - Agent Platform User: Model Garden, endpoints, Gemini;
  - Service Usage Consumer;
  - Browser: the project picker;
  - Monitoring Viewer: "Import from Cloud Monitoring";
  - Cloud Quotas Viewer: the GPU quotas in the hardware drawer;
  - Storage Object Admin on the sessions bucket only;
  - Artifact Registry Reader on `dilemma-dockerhub` only, a Docker Hub pull-through repository the script creates
    in the service's region for custom serving images.

  Benchmarks deploy GPU endpoints *as this service account*, not as the person using the app. Pass `--grant-act-as` if your project needs `iam.serviceAccountUser` for Agent Platform deploys.
- **A Cloud Storage bucket** `<project>-mlops-dilemma-sessions`, mounted at `/data` for sessions and caches.
- **An Artifact Registry repository** for the image, built with Cloud Build.
- **IAP access** for `--user-email`.

To grant more access at deploy time, repeat `--member` (`user:`, `group:` or `domain:`); IAP access belongs to the
service, so a teardown removes it and the next deploy re-applies these:

```bash
./deploy/cloudrun.sh --project my-gcp-project --user-email you@example.com --member group:ai-ops@example.com
```

To add a teammate to a running deployment:

```bash
gcloud beta iap web add-iam-policy-binding --resource-type cloud-run --service mlops-dilemma \
  --region us-central1 --member user:teammate@example.com --role roles/iap.httpsResourceAccessor
```

**Cost.** One always-on instance at 2 vCPU / 4 GiB is roughly $115 a month at us-central1 list prices (instance-based billing; free tier and committed-use discounts lower it). GPU endpoints created by benchmarks are billed separately, by the hour, until they're torn down.

**Hugging Face token (optional).** Store a read-only token in Secret Manager and pass `--hf-token-secret NAME`: the
script grants the runtime service account access to that one secret and maps it to `HF_TOKEN` (a token saved in the
app's Settings would only live in the container's filesystem and be lost on restart).

```bash
printf %s "hf_..." | gcloud secrets create hf-token --data-file=- --project my-gcp-project
./deploy/cloudrun.sh --project my-gcp-project --user-email you@example.com --hf-token-secret hf-token
```

**Who can do what.** As the note above says, everyone you let through IAP uses the app *as its service account*:
they can start benchmark endpoints (billed to the project), call Gemini and see every session. Granting a whole
domain (`--member domain:example.com`) opens all of that to everyone in it; prefer a Google Group of trusted AI-Ops
users.

**Cloud mode.** On Cloud Run (`K_SERVICE` is set; `DILEMMA_CLOUD=1|0` forces it on or off) the app hardens itself:

- **Every request must come through IAP.** The app verifies IAP's signed header (`x-goog-iap-jwt-assertion`:
  signature, issuer, expiry and this service's audience, read from the metadata server) on every page, asset and API
  call, so knowing a URL is not enough, even for someone who can invoke the service directly. Requests without a
  valid header get 401 and a `access_denied` log line.
- **Turned off:** Google sign-in (the app is its service account), switching projects (fixed to the deployment's;
  a switch would apply to everyone), saving or removing the Hugging Face token in Settings (use
  `--hf-token-secret`), and the `/docs` API pages.
- **File paths in a session's config** (`prompts.file`, `prompts.system_prompts_file`, `traffic.hourly_csv`) must
  point inside the sessions folder, so a config can't read other files on the server (such as its environment).

**Audit log: who did what.** Each action (every non-GET request, and downloads) writes one structured line naming
the IAP user, the method, path, session and status; benchmark jobs also record `started_by` in `job.json`. In Logs
Explorer:

```
resource.type="cloud_run_revision" resource.labels.service_name="mlops-dilemma" jsonPayload.audit.user="alice@example.com"
```

`jsonPayload.audit.event="access_denied"` shows refused requests. Cloud Run's own request logs and Agent Platform's
audit logs don't show the person: Agent Platform records the service account as the caller. IAP can also log
every request with the user's email if you enable Data Access audit logs for "Cloud Identity-Aware Proxy" (IAM & Admin
→ Audit Logs); they're off by default.

**Caveats.**

- If Cloud Run restarts the instance during a benchmark, the app marks the job as interrupted and offers to tear the endpoint down.

To remove everything:

```bash
./deploy/teardown-cloudrun.sh --project my-gcp-project          # the service
./deploy/teardown-cloudrun.sh --project my-gcp-project --all    # also the bucket, service account and image repo
```

**Local container test** (local mode: no IAP check; add `-e DILEMMA_CLOUD=1 -e DILEMMA_REQUIRE_IAP=0` to preview
cloud mode's restrictions without IAP).

```bash
docker build -t mlops-dilemma:local .
docker run --rm -p 8110:8080 --user $(id -u):$(id -g) -e HOME=/tmp \
  -e DILEMMA_RUNS_DIR=/data/runs -e DILEMMA_CACHE_DIR=/data/cache -v $(mktemp -d):/data \
  -v ~/.config/gcloud/application_default_credentials.json:/secrets/adc.json:ro \
  -e GOOGLE_APPLICATION_CREDENTIALS=/secrets/adc.json mlops-dilemma:local
```

## Repo layout

```
src/aiops_dilemma/
  server.py     web app backend (FastAPI; assets/ holds the UI and the in-browser cost model, model.js)
  auth.py       ADC identity, browser sign-in, project picker   jobs.py  durable benchmark jobs
  config.py     workload schema + defaults                      recommender.py  model scoring
  gemini.py     live Gemini catalog + role aliases              simulate.py  request-level replay
  cli.py        commands          planner.py  GPU fit + ranking      pricing.py   Cloud Billing prices
  ingest.py     uploads, traffic time-series import                    monitoring.py  Cloud Monitoring traffic
  catalog.py    chip specs, fallback shapes   modelinfo.py  model catalog, MG options   traffic.py  weekly profile
  hfmeta.py     Hugging Face model metadata   hardware.py   region accelerators (Compute Engine API)
  quotas.py     Agent Platform serving quotas per accelerator and region (Cloud Quotas API)
  hftoken.py    Hugging Face token (Settings: env > 0600 file, never in sessions)
  prompts.py    prompt sets       deploy.py   Agent Platform endpoint         loadtest.py  load generator
  serving.py    custom serving container (image, command, args, env, routes, patch) + tuned-config import
  userprompts.py  your own prompts (parse, store, summary, answer-length defaults, file routing)
  analyze.py    peak throughput   costmodel.py fleet vs MaaS          report.py / dashboard.py
examples/       workload definitions
blog/example/   the blog post's example workload and run (also the app's built-in example)
```

## Disclaimer

This code is provided “as-is” as a demonstration only to illustrate a potential solution. The code does not constitute a Google product or service of any kind, and Google offers no support, warranties, or liability of any kind with its regard. Whoever chooses to use this code accepts all responsibility related to it, including for its implementation, use, and ongoing maintenance. For the avoidance of doubt, this code is not eligible for the Google Open Source Software Vulnerability Rewards Program.
