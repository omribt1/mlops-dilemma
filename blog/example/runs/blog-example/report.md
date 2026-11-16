# Self-hosting vs MaaS — blog-example

> **Estimated throughput, not measured on your workload.** Throughput figures here are estimates (from reference benchmarks or GPU specs), not measured on your workload. Prompt sizes, thinking and output length, long prompts and caching change throughput by several times (unbudgeted thinking cut throughput about 4x in our runs). For a decision, run a benchmark with your own prompts and traffic.

Model **google/gemma4@gemma-4-26b-a4b-it** (fp8) on **1x NVIDIA_RTX_PRO_6000** (g4-standard-48), us-central1. MaaS reference: $0.15/$0.6/$0.015 per 1M input/output/cached tokens, limit 15 req/s. Gemini on Agent Platform (Cloud Billing Catalog): Gemini 3.5 Flash-Lite $0.3/$2.5/$0.03, Gemini 3.8 Flash $1.5/$7.5/$0.15, Gemini 3.1 Pro $2/$12/$0.2, Gemini 4 Carbon $4/$20/$0.2 per 1M input/output/cached tokens.

Throughput: Assumed 9,000 input tok/s per chip: an assumption for this illustrative example (thinking-heavy output), not measured. Run a benchmark to measure it.

## Verdict

MaaS can't serve your peak: limit 15 req/s (planned at 11.5 req/s per hour to absorb bursts) vs 28.0 req/s at peak; MaaS up to its limit plus Gemini 3.5 Flash-Lite overflow would cost **$32K/month** ($10K MaaS + $21K Gemini 3.5 Flash-Lite).

Your pricing option: **Agent Platform + reservation, 1-yr** — **$36K/month** with 12 GPUs (+ MaaS overflow within its limit), **112% of that baseline**. For reference, MaaS-only would cost $17K/month if the limit didn't apply (self-hosting = 206% of that). Average GPU utilization 23%.

Stay on MaaS up to its limit with Gemini 3.5 Flash-Lite overflow: self-hosting costs more at this traffic shape. It still makes sense for data-control reasons, or if your off-peak load grows (see sensitivity).

Allowing Gemini 3.5 Flash-Lite overflow at peaks would cut it to 5 GPUs and save about $6.7K a month. (Set `hybrid.peak_overflow_target: gemini_flash_lite` to allow it.)

- **Use Pay-Per-Use Google Gemini on Agent Platform for HA Failover Instead of Idle Redundant GPUs/TPUs:** Always-on zone redundancy (1.5x) adds 6 idle chips (+$16,261/month), and two-region redundancy (2.0x) adds 12 chips (+$32,522/month). Failing over to Gemini 3.5 Flash-Lite on Agent Platform costs $0/month when healthy and $203 for a full 1-hour peak outage, so idle hardware only pays off above ~80 peak outage hours/month.


## Throughput

Assumed 9,000 input tok/s per chip: an assumption for this illustrative example (thinking-heavy output), not measured. Run a benchmark to measure it. No load test was run, so there is no concurrency sweep or latency data.


## Monthly cost by deployment option (today)

| Option | $/GPU-hour | GPUs (always on) | Monthly (+ overflow) | vs baseline | vs MaaS-only (no limit) | Break-even utilization |
|---|---|---|---|---|---|---|
| Agent Platform, on demand | $5.85 | 12 | $55K | 171% | 314% | 82% |
| Agent Platform + reservation, 1-yr | $3.71 | 12 | $36K | 112% | 206% | 52% |
| Agent Platform + reservation, 3-yr | $2.52 | 12 | $25K | 79% | 146% | 35% |
| Self-managed GKE, 1-yr | $3.11 | 12 | $30K | 95% | 175% | 44% |
| Self-managed GKE, 3-yr | $1.98 | 12 | $21K | 65% | 119% | 28% |

Baseline: Open-model MaaS up to 15 req/s + Gemini 3.5 Flash-Lite overflow, $32K/month for 79B input tokens (32% cached); MaaS-only would be $17K/month if the 15 req/s limit didn't apply. Break-even utilization = how busy a GPU must be to beat MaaS on the same tokens; a fleet sized for the peak averages 23%.


## Hybrid Routing: Diverting Long-Context (>32K) & Peak Traffic to MaaS / Google Gemini on Agent Platform

Prompt distribution: **Short (≤32K)** = 100.0% of requests (100% of tokens), **Medium (32K–128K)** = 0.0% (0% of tokens), **Long (>128K)** = 0.0% (0% of tokens). If served on self-hosted hardware, prompts >32K consume **0% of accelerator time**.

| Routing Strategy | Self-Hosted Chips | Hardware / mo | MaaS + Gemini / mo | Total / mo | vs baseline |
|---|---|---|---|---|---|
| All prompts on self-hosted fleet (+ peak overflow) | 12 | $33K | $3K | $36K | 112% |
| Divert >32K prompts to Open-Model MaaS | 12 | $33K | $3K | $36K | 112% |
| Divert >32K prompts to Gemini 3.5 Flash-Lite (Agent Platform) | 12 | $33K | $3K | $36K | 112% |
| Divert >32K prompts to Gemini 3.8 Flash (Agent Platform) | 12 | $33K | $3K | $36K | 112% |

## Reliability & Resilience: Always-On Redundant Hardware vs. Google Gemini on Agent Platform Failover

Instead of provisioning idle redundant GPUs/TPUs across zones (+50% hardware) or regions (+100% hardware) that bill 24/7, a **3-Layer Google Cloud architecture** runs a 1.0× single-zone fleet (Layer 1), bursts normal peaks to Open-Model MaaS (Layer 2), and fails over to **Google Gemini on Agent Platform (`Gemini 3.5 Flash-Lite` or `Gemini 3.8 Flash`)** as a pay-per-use Layer 3 ($0 when idle).

| Incident / Overflow Scenario | Spilled Input Tokens | Gemini 3.5 Flash-Lite cost | Gemini 3.8 Flash cost | Open MaaS Reference |
|---|---|---|---|---|
| Accelerator fleet down for peak hour (1h) | 295.9M | $203 | $736 | $65 |
| Accelerator fleet down for full busy weekday (24h) | 1,435.7M | $987 | $4K | $316 |
| Open-Model MaaS rate-limited during peak hour (1h) | 207.7M | $143 | $517 | $46 |

Compare against always-on hardware redundancy on **Agent Platform + reservation, 1-yr**:
- **Zone-redundant (1.5× fleet):** +6 always-on chips = **+$16K/month** every month.
- **Two-region (2.0× fleet):** +12 always-on chips = **+$33K/month** every month.
- **Break-even:** always-on zone redundancy only becomes cheaper than failing over to **Gemini 3.5 Flash-Lite on Agent Platform** above ~80 peak outage hours a month.


## Sensitivity


### Off-peak load grows (busiest hour unchanged)

| Scenario | MaaS-only | Baseline | Fleet (Agent Platform + reservation, 1-yr) | vs MaaS-only | vs baseline | Avg GPU util |
|---|---|---|---|---|---|---|
| off-peak ×1 | $17K | $32K (+ overflow) | $36K (12 GPUs) | 206% | 112% | 23% |
| off-peak ×2 | $24K | $47K (+ overflow) | $38K (12 GPUs) | 162% | 82% | 28% |
| off-peak ×4 | $29K | $58K (+ overflow) | $40K (12 GPUs) | 136% | 69% | 35% |

### Overall growth

| Scenario | MaaS-only | Baseline | Fleet (Agent Platform + reservation, 1-yr) | vs MaaS-only | vs baseline | Avg GPU util |
|---|---|---|---|---|---|---|
| volume ×1 | $17K | $32K (+ overflow) | $36K (12 GPUs) | 206% | 112% | 23% |
| volume ×2 | $35K | $81K (+ overflow) | $90K (33 GPUs) | 260% | 111% | 20% |
| volume ×5 | $87K | $234K (+ overflow) | $252K (93 GPUs) | 290% | 108% | 18% |
| volume ×10 | $174K | $494K (+ overflow) | $526K (194 GPUs) | 303% | 106% | 17% |

### MaaS rate limit

| Scenario | MaaS-only | Baseline | Fleet (Agent Platform + reservation, 1-yr) | vs MaaS-only | vs baseline | Avg GPU util |
|---|---|---|---|---|---|---|
| 0 req/s | $17K | $54K (+ overflow) | $57K (21 GPUs) | 328% | 105% | 16% |
| 15 req/s | $17K | $32K (+ overflow) | $36K (12 GPUs) | 206% | 112% | 23% |
| 30 req/s | $17K | $20K (+ overflow) | $22K (4 GPUs) | 124% | 109% | 32% |
| 60 req/s | $17K | $17K | $17K (1 GPUs) | 100% | 100% | 52% |

### Throughput error (vs assumed)

| Scenario | MaaS-only | Baseline | Fleet (Agent Platform + reservation, 1-yr) | vs MaaS-only | vs baseline | Avg GPU util |
|---|---|---|---|---|---|---|
| 0.75× assumed | $17K | $32K (+ overflow) | $47K (16 GPUs) | 269% | 146% | 23% |
| 1.00× assumed | $17K | $32K (+ overflow) | $36K (12 GPUs) | 206% | 112% | 23% |
| 1.25× assumed | $17K | $32K (+ overflow) | $30K (10 GPUs) | 173% | 94% | 22% |

### Resilience

| Scenario | MaaS-only | Baseline | Fleet (Agent Platform + reservation, 1-yr) | vs MaaS-only | vs baseline | Avg GPU util |
|---|---|---|---|---|---|---|
| single zone | $17K | $32K (+ overflow) | $36K (12 GPUs) | 206% | 112% | 23% |
| zone-redundant | $17K | $32K (+ overflow) | $52K (18 GPUs) | 300% | 163% | 23% |
| two regions | $17K | $32K (+ overflow) | $68K (24 GPUs) | 393% | 214% | 23% |

## Planner candidates

| Accelerator | Type | Chips/replica | Machine | Verified | Agent Platform $/chip-h |
|---|---|---|---|---|---|
| RTX PRO 6000 | GPU | 1 | g4-standard-48 | no | $5.85 |
| A100 40GB | GPU | 2 | a2-highgpu-2g | no | $4.92 |
| L4 | GPU | 2 | g2-standard-24 | no | $1.30 |
| H100 80GB | GPU | 1 | a3-highgpu-1g | no | $14.27 |
| A100 80GB | GPU | 1 | a2-ultragpu-1g | no | $6.54 |
| B200 180GB | GPU | 8 | a4-highgpu-8g | no | $20.94 |
| H200 141GB | GPU | 8 | a3-ultragpu-8g | no | $12.11 |
| Cloud TPU v6e (Trillium) | TPU | 4 | ct6e-standard-4t | no | $3.35 |
| Cloud TPU v5e | TPU | 4 | ct5lp-hightpu-4t | no | $1.50 |
| Cloud TPU v5p | TPU | 4 | ct5p-hightpu-4t | no | $5.25 |

*Estimates for planning, not a quote. Fleet is fixed (no autoscaling); prices from the Cloud Billing catalog (Agent Platform + reservation = GCE CUD rates + Agent Platform management fee with a Flexible Savings Plan).*
