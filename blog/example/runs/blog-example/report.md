# Self-hosting vs MaaS — blog-example

> **Estimated throughput, not measured on your workload.** Throughput figures here are estimates (from reference benchmarks or GPU specs), not measured on your workload. Prompt sizes, thinking and output length, long prompts and caching change throughput by several times (unbudgeted thinking cut throughput about 4x in our runs). For a decision, run a benchmark with your own prompts and traffic.

Model **google/gemma4@gemma-4-26b-a4b-it** (fp8) on **1x NVIDIA_RTX_PRO_6000** (g4-standard-48), us-central1. MaaS reference: $0.15/$0.6/$0.015 per 1M input/output/cached tokens, limit 15 req/s. Gemini on Agent Platform (Cloud Billing Catalog): Gemini 3.5 Flash-Lite $0.3/$2.5/$0.03, Gemini 3.8 Flash $1.5/$7.5/$0.15, Gemini 3.1 Pro $2/$12/$0.2, Gemini 4 Carbon $4/$20/$0.2 per 1M input/output/cached tokens.

Throughput: 17,946 input tok/s per chip on short prompts, measured in a benchmark of this illustrative example's workload on 1 × RTX PRO 6000 (2026-11-16); long prompts use its measured per-class rates. Your workload will differ: run a benchmark to measure it.

## Verdict

MaaS can't serve your peak: limit 15 req/s (planned at 11.5 req/s per hour to absorb bursts) vs 28.0 req/s at peak; MaaS up to its limit plus Gemini 3.5 Flash-Lite overflow would cost **$22K/month** ($8K MaaS + $14K Gemini 3.5 Flash-Lite).

Your pricing option: **Agent Platform + reservation, 1-yr** — **$29K/month** with 10 GPUs (+ MaaS overflow within its limit), **132% of that baseline**. For reference, MaaS-only would cost $13K/month if the limit didn't apply (self-hosting = 224% of that). Average GPU utilization 22%.

Stay on MaaS up to its limit with Gemini 3.5 Flash-Lite overflow: self-hosting costs more at this traffic shape. It still makes sense for data-control reasons, or if your off-peak load grows (see sensitivity).

Allowing Gemini 3.5 Flash-Lite overflow at peaks would cut it to 1 GPUs and save about $8.1K a month. (Set `hybrid.peak_overflow_target: gemini_flash_lite` to allow it.)

- **Divert >32K Long-Context Requests to Google Gemini on Agent Platform:** Requests >32K tokens are 1.25% of requests but consume 52% of accelerator time. Routing >32K prompts to Gemini 3.5 Flash-Lite cuts the self-hosted fleet by 5 chips and saves $8,948/month.

- **Use Pay-Per-Use Google Gemini on Agent Platform for HA Failover Instead of Idle Redundant GPUs/TPUs:** Always-on zone redundancy (1.5x) adds 5 idle chips (+$13,551/month), and two-region redundancy (2.0x) adds 10 chips (+$27,101/month). Failing over to Gemini 3.5 Flash-Lite on Agent Platform costs $0/month when healthy and $137 for a full 1-hour peak outage, so idle hardware only pays off above ~99 peak outage hours/month.


## Throughput

17,946 input tok/s per chip on short prompts, measured in a benchmark of this illustrative example's workload on 1 × RTX PRO 6000 (2026-11-16); long prompts use its measured per-class rates. Your workload will differ: run a benchmark to measure it. No load test was run, so there is no concurrency sweep or latency data.


## Monthly cost by deployment option (today)

| Option | $/GPU-hour | GPUs (always on) | Monthly (+ overflow) | vs baseline | vs MaaS-only (no limit) | Break-even utilization |
|---|---|---|---|---|---|---|
| Agent Platform, on demand | $5.85 | 10 | $45K | 201% | 343% | 89% |
| Agent Platform + reservation, 1-yr | $3.71 | 10 | $29K | 132% | 224% | 56% |
| Agent Platform + reservation, 3-yr | $2.52 | 10 | $21K | 93% | 158% | 38% |
| Self-managed GKE, 1-yr | $3.11 | 10 | $25K | 112% | 191% | 47% |
| Self-managed GKE, 3-yr | $1.98 | 10 | $17K | 75% | 128% | 30% |

Baseline: Open-model MaaS up to 15 req/s + Gemini 3.5 Flash-Lite overflow, $22K/month for 79B input tokens (32% cached); MaaS-only would be $13K/month if the 15 req/s limit didn't apply. Break-even utilization = how busy a GPU must be to beat MaaS on the same tokens; a fleet sized for the peak averages 22%.


## Hybrid Routing: Diverting Long-Context (>32K) & Peak Traffic to MaaS / Google Gemini on Agent Platform

Prompt distribution: **Short (≤32K)** = 98.8% of requests (78% of tokens), **Medium (32K–128K)** = 0.9% (11% of tokens), **Long (>128K)** = 0.3% (10% of tokens). If served on self-hosted hardware, prompts >32K consume **52% of accelerator time**.

This mix costs **1.62×** the accelerator time per token of short prompts: 11,044 input tok/s per chip with every prompt on the fleet, 17,946 with >32K prompts routed away.

| Routing Strategy | Self-Hosted Chips | Hardware / mo | MaaS + Gemini / mo | Total / mo | vs baseline |
|---|---|---|---|---|---|
| All prompts on self-hosted fleet (+ peak overflow) | 10 | $27K | $2K | $29K | 132% |
| Divert >32K prompts to Open-Model MaaS | 5 | $14K | $4K | $18K | 79% |
| Divert >32K prompts to Gemini 3.5 Flash-Lite (Agent Platform) | 5 | $14K | $7K | $20K | 92% |
| Divert >32K prompts to Gemini 3.8 Flash (Agent Platform) | 5 | $14K | $27K | $41K | 184% |

## Reliability & Resilience: Always-On Redundant Hardware vs. Google Gemini on Agent Platform Failover

Instead of provisioning idle redundant GPUs/TPUs across zones (+50% hardware) or regions (+100% hardware) that bill 24/7, a **3-Layer Google Cloud architecture** runs a 1.0× single-zone fleet (Layer 1), bursts normal peaks to Open-Model MaaS (Layer 2), and fails over to **Google Gemini on Agent Platform (`Gemini 3.5 Flash-Lite` or `Gemini 3.8 Flash`)** as a pay-per-use Layer 3 ($0 when idle).

| Incident / Overflow Scenario | Spilled Input Tokens | Gemini 3.5 Flash-Lite cost | Gemini 3.8 Flash cost | Open MaaS Reference |
|---|---|---|---|---|
| Accelerator fleet down for peak hour (1h) | 295.9M | $137 | $536 | $49 |
| Accelerator fleet down for full busy weekday (24h) | 1,435.7M | $664 | $3K | $239 |
| Open-Model MaaS rate-limited during peak hour (1h) | 207.7M | $96 | $376 | $35 |

Compare against always-on hardware redundancy on **Agent Platform + reservation, 1-yr**:
- **Zone-redundant (1.5× fleet):** +5 always-on chips = **+$14K/month** every month.
- **Two-region (2.0× fleet):** +10 always-on chips = **+$27K/month** every month.
- **Break-even:** always-on zone redundancy only becomes cheaper than failing over to **Gemini 3.5 Flash-Lite on Agent Platform** above ~99 peak outage hours a month.


## Sensitivity


### Off-peak load grows (busiest hour unchanged)

| Scenario | MaaS-only | Baseline | Fleet (Agent Platform + reservation, 1-yr) | vs MaaS-only | vs baseline | Avg GPU util |
|---|---|---|---|---|---|---|
| off-peak ×1 | $13K | $22K (+ overflow) | $29K (10 GPUs) | 224% | 132% | 22% |
| off-peak ×2 | $18K | $32K (+ overflow) | $31K (10 GPUs) | 176% | 97% | 28% |
| off-peak ×4 | $22K | $40K (+ overflow) | $32K (10 GPUs) | 146% | 81% | 35% |

### Overall growth

| Scenario | MaaS-only | Baseline | Fleet (Agent Platform + reservation, 1-yr) | vs MaaS-only | vs baseline | Avg GPU util |
|---|---|---|---|---|---|---|
| volume ×1 | $13K | $22K (+ overflow) | $29K (10 GPUs) | 224% | 132% | 22% |
| volume ×2 | $26K | $56K (+ overflow) | $74K (27 GPUs) | 282% | 133% | 20% |
| volume ×5 | $66K | $159K (+ overflow) | $206K (76 GPUs) | 314% | 129% | 18% |
| volume ×10 | $131K | $334K (+ overflow) | $428K (158 GPUs) | 327% | 128% | 17% |

### MaaS rate limit

| Scenario | MaaS-only | Baseline | Fleet (Agent Platform + reservation, 1-yr) | vs MaaS-only | vs baseline | Avg GPU util |
|---|---|---|---|---|---|---|
| 0 req/s | $13K | $36K (+ overflow) | $46K (17 GPUs) | 351% | 126% | 16% |
| 15 req/s | $13K | $22K (+ overflow) | $29K (10 GPUs) | 224% | 132% | 22% |
| 30 req/s | $13K | $15K (+ overflow) | $16K (3 GPUs) | 126% | 113% | 33% |
| 60 req/s | $13K | $13K | $14K (1 GPUs) | 103% | 103% | 48% |

### Throughput error (vs assumed)

| Scenario | MaaS-only | Baseline | Fleet (Agent Platform + reservation, 1-yr) | vs MaaS-only | vs baseline | Avg GPU util |
|---|---|---|---|---|---|---|
| 0.75× assumed | $13K | $22K (+ overflow) | $38K (13 GPUs) | 288% | 169% | 23% |
| 1.00× assumed | $13K | $22K (+ overflow) | $29K (10 GPUs) | 224% | 132% | 22% |
| 1.25× assumed | $13K | $22K (+ overflow) | $24K (8 GPUs) | 183% | 107% | 22% |

### Resilience

| Scenario | MaaS-only | Baseline | Fleet (Agent Platform + reservation, 1-yr) | vs MaaS-only | vs baseline | Avg GPU util |
|---|---|---|---|---|---|---|
| single zone | $13K | $22K (+ overflow) | $29K (10 GPUs) | 224% | 132% | 22% |
| zone-redundant | $13K | $22K (+ overflow) | $43K (15 GPUs) | 328% | 192% | 22% |
| two regions | $13K | $22K (+ overflow) | $57K (20 GPUs) | 431% | 253% | 22% |

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
