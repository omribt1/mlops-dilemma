# Methodology: how aiops-dilemma calculates its answers

This document explains every number the app and the CLI produce: the traffic model, the request model, throughput
per chip, fleet sizing and cost, the hybrid options, the benchmark the app runs on a real GPU, and the request-level
simulation. Each section names the module that implements it, so you can check the code. The Python engine
(`costmodel.py`) and the in-browser engine (`assets/model.js`) implement the same formulas, and the tests check
that they agree.

## 1. The question and the baseline

The tool compares two ways to serve the same traffic for a month:

- **Self-hosting:** a fixed fleet of GPUs or TPUs running an open model on Agent Platform (or GKE), plus whatever
  overflow it sends to per-token services.
- **The per-token alternative (the baseline):** open-model MaaS (the managed, per-token version of the same model).

MaaS has a rate limit. If your busiest hour needs more requests per second than the limit, "MaaS only" isn't an
option, so the honest baseline is **MaaS up to its limit every hour, with the rest sent to a Gemini model** on Agent
Platform (`hybrid.failover_target`, Gemini Flash-Lite by default). If the peak fits under the limit, the baseline is
plain MaaS (`costmodel.baseline`).

Hourly averages hide minutes that run hotter, so MaaS is **planned at `limit / burst_headroom`** of hourly average
(default 15 / 1.3 ≈ 11.5 req/s): exactly the burst room a fleet's overflow gets (section 5). Both sides of the
comparison use the same planned capacity.

Every result is reported as **self-hosting cost as a % of the baseline**. Below 100%, self-hosting is cheaper.

## 2. Traffic model

Traffic is a **week of 168 hourly averages of input tokens per second**, Monday 00:00 UTC first (`traffic.py`). A
month is that week scaled to 730 hours. Sources, in order of precedence:

1. **Your own series** (`traffic.hourly_csv`): imported from Cloud Monitoring, a Metrics Explorer export or a
   request log. Imports are resampled by time into hourly means; at least 7 days become a typical week (mean per
   weekday-hour), 1–6 days a weekday and a weekend curve, and less than a day a daily shape you confirm
   (`ingest.py`, `monitoring.py`).
2. **A drawn week** (`traffic.custom_hourly`): 168 values painted on the chart, or proposed from a document.
3. **A traffic shape** (`traffic.preset`) at your weekly average: US business hours, US + Europe, follow the sun or
   steady. Each is built from regional users (US East and West, Europe, Asia-Pacific) who are busy in their own
   local working day, a night-time floor and a weekend level, then scaled to your average.
4. **A parametric week** from peak, average and quietest hour, a weekday busy window and a weekend factor. The shape
   is solved so the busiest hour, the quietest hour and the weekly average all match what you entered.

`traffic.monthly_input_tokens` optionally fixes the monthly volume. An imported, drawn or preset week is rescaled to
it; a parametric week is re-solved for the implied average with its peak and quietest hour kept (falling back to a
plain rescale only if that average can't be reached).

**Off-peak growth** (`traffic.raise_off_peak`) multiplies every hour by a factor (1–10) but caps it at the
original busiest hour, so nights and weekends grow while the peak stays the same.

## 3. Request model

Per request (`config.py`, `costmodel.request_mix`):

- `avg_input_tokens`: the whole prompt, system prompt included.
- **Output** = `avg_output_tokens` (the visible answer) + `thinking_share` × `avg_thinking_tokens` (reasoning
  tokens are billed and generated like output).
- **Cached input** = `cached_input_tokens` when set (for example resent chat history), else
  `system_prompt_tokens` × `system_prompt_cache_hit`.

From these: the output ratio `o = output / input` and the cached share `c = cached / input`.

**Size classes** (`costmodel.buckets`): short (≤ 32K tokens), medium (32K–128K) and large (> 128K), from
`pct_over_32k` and `pct_over_128k`, with mean sizes `medium_mean` (default 69K) and `large_mean` (188K). The short
class's mean is solved so the overall average matches `avg_input_tokens`.

**Per-token prices.** MaaS and Gemini are priced per input token with output and caching folded in:

```
rate = ((1 - c) * price_in + c * price_cached + o * price_out) / 1e6        $ per input token
```

Gemini prompts above the model's long-context threshold (200K) pay its long-context prices for the share of tokens
above the threshold (`costmodel.gemini_long_rate`). This applies wherever prompts over 200K can reach Gemini: routed
long prompts, and the whole mix when it goes to Gemini as baseline overflow, peak spill or failover
(`costmodel.gemini_mix_rate`). A Gemini model without a published cached price is priced at 10%
of its input price for cached tokens.

`avg_input_tokens` includes the long prompts. If the long tail alone is larger than that average allows (the short
prompts would have to average under 800 tokens), the CLI and the app warn and ask you to raise the average or lower
the tail; requests per second always use `avg_input_tokens`. The request-level simulation (section 8) instead
draws arrivals at the mix's real mean size, so its tokens match the demand.

## 4. Throughput per chip

Every cost depends on how many input tokens per second one chip serves on your mix. The app always says where that
number came from (`planner.reference_throughput`), in this order of preference:

| Source | When | How |
|---|---|---|
| **Measured** | you ran a benchmark | the load test's peak (section 7) |
| **Reference est.** | a reference benchmark exists for this GPU, count, model and precision | its measured rate, adjusted to your workload |
| **Scaled est.** | a reference benchmark exists for this GPU with another model, precision or count | its rate × the spec formula's model ratio (within 3×), adjusted to your workload |
| **Rough estimate** | no reference benchmark for this GPU | the spec formula alone |

**Reference benchmarks** are measured by this project on one fixed workload (`examples/reference-workload.yaml`:
Gemma 4 26B-A4B FP8, ~5K-token prompts with a 1,800-token system prompt, ~500-token JSON answers, no thinking,
nothing over 32K).

**The spec formula** (`planner._spec_rate`) estimates seconds of chip time per input token:

```
P_eff   = max(3.5B, 0.85 * active_params + 0.15 * total_params)   (mixture-of-experts reads its active experts)
prefill = 2 * P_eff / (0.38 * TFLOPS) * (1 - 0.75 * c)            seconds per input token, c capped at 0.85
decode  = P_eff * bytes_per_param / (18 * 0.55 * bandwidth)    seconds per output token
per_chip_tps = TP_efficiency / (prefill + o * decode)
```

Here `o` and `c` are taken at the **short** prompts' mean size when the workload has prompts over 32K (the formula
gives a short-prompt rate; the mix factor below accounts for long ones). FP8 raises effective TFLOPS by 1.35× and
bandwidth by 1.15×; INT4 and MXFP4 read fewer bytes per parameter when
decoding. Tensor-parallel efficiency depends on the interconnect and the chip count (`catalog.TP_EFF`). Per-chip
TFLOPS, bandwidth and memory come from vendor data sheets (`catalog.CHIP_SPECS`), the one table with no API behind
it. The formula is only used as a **ratio** when a reference benchmark exists, never for its absolute level.

**Workload adjustment.** A reference rate is moved to your workload by the formula's ratio at your workload over the
reference workload, both on your model. That mostly captures the decode term (more output or thinking per input
token means fewer input tokens per second), plus prompt size and cache share.

**Long prompts.** The rate above is a short-prompt rate. Longer prompts cost more chip time per token: attention
grows with length, less of the prompt is a cached system prompt, and a long request holds batch slots. Each class
has a slowdown, by default **3× for 32K–128K and 4× for > 128K** (measured on Gemma 4 FP8 on an RTX PRO 6000 in
mixed traffic), replaced by measured per-class rates when you have them (section 7). The mix factor

```
f = sum over classes of (token_share * slowdown)       (1.0 when nothing is over 32K)
blended_tps = short_tps / f
```

turns the short-prompt rate into the rate for your mix (`costmodel.buckets`, `ref_mix_factor`).

## 5. Fleet sizing and cost

**GPU capacity is scarce, so the fleet is fixed**, sized once and running every hour; the model doesn't assume
autoscaling (`costmodel.fixed_fleet`, `eval_fleet`).

For a fleet of `n` chips, each hour `h` of the week with demand `D_h` (input tok/s):

```
capacity  = n * blended_tps * 0.85                  (chips planned at 85% busy)
on_fleet  = min(D_h, capacity)
spill     = D_h - on_fleet
to_maas   = min(spill, maas_cap / headroom)         maas_cap = rate limit (req/s) * avg_input_tokens
to_gemini = spill - to_maas
```

`headroom` (`constraints.burst_headroom`, default 1.3) covers bursts inside an hour: hourly averages hide minutes
that run hotter, so only part of the MaaS limit is planned against.

**Fleet policy.** A fleet is acceptable when:

1. no hour sends more overflow than its targets may take: open-model MaaS only, by default
   (`hybrid.peak_overflow_target: open_maas`). If you allow a Gemini model at peaks, MaaS takes overflow first and
   Gemini the rest, within an assumed quota of 150 requests/s; and
2. the fleet plus that overflow can carry the busiest hour with headroom:
   `n ≥ (peak × headroom − maas_cap − gemini_cap) / blended_tps`, in whole replicas, where `maas_cap` is the rate limit
   in tokens per second (limit × `avg_input_tokens`) and `gemini_cap` is 0 unless Gemini may take peak overflow.

The app chooses **the cheapest acceptable fleet**, comparing each candidate size at its full cost per serving chip
(price × the resilience multiplier, section 6). The draggable fleet line in the app uses the same rule, so its
cheapest acceptable position is always the recommended fleet. Its position is the number of **serving** replicas;
standby chips come on top and are priced in the same way. When allowing Gemini at peaks would be at least 3%
cheaper, the app says so and offers to switch.

**Monthly cost:**

```
chips      = n, rounded up to whole replicas after × resilience_multiplier (section 6)
fleet cost = chips * price_per_chip_hour * 730
total      = fleet cost + sum(to_maas) * maas_rate + sum(to_gemini) * gemini_rate      (over a 730-hour month)
```

**Pricing options** (`pricing.py`, live from the Cloud Billing Catalog for your region; a labelled fallback table
otherwise), each per chip-hour:

| Option | Built from |
|---|---|
| Agent Platform, on demand | accelerator + its share of the machine's vCPU and memory + Agent Platform management fee |
| Agent Platform + reservation, 1-yr / 3-yr | Compute Engine SKUs at committed-use rates + the management fee with a Flexible Savings Plan discount |
| Self-managed GKE, 1-yr / 3-yr | Compute Engine SKUs at committed-use rates, no management fee |

The cheapest fleet can differ between pricing options: when a chip-hour is expensive, fewer chips and more overflow
win; when it's cheap, more chips and less overflow. The hardware view lists every option with its own fleet.

**Break-even utilization** is the share of the month a chip must be busy to beat MaaS on the same tokens:

```
break-even = price_per_chip_hour / (blended_tps * 3600 * maas_rate)
```

A fixed fleet runs at roughly *average ÷ peak* of its capacity, so compare the two: this is why the shape of the
week matters more than its volume.

## 6. Hybrid options, resilience and incidents

- **Peak overflow** (above): where the busiest hours spill.
- **Long-prompt routing** (`hybrid.route_long`): prompts over 32K go to MaaS or a Gemini model instead of the
  fleet. The fleet then serves only short prompts at the short-prompt rate, and the long prompts are priced per token
  at that target's rates. Routed to open-model MaaS, long prompts use its rate limit first, with the same burst room as
  overflow (their busiest hour's requests/s × `burst_headroom`); the short prompts' peak overflow gets what's left.
- **Failover** (`hybrid.failover_target`): during an outage, traffic goes per token instead of to standby chips.

**Resilience** (`constraints.resilience_multiplier`): 1.5 models zone redundancy and 2 two regions. The extra chips
are **standby capacity**: they don't serve traffic while the primary fleet is healthy, so overflow is the same as
without redundancy. The total chips are the serving fleet × the multiplier, rounded up to whole replicas, and the
serving fleet is chosen at the real cost of a serving chip (price × multiplier), which favours a slightly smaller
fleet with more overflow. **Incident costs** (`costmodel.resilience_and_hybrid_summary`) take the week's busiest hour and
the day that contains it: losing every chip for that hour, or for that whole day, sends the traffic above the planned
MaaS limit (limit / headroom) to the failover model, priced per token. The **break-even outage time** is the redundancy's extra monthly
cost divided by the cost of an outage hour (capped at a month): redundancy pays off only if you'd otherwise expect
that many hours of full outage.

## 7. The benchmark (what the app sends to the GPU)

A benchmark replaces the estimate with a measurement (`jobs.py`, `deploy.py`, `prompts.py`, `loadtest.py`,
`analyze.py`). Steps: plan → prompts → deploy → load → analyze → calibrate (only with prompts over 32K) → report →
teardown. The endpoint is torn down even when a step fails or the run is cancelled.

**Prompts.** Either **your own** (uploaded JSONL, JSON, CSV or text; answer lengths and thinking come from the file
or the workload's defaults), or **synthetic ones shaped like your workload**:

- Sizes are drawn to match your size mix: short prompts from a lognormal distribution clipped to 400–32K tokens,
  its parameters solved so the clipped sizes average what the overall `avg_input_tokens` requires; medium prompts 0.65–1.35× `medium_mean`, clamped to
  32.5K–128K; large ones 0.75–1.25× `large_mean`, at least 128.5K (no upper cap). A class mean near a boundary
  therefore shifts slightly. Tokens are counted with the model's own tokenizer.
- Content is filler records and sentences, so no two requests share user content.
- A small set of shared system prompts (8 by default, sized 0.4–1.6× `system_prompt_tokens`) prefixes the requests,
  so prefix caching helps exactly as much as a shared system prompt would in production.
- Answers are JSON. With `match_output_length` the system prompt asks for about `avg_output_tokens`, so the
  benchmark measures the answer length you configured.
- `thinking_share` of the prompts run with reasoning on. Each request's `max_tokens` is 2 × `avg_output_tokens`, plus
  1.5 × `avg_thinking_tokens` when reasoning is on (0 if unset, as in the cost model); prompts over 32K without reasoning are capped at 400 output tokens
  (long prompts usually return short structured answers).
- 1,600 prompts by default: enough for 4 levels × 300 requests without reusing a prompt.

**Deployment.** One replica on a dedicated Agent Platform endpoint, from Model Garden's verified serving recipe for
the model and machine (or the pinned serving image), on the planned GPU and count. A custom serving container
(`serving.py`; README, "Custom serving containers and tuned configs") replaces the image, command, arguments,
environment, routes or shared memory; with `args_mode: append` a flag you give replaces the recipe's copy, with
`replace` your list is used as-is. Throughput measured this way belongs to that serving setup, and the report says so.

**Pre-flight check.** Before measuring, one streamed request is forced to generate for 45–90 s (half the latency
budget; `min_tokens` + `ignore_eos`), and one long prompt is sent (~100K tokens when your workload has prompts over
32K, else your largest prompt). If either is cut, times out or gets a server error, the run stops with a message naming the
serving image, about two minutes in.

**Concurrency sweep.** A closed-loop load test at increasing concurrency (16, 32, 64, 128 by default; 300 requests
per level, or 2 × the concurrency if that's more): at concurrency `c`, `c` workers each send their next request as soon as the previous one finishes.
Requests are streamed, so each records time to first token, total latency, prompt and output tokens, and whether
the answer was cut off at its token limit. **Every level gets its own sample with the same mix** (shares of short,
medium and long prompts, thinking on or off, and size quartiles), so levels are comparable and caching can't make a
later level look faster.

**Analysis.** Each level's input tokens per second, latency percentiles and error rate are computed. Levels with
more than 10% failed requests (HTTP errors or no usage reported) are rejected. Answers cut off at their token limit or
returning invalid JSON still count as served throughput; their share is reported per level, because a reasoning
model without a thinking budget would otherwise fail every level. Treat a high truncation share as a sign to raise
`max_tokens` or cap reasoning, and size for the output you see. **The peak** is the highest-throughput level whose p95 latency fits
`constraints.latency_budget_s` (600 s by default); throughput per chip = that level's input tok/s ÷ chips. The run
records the prompt mix it measured, so the cost model can rescale it if your mix changes later.

**Long-prompt calibration.** A blended measurement can't tell short prompts from long ones, so with prompts over 32K
the benchmark runs a second, short-only pass on the same endpoint (`analyze.calibrate_classes`). With `S` the
short-only peak rate, `T` the mixed run's peak rate, and `t_s`, `t_m`, `t_l` the token shares by class in the mixed
run's peak level:

```
1/T = t_s/S + k * (3 * t_m + 4 * t_l)/S
```

solves one long-prompt factor `k`. The measured slowdowns become `3k` and `4k` (at least 1), and the per-class rates
are stored with the measurement (`bucket_tps`) and used by every view.

## 8. Request-level simulation

The cost model works on hourly averages; the simulation checks it with individual requests (`simulate.py`):

- Poisson arrivals that follow the hourly curve, with sizes drawn from the size classes. The baseline it compares
  against is the hourly model's (MaaS planned at limit ÷ headroom).
- Each replica serves its queue first-come-first-served (like vLLM's scheduler). A request costs
  `tokens / class_rate` replica-seconds.
- A gateway picks the less loaded of two random replicas, and sends a request to MaaS (within its rate limit) when
  it would otherwise wait. MaaS is treated as answering immediately, and there is no Gemini tier: a request that
  can't go to MaaS waits for a replica. The simulation checks whether a fleet keeps up; the hourly model (section 5)
  prices the Gemini overflow.
- A request is late if it finishes after the latency budget. A fleet passes when at most 1% of short **and** at most
  1% of long requests are late.
- Scenarios: long prompts on the fleet as they arrive, spread evenly over each day, deferred to off-peak hours, or
  routed to MaaS. Optional 10-minute surges test bursts inside the hour.

The simulation finds the smallest fleet that passes for each scenario and prices it.

## 9. Assumptions and limits

- **Prices are list prices** in one region from the Cloud Billing Catalog (or a labelled fallback). Negotiated
  discounts, sustained-use discounts and taxes are not modelled.
- **Not included:** engineering time to run a fleet, staging or test capacity, monitoring, model-quality validation
  (including for FP8 or INT4), data transfer and storage.
- **Different models answer differently.** Overflow, long-prompt routing and failover send traffic from your open
  model to open-model MaaS (the same model) or to Gemini (a different model family). The cost model assumes the
  answers are acceptable; check your prompts, output formats and quality on every model you route to.
- **Estimates are labelled.** Throughput you haven't measured on your own prompts can be off by several times; the
  app tags every figure with its source.
- **No autoscaling.** The fleet is sized for the busiest hour and runs all month.
- **Latency is a constraint, not a cost.** The benchmark's peak is the highest throughput whose p95 latency fits
  `constraints.latency_budget_s`; a tighter budget lowers throughput per chip and so raises the fleet. Per-token
  services' latency (shared capacity, a different model for Gemini overflow) isn't modelled.
- **Fleet operations aren't priced.** Tensor parallelism across a replica's GPUs is in the throughput estimate
  (`catalog.TP_EFF`, lower without NVLink); load balancing across replicas, rolling upgrades, 10–30-minute deploys,
  GPU capacity shortages and spare test capacity are not. Agent Platform prices include its management fee;
  self-managed GKE prices don't, and the cluster is then yours to run.
- **One replica per benchmark.** Production fleets are modelled as identical replicas of what was measured.
- **Rate limits, prices and GPU availability change.** The data sources (Model Garden, Hugging Face, Compute
  Engine, Cloud Billing, Cloud Quotas, the live Gemini catalog) and their fallbacks are listed in the README under
  "Where the data comes from".
