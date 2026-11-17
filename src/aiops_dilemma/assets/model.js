// Browser port of costmodel.py: fixed GPU/TPU fleet + Open-Model MaaS / Google Gemini on Agent Platform
// hybrid routing (Short <=32K, Medium 32K-128K, Long >128K) + Layer-3 Gemini resilience & incident model.
// The baseline respects the open-model MaaS rate limit (MaaS up to its limit + overflow to the failover target).
(function (root) {
  "use strict";

  // Output tokens per request incl. thinking: visible answer + thinkingShare (0-1) x thinkingTokens
  // (config.effective_output_tokens).
  function effectiveOutput(p) {
    const num = (v) => (v != null && v !== "" && isFinite(+v) ? +v : 0);
    const share = Math.min(1, Math.max(0, num(p.thinkingShare)));
    return Math.max(0, num(p.avgOutput)) + share * Math.max(0, num(p.thinkingTokens));
  }

  // Off-peak load multiplier, valid range 1-10 (config.clamp_off_peak).
  function clampOffPeak(v) {
    const x = +v;
    return isFinite(x) ? Math.min(10, Math.max(1, x)) : 1;
  }

  function maasRate(p, outRatio, cache) {
    return ((1 - cache) * p.maasIn + cache * p.maasCached + outRatio * p.maasOut) / 1e6;
  }

  function geminiRate(gem, outRatio, cache) {
    return ((1 - cache) * gem.in + cache * gem.cached + outRatio * gem.out) / 1e6;
  }

  // Per-token slowdown of 32K-128K and >128K prompts vs short prompts: measured per-class rates (user-set
  // p.mediumTps / p.largeTps, else a benchmark's classTps) divided into the short-prompt rate, else 3x / 4x (measured on one model and GPU).
  const DEFAULT_PENALTIES = { medium: 3.0, large: 4.0 };
  function classPenalties(p, shortTps, classTps) {
    const out = { source: {} };
    const user = { medium: p.mediumTps, large: p.largeTps };
    for (const k of ["medium", "large"]) {
      const rate = +(user[k] || (classTps && classTps[k]) || 0);
      if (shortTps > 0 && rate > 0) { out[k] = Math.min(50, Math.max(1, shortTps / rate)); out.source[k] = "measured"; }
      else { out[k] = DEFAULT_PENALTIES[k]; out.source[k] = "default"; }
    }
    return out;
  }

  // Cached input tokens per request: p.cachedInput if set (chat / agent workloads), else system prompt x hit rate.
  function cachedInput(p) {
    const v = p.cachedInput != null && p.cachedInput !== "" && +p.cachedInput >= 0
      ? +p.cachedInput : (p.systemPrompt || 0) * ((p.systemHit || 0) / 100);
    return p.avgInput > 0 ? Math.min(v, p.avgInput) : v;
  }

  function buckets(p, pen) {
    pen = pen || classPenalties(p, 0, null);
    const over32 = Math.max(0, Math.min(1, (p.pctOver32 || 0) / 100));
    const over128 = Math.max(0, Math.min(over32, (p.pctOver128 || 0) / 100));
    const medReq = Math.max(0, over32 - over128);
    const lrgReq = over128;
    const smlReq = Math.max(1e-6, 1 - medReq - lrgReq);
    const medMean = p.mediumMean || 69000;
    const lrgMean = p.largeMean || 188000;
    const smlMean = Math.max(800, (p.avgInput - medReq * medMean - lrgReq * lrgMean) / smlReq);
    const totalMean = smlReq * smlMean + medReq * medMean + lrgReq * lrgMean;
    const cachedTok = cachedInput(p);
    const effOut = effectiveOutput(p);

    const b = {
      small: { req: smlReq, mean: smlMean, penalty: 1.0 },
      medium: { req: medReq, mean: medMean, penalty: pen.medium },
      large: { req: lrgReq, mean: lrgMean, penalty: pen.large },
    };
    let weightedTime = 0;
    for (const k in b) {
      const v = b[k];
      v.tokenShare = (v.req * v.mean) / Math.max(1, totalMean);
      v.cached = Math.min(1, cachedTok / Math.max(1, v.mean));
      v.outRatio = effOut / Math.max(1, v.mean);
      v.relGpuSec = v.tokenShare * v.penalty;
      weightedTime += v.relGpuSec;
    }
    const smallTimeShare = b.small.relGpuSec / Math.max(1e-9, weightedTime);
    b.longTokenShare = b.medium.tokenShare + b.large.tokenShare;
    b.longReqShare = medReq + lrgReq;
    b.longGpuShare = Math.max(0, 1 - smallTimeShare);
    b.smallSpeedup = over32 > 0 ? Math.max(1, b.small.tokenShare / Math.max(1e-6, smallTimeShare)) : 1;
    // Mix cost factor: GPU time per token of this mix relative to short (<=32K) prompts (1 with none over 32K).
    b.mixFactor = over32 > 0 ? Math.max(1, weightedTime) : 1;
    b.penaltySource = pen.source;
    b.totalMean = totalMean;   // mean request size of the mix (costmodel buckets total_mean)
    return b;
  }
  // Average input tokens per request of the size mix (costmodel.mean_request_tokens).
  function meanRequestTokens(p) { return +p.avgInput || 0; }   // the configured average (see costmodel)

  function demand(base, p) {
    const peak = Math.max(...base);
    const k = p.offPeak == null ? 1 : clampOffPeak(p.offPeak);
    const growth = p.growth == null ? 1 : +p.growth;
    return base.map((d) => (k !== 1 ? Math.min(d * k, peak) : d) * growth);
  }

  // Fleet policy, one rule for the optimizer and the waterline (costmodel.eval_fleet / fixed_fleet).
  // Burst-headroom bound: n >= (peak - cap) x headroom / perGpu, cap = (MaaS cap + Gemini quota when a Gemini tier
  // takes the peak overflow) / headroom; whole replicas, at least one.
  function headroomBound(D, perGpu, replica, maasCapTps, headroom, gemCapTps) {
    const cap = (maasCapTps + (gemCapTps || 0)) / headroom;
    return Math.max(replica, Math.ceil(Math.max((Math.max(...D) - cap) * headroom / perGpu, 0) / replica) * replica);
  }

  // Cheapest policy-OK fleet (= the cheapest policy-OK point on the waterline); minPolicyGpus = the smallest one.
  function fixedFleet(D, perGpu, replica, price, maasCapTps, rate, headroom, gemRate, gemCapTps) {
    const U = 0.85, peak = Math.max(...D);
    const gcap = gemCapTps == null ? null : gemCapTps;
    const rnd = (g) => Math.max(replica, Math.ceil(Math.max(g, 0) / replica) * replica);
    const forPeak = rnd((peak * headroom) / perGpu);
    const allowed = (maasCapTps + (gcap || 0)) / headroom;
    const lo = Math.max(rnd((peak - allowed) / (perGpu * U)), headroomBound(D, perGpu, replica, maasCapTps, headroom, gcap));
    let best = null, first = null;
    for (let n = lo; n <= Math.max(lo, forPeak); n += replica) {
      const r = evalFleet(D, n, perGpu, price, maasCapTps, rate, headroom, gemRate || 0, gcap, replica);
      if (!r.policyOk) continue;
      if (first == null) first = n;
      if (!best || r.cost < best.cost) best = r;
    }
    if (best) best.minPolicyGpus = first;
    return best;
  }

  // Reference fallbacks only (catalog.GEMINI_MODELS); DATA.gemini (from /api/catalog or the exported dashboard) is
  // the live Gemini catalog: role aliases (gemini_flash_lite, gemini_flash, gemini_pro, gemini_4) resolved to live
  // models, plus every live Gemini text model under its id ("gemini-4-carbon"), with long_context tiers.
  const GEMINI_FALLBACK = {
    gemini_flash_lite: { name: "Gemini Flash-Lite", input_per_m: 0.30, output_per_m: 2.50, cached_per_m: 0.03 },
    gemini_flash: { name: "Gemini Flash", input_per_m: 1.50, output_per_m: 7.50, cached_per_m: 0.15 },
    gemini_pro: { name: "Gemini Pro", input_per_m: 2.00, output_per_m: 12.00, cached_per_m: 0.20,
                  long_context: { threshold: 200000, input_per_m: 4.00, output_per_m: 18.00, cached_per_m: 0.40 } },
    gemini_4: { name: "Gemini 4", input_per_m: 4.00, output_per_m: 20.00, cached_per_m: 0.20 },
  };
  const OVERFLOW_TARGETS = ["open_maas", "gemini_flash_lite", "gemini_flash", "gemini_pro", "gemini_4"];   // + model ids
  const GEMINI_ALIASES = ["gemini_flash_lite", "gemini_flash", "gemini_pro", "gemini_4"];
  // Legacy per-alias price overrides (p.liteIn …); p.gemOverrides = {id: {in, out}} is the general form.
  const LEGACY_OVERRIDES = { gemini_flash_lite: ["liteIn", "liteOut"], gemini_flash: ["flashIn", "flashOut"],
                             gemini_pro: ["proIn", "proOut"], gemini_4: ["g4In", "g4Out"] };

  // Long-context tier {threshold, in, out, cached} of a catalog entry, or null (costmodel.long_tier).
  function longTier(src) {
    const inp = +src.input_per_m, lc = src.long_context;
    const cachedBase = src.cached_per_m != null ? +src.cached_per_m : inp * 0.1;
    if (lc && lc.input_per_m != null) {
      return { threshold: +(lc.threshold || 200000), in: +lc.input_per_m,
               out: lc.output_per_m != null ? +lc.output_per_m : +src.output_per_m,
               cached: lc.cached_per_m != null ? +lc.cached_per_m : cachedBase * +lc.input_per_m / Math.max(inp, 1e-12) };
    }
    const li = src.long_input_per_m, lo = src.long_output_per_m;
    if ((li != null && +li !== inp) || (lo != null && +lo !== +src.output_per_m)) {
      const lin = li != null ? +li : inp;
      return { threshold: 200000, in: lin, out: lo != null ? +lo : +src.output_per_m, cached: cachedBase * lin / Math.max(inp, 1e-12) };
    }
    return null;
  }

  // {alias or model id: {name, in, out, cached, long, modelId, alias}} with the user's overrides applied: per id
  // (p.gemOverrides[id], or of the model an alias resolves to) or the legacy per-alias fields (p.liteIn …), which
  // also apply to the model that alias resolves to. Overriding input keeps the cached / long-tier ratios.
  function getGeminiPrices(DATA, p) {
    p = p || {};
    const g = (DATA && DATA.gemini) || {};
    const keys = GEMINI_ALIASES.concat(Object.keys(g).filter((k) => GEMINI_ALIASES.indexOf(k) < 0));
    const num = (v) => (v != null && v !== "" && isFinite(+v) ? +v : null);
    const mid = (k) => (g[k] && (g[k].model_id || g[k].id)) || k;
    const ov = p.gemOverrides || {};
    const legacy = {};   // model id -> {in, out} from the legacy alias fields
    for (const a of GEMINI_ALIASES) {
      const [ik, ok] = LEGACY_OVERRIDES[a], i = num(p[ik]), o = num(p[ok]);
      if (i == null && o == null) continue;
      for (const key of [a, mid(a)]) { const l = legacy[key] || (legacy[key] = {}); if (i != null && l.in == null) l.in = i; if (o != null && l.out == null) l.out = o; }
    }
    const out = {};
    for (const key of keys) {
      const src = Object.assign({}, GEMINI_FALLBACK[key] || {}, g[key] || {});
      if (src.input_per_m == null || src.output_per_m == null) continue;
      const m = mid(key);
      const pick = (f) => {
        for (const o of [ov[key], ov[m], legacy[key], legacy[m]]) if (o && num(o[f]) != null) return num(o[f]);
        return null;
      };
      const baseIn = +src.input_per_m, baseOut = +src.output_per_m;
      const inP = pick("in") ?? baseIn, outP = pick("out") ?? baseOut;
      // Keep the catalog's cached/input ratio when the user overrides the input price.
      const cachedRatio = src.cached_per_m != null && baseIn > 0 ? src.cached_per_m / baseIn : 0.1;
      const lt = longTier(src);
      const kIn = baseIn > 0 ? inP / baseIn : 1, kOut = baseOut > 0 ? outP / baseOut : 1;
      out[key] = { name: src.name, in: inP, out: outP, cached: inP * cachedRatio, modelId: m, alias: GEMINI_ALIASES.indexOf(key) >= 0,
                   family: src.family, long: lt ? { threshold: lt.threshold, in: lt.in * kIn, out: lt.out * kOut, cached: lt.cached * kIn } : null };
    }
    return out;
  }

  // Role alias that replaces a model id no longer offered (gemini.alias_for_id).
  function aliasForId(t) {
    t = String(t || "").toLowerCase();
    if (GEMINI_ALIASES.indexOf(t) >= 0) return t;
    const m = /^gemini-(\d+)(?:\.\d+)?-(.+)$/.exec(t);
    if (!m) return "gemini_flash_lite";
    const fam = m[2];
    if (fam.indexOf("flash-lite") === 0) return "gemini_flash_lite";
    if (fam.indexOf("flash") === 0) return "gemini_flash";
    if (fam.indexOf("pro") === 0) return "gemini_pro";
    return +m[1] >= 4 ? "gemini_4" : "gemini_flash_lite";
  }
  // The gemMap key that prices a target (costmodel.gemini_key): itself, else its role alias; open_maas -> null.
  function geminiKey(target, gemMap) {
    const t = String(target || "").toLowerCase();
    if (t !== "open_maas" && gemMap[t]) return t;
    if (t.indexOf("gemini") === 0) { const a = aliasForId(t); return gemMap[a] ? a : null; }
    return null;
  }
  // Gemini model for traffic beyond the open-model MaaS limit (open_maas can't take it: Flash-Lite).
  function failoverKey(target, gemMap) {
    return geminiKey(target, gemMap) || "gemini_flash_lite";
  }

  // Share of a prompt class above a size threshold, sizes spread evenly from lo to 2 x mean - lo
  // (costmodel.above_threshold): [token share, request share].
  const BUCKET_LOW = { small: 0, medium: 32000, large: 128000 };
  function aboveThreshold(lo, mean, threshold) {
    const hi = 2 * mean - lo;
    if (hi <= lo + 1e-9) return mean > threshold ? [1, 1] : [0, 0];
    if (threshold <= lo) return [1, 1];
    if (threshold >= hi) return [0, 0];
    return [(hi * hi - threshold * threshold) / (hi * hi - lo * lo), (hi - threshold) / (hi - lo)];
  }
  // $ per input token for the >32K classes routed to a Gemini model, with its long-context tier on the share of each
  // class above the threshold (costmodel.gemini_long_rate).
  function geminiLongRate(gem, b, classes) {
    classes = classes || ["medium", "large"];
    const lt = gem.long;
    let tot = 0, cost = 0;
    for (const k of classes) {
      const v = b[k], share = v.tokenShare;
      if (!(share > 0)) continue;
      const [st, sr] = lt ? aboveThreshold(BUCKET_LOW[k], v.mean, lt.threshold) : [0, 0];
      const li = lt ? lt.in : gem.in, lo = lt ? lt.out : gem.out, lc = lt ? lt.cached : gem.cached;
      const c = v.cached, r = v.outRatio;
      const rate = ((1 - c) * ((1 - st) * gem.in + st * li) + c * ((1 - st) * gem.cached + st * lc) + r * ((1 - sr) * gem.out + sr * lo)) / 1e6;
      tot += share; cost += share * rate;
    }
    return tot > 0 ? cost / tot : 0;
  }

  // Pay-per-token baseline honest about the MaaS rate limit (costmodel.baseline): everything on MaaS when the peak
  // fits its limit, else MaaS up to the limit each hour + overflow to the failover target.
  // MaaS planned at limit / headroom per hour, as for a fleet's overflow (costmodel.baseline).
  // Whole prompt mix to a Gemini model: long-context tier on the share above its threshold (costmodel.gemini_mix_rate).
  function geminiMixRate(gem, b, outRatio, cache) {
    return longTier(gem) ? geminiLongRate(gem, b, ["small", "medium", "large"]) : geminiRate(gem, outRatio, cache);
  }

  function baselineOf(D, p, rate, gemMap, failTarget, outRatio, cache, b) {
    const scale = 730 / D.length;
    const meanIn = meanRequestTokens(p);
    const peakRps = Math.max(...D) / Math.max(1, meanIn);
    const capRps = Math.max(0, +p.maasRps || 0);
    const headroom = Math.max(1, +p.headroom || 1);
    const feasible = peakRps <= capRps / headroom + 1e-9;
    let base;
    if (feasible) {
      const total = D.reduce((s, d) => s + d, 0) * 3600 * scale * rate;
      base = { kind: "maas_only", total, maas_cost: total, overflow_cost: 0, overflow_target: null,
               label: "Open-model MaaS only" };
    } else {
      const key = failoverKey(failTarget, gemMap), gem = gemMap[key], capTps = capRps * meanIn / headroom;
      let onMaas = 0, over = 0;
      for (const d of D) { const m = Math.min(d, capTps); onMaas += m; over += d - m; }
      const maasCost = onMaas * 3600 * scale * rate;
      const overCost = over * 3600 * scale * (b ? geminiMixRate(gem, b, outRatio, cache) : geminiRate(gem, outRatio, cache));
      base = { kind: "maas_plus_overflow", total: maasCost + overCost, maas_cost: maasCost, overflow_cost: overCost,
               overflow_target: key,
               label: capRps > 0 ? `Open-model MaaS up to ${capRps} req/s + ${gem.name} overflow`
                                 : `${gem.name} only (open-model MaaS limit is 0 req/s)` };
    }
    return { maasOnlyFeasible: feasible, maasPeakRps: peakRps, maasCapRps: capRps, maasPlannedRps: capRps / headroom,
             baseline: base };
  }

  // Hardware redundancy multiplier: missing, blank, NaN or < 1 -> 1 (a blank <select> once gave 0 GPUs).
  function safeResilience(v) {
    const x = +v;
    return isFinite(x) && x >= 1 ? x : 1;
  }

  // Evaluate a fixed fleet of n chips: the fleet serves up to its capacity, the rest spills to MaaS up to its cap and
  // beyond that to Gemini (geminiRate). gemCapTps null/undefined: Gemini is not allowed at peaks (spill beyond the
  // MaaS cap is priced but flags needsGeminiAtPeaks); a number: a Gemini tier takes the peak overflow within that
  // quota (beyond it: overOverflowQuota). shortOfHeadroom: n below headroomBound. Per-hour series for charting.
  function evalFleet(D, n, perGpu, price, maasCapTps, rate, headroom, geminiRate, gemCapTps, replica) {
    const U = 0.85, scale = 730 / D.length, cap = maasCapTps / headroom, capTps = n * perGpu * U;
    const gcap = gemCapTps == null ? null : gemCapTps;
    const allowed = cap + (gcap != null ? gcap / headroom : 0);
    let onGpu = 0, maas = 0, gem = 0, beyond = false;
    const sFleet = [], sMaas = [], sGem = [];
    for (const d of D) {
      const g = Math.min(d, capTps), s = d - g, m = Math.min(s, cap), x = Math.max(0, s - m);
      if (s > allowed + 1e-6) beyond = true;
      onGpu += g; maas += m; gem += x;
      sFleet.push(g); sMaas.push(m); sGem.push(x);
    }
    const total = onGpu + maas + gem;
    const bound = headroomBound(D, perGpu, replica || 1, maasCapTps, headroom, gcap);
    const short = n < bound;
    const r = {
      gpus: n, gpuCost: n * price * 730, maasCost: maas * 3600 * scale * rate,
      geminiCost: gem * 3600 * scale * geminiRate,
      util: n > 0 ? onGpu / D.length / (n * perGpu) : 0,
      gpuTokenShare: total > 0 ? onGpu / total : 0, geminiTokenShare: total > 0 ? gem / total : 0,
      capacityTps: capTps, series: { fleet: sFleet, maas: sMaas, gemini: sGem },
      policyOk: !(beyond || short), needsGeminiAtPeaks: beyond && gcap == null, overOverflowQuota: beyond && gcap != null,
      shortOfHeadroom: short, headroomGpus: bound,
    };
    r.cost = r.gpuCost + r.maasCost + r.geminiCost;
    return r;
  }

  const PEAK_HINT_MIN_SAVING = 0.03;
  const GEMINI_OVERFLOW_RPS = 150;   // Gemini quota beyond the MaaS limit when a Gemini tier takes peak overflow

  function scenario(DATA, p) {
    const D = demand(DATA.profile, p);
    const scale = 730 / D.length;
    const effOut = effectiveOutput(p);
    const outRatio = effOut / p.avgInput;
    const cache = Math.min(1, cachedInput(p) / p.avgInput);
    const rate = maasRate(p, outRatio, cache);
    const totalTok = D.reduce((s, d) => s + d, 0) * 3600 * scale;
    const price = DATA.prices[p.pricing];
    const b = buckets(p, classPenalties(p, DATA.perGpu * (DATA.mixFactorRef ?? 1), DATA.classTps));
    // DATA.perGpu was obtained at a mix with factor DATA.mixFactorRef (1 = short prompts). All prompts on the
    // fleet run at perGpu * fRef / fNow; with >32K prompts routed away the short-prompt rate is perGpu * fRef.
    const chipTps = DATA.perGpu * (p.tputMult || 1);
    const mixFactorRef = DATA.mixFactorRef ?? 1;
    const perGpu = chipTps * mixFactorRef / b.mixFactor;
    const gemMap = getGeminiPrices(DATA, p);
    const routeTarget = (p.routeLongTarget || "gemini_flash_lite") === "open_maas" ? "open_maas"
      : (geminiKey(p.routeLongTarget || "gemini_flash_lite", gemMap) || "gemini_flash_lite");
    const doRoute = !!p.routeLong && b.longTokenShare > 0;
    const replica = DATA.replica || 1;
    const resilience = safeResilience(p.resilience);

    // Fleet policy (costmodel.scenario): peak overflow fills open-model MaaS within its limit first. "open_maas":
    // Gemini is not allowed at peaks (an overridden fleet that spills beyond the limit is priced at the failover
    // tier and flagged); a Gemini tier takes the overflow beyond the limit within GEMINI_OVERFLOW_RPS.
    const overflowTarget = geminiKey(p.peakOverflowTarget, gemMap) || "open_maas";
    const ovGem = !!gemMap[overflowTarget];
    const maasRps = Math.max(0, +p.maasRps || 0);
    const failTarget = failoverKey(p.failoverTarget, gemMap);
    const spillTarget = ovGem ? overflowTarget : failTarget;   // Gemini tier for spill beyond the MaaS cap
    const out = Object.assign({
      demand: D, maasOnly: totalTok * rate, totalTok, price, overallCache: cache, overallOut: outRatio,
      effectiveOutput: effOut, buckets: b, routing: doRoute, routeTarget, overflowTarget, failoverTarget: failTarget,
      longMaasCost: 0, longRps: 0, spillGeminiTarget: spillTarget, geminiAllowedAtPeaks: ovGem,
      policyOk: true, needsGeminiAtPeaks: false, overOverflowQuota: false, shortOfHeadroom: false,
      throughput: { all: perGpu, small: perGpu * b.smallSpeedup, longGpuShare: b.longGpuShare,
                    perChip: chipTps, mixFactor: b.mixFactor, mixFactorRef },
      resilience,
    }, baselineOf(D, p, rate, gemMap, failTarget, outRatio, cache, b));
    const overrideReplicas = p.fleetOverride != null && p.fleetOverride !== "" && isFinite(+p.fleetOverride)
      ? Math.max(0, Math.round(+p.fleetOverride)) : null;
    let optimal = null;
    const sizeFleet = (Dfleet, perChip, capTps, rate_, outR, cacheR, meanIn, wholeMix) => {
      const gRate = wholeMix ? geminiMixRate(gemMap[spillTarget], b, outR, cacheR) : geminiRate(gemMap[spillTarget], outR, cacheR);
      const gCap = ovGem ? GEMINI_OVERFLOW_RPS * meanIn : null;
      // Standby chips (resilience > 1) make each serving chip cost resilience x price (costmodel.scenario).
      const opt = fixedFleet(Dfleet, perChip, replica, price * resilience, capTps, rate_, p.headroom, gRate, gCap);
      optimal = opt;
      out.optimalGpus = opt ? opt.gpus : null;
      out.optimalReplicas = opt ? opt.gpus / replica : null;
      out.minPolicyGpus = opt ? opt.minPolicyGpus : null;
      out.headroomGpus = headroomBound(Dfleet, perChip, replica, capTps, p.headroom, gCap);
      out.geminiCapTps = gCap;
      if (overrideReplicas == null) return opt;
      out.overridden = true;
      out.fleetReplicas = overrideReplicas;
      // overrideReplicas = serving replicas; standby chips priced in, as for the optimum (costmodel.scenario)
      return evalFleet(Dfleet, overrideReplicas * replica, perChip, price * resilience, capTps, rate_, p.headroom, gRate, gCap, replica);
    };

    let fleet = null, Dfleet = D;
    if (!doRoute) {
      fleet = sizeFleet(D, perGpu, maasRps * meanRequestTokens(p), rate, outRatio, cache, meanRequestTokens(p), true);
      out.capPerGpu = perGpu;
      out.maasCapTps = maasRps * meanRequestTokens(p);
    } else {
      const longShare = b.longTokenShare;
      const longReq = b.longReqShare;
      const longMean = longReq > 0 ? (b.medium.req * b.medium.mean + b.large.req * b.large.mean) / longReq : 1;
      const longCached = longShare > 0 ? (b.medium.tokenShare * b.medium.cached + b.large.tokenShare * b.large.cached) / longShare : 0;
      const longOutRatio = effOut / Math.max(1, longMean);
      const longTps = D.map((d) => d * longShare);
      const longRps = Math.max(...longTps) / Math.max(1, longMean);
      out.longRps = longRps;

      let longRate, spareRps;
      if (gemMap[routeTarget]) {
        longRate = geminiLongRate(gemMap[routeTarget], b);   // long-context tier on the share over its threshold
        spareRps = maasRps; // Gemini on Agent Platform handles >32K on separate high quota
      } else {
        longRate = maasRate(p, longOutRatio, longCached);
        spareRps = maasRps - longRps * Math.max(1, +p.headroom || 1);   // long prompts get the same burst room
      }
      if (spareRps < 0) {
        out.infeasible = "Long-prompt traffic exceeds Open-Model MaaS limit — select Gemini on Agent Platform for >32K routing";
        return out;
      }
      const Ds = D.map((d) => d * b.small.tokenShare);
      Dfleet = Ds;
      const smallPerGpu = perGpu * b.smallSpeedup;
      const smallRate = maasRate(p, b.small.outRatio, b.small.cached);
      fleet = sizeFleet(Ds, smallPerGpu, spareRps * b.small.mean, smallRate, b.small.outRatio, b.small.cached, b.small.mean);
      out.capPerGpu = smallPerGpu;
      out.maasCapTps = spareRps * b.small.mean;
      out.longMaasCost = longTps.reduce((s, d) => s + d, 0) * 3600 * scale * longRate;
    }

    out.fleet = fleet;
    if (!fleet) { out.infeasible = "No fixed fleet satisfies the MaaS rate limit"; return out; }
    out.capacityTps = fleet.capacityTps;
    if (out.fleetReplicas == null) out.fleetReplicas = fleet.gpus / replica;
    out.geminiOverflowCost = fleet.geminiCost || 0;   // beyond the MaaS limit, to spillGeminiTarget
    out.maasOverflowCost = fleet.maasCost;            // peak overflow within the MaaS limit
    // Whole-replica rounding for hardware resilience (redundant chips are idle capacity on top of the fleet)
    const rndRes = (g) => Math.ceil((g * resilience) / replica) * replica;
    out.gpus = rndRes(fleet.gpus);
    out.gpuCost = out.gpus * price * 730;
    out.total = out.gpuCost + fleet.maasCost + (fleet.geminiCost || 0) + out.longMaasCost;
    out.perM = out.total / (totalTok / 1e6);
    out.vsMaas = out.total / out.maasOnly;
    out.vsBaseline = out.baseline.total > 0 ? out.total / out.baseline.total : null;
    for (const k of ["policyOk", "needsGeminiAtPeaks", "overOverflowQuota", "shortOfHeadroom"]) out[k] = !!fleet[k];
    if (optimal) {
      out.optimalTotal = rndRes(optimal.gpus) * price * 730 + optimal.maasCost + (optimal.geminiCost || 0) + out.longMaasCost;
      if (!ovGem && !p._noHint) {
        const alt = scenario(DATA, Object.assign({}, p, { peakOverflowTarget: failTarget, fleetOverride: null, _noHint: true }));
        out.peakGeminiHint = peakGeminiHint(out.optimalTotal, alt, gemMap);
      }
    }
    return out;
  }

  // The optimum with peak overflow to the failover Gemini tier, when >= 3% cheaper (costmodel.peak_gemini_hint).
  function peakGeminiHint(currentTotal, alt, gemMap) {
    if (!alt || alt.total == null || !currentTotal) return null;
    const saving = currentTotal - alt.total;
    if (saving < PEAK_HINT_MIN_SAVING * currentTotal) return null;
    const key = alt.overflowTarget, name = (gemMap[key] && gemMap[key].name) || key;
    return { target: key, name, gpus: alt.gpus, total: alt.total, saving, savingShare: saving / currentTotal,
             vsBaseline: alt.vsBaseline, text: hintText(name, alt.gpus, saving) };
  }
  function hintText(name, gpus, saving, chips) {
    return `Allowing ${name} overflow at peaks would cut it to ${gpus} ${chips || "GPUs"} and save about $${(saving / 1000).toLocaleString("en-US", { minimumFractionDigits: 1, maximumFractionDigits: 1 })}K a month.`;
  }

  function incidents(DATA, p, r) {
    // The week's busiest hour and the day containing it (costmodel.resilience_and_hybrid_summary).
    const peakHour = r.demand.indexOf(Math.max(...r.demand));
    const d0 = Math.floor(peakHour / 24) * 24;
    const day = r.demand.slice(d0, d0 + 24);
    const peak = r.demand[peakHour];
    const maasCap = p.maasRps * meanRequestTokens(p) / Math.max(1, +p.headroom || 1);   // as planned for overflow
    const spillDay = day.reduce((s, d) => s + Math.max(d - maasCap, 0), 0) * 3600;
    const spillPeakHour = Math.max(peak - maasCap, 0) * 3600;
    const maasPeakHour = Math.min(peak, maasCap) * 3600;
    const gemMap = getGeminiPrices(DATA, p);
    const openRate = maasRate(p, r.overallOut, r.overallCache);
    const liteRate = geminiMixRate(gemMap.gemini_flash_lite, r.buckets, r.overallOut, r.overallCache);
    const flashRate = geminiMixRate(gemMap.gemini_flash, r.buckets, r.overallOut, r.overallCache);
    const proRate = geminiMixRate(gemMap.gemini_pro, r.buckets, r.overallOut, r.overallCache);
    const g4Rate = geminiMixRate(gemMap.gemini_4, r.buckets, r.overallOut, r.overallCache);
    const failKey = failoverKey(p.failoverTarget, gemMap);
    const failRate = geminiMixRate(gemMap[failKey], r.buckets, r.overallOut, r.overallCache);

    const rows = [
      ["Fleet down for peak hour (1h)", spillPeakHour],
      ["Fleet down for full weekday (24h)", spillDay],
      ["Open MaaS rate-limited for peak hour (1h)", maasPeakHour],
    ].map(([name, tok]) => ({
      name,
      tok,
      lite: tok * liteRate,
      flash: tok * flashRate,
      pro: tok * proRate,
      g4: tok * g4Rate,
      fail: tok * failRate,   // the failover target (a role alias or a pinned model)
      open: tok * openRate,
    }));

    const base = r.fleet ? r.fleet.gpus : 0;
    const HOURS = 730;
    const replica = DATA.replica || 1;
    const zoneGpus = Math.ceil((base * 1.5) / replica) * replica - base;
    const regGpus = Math.ceil((base * 2.0) / replica) * replica - base;
    const zoneMonthly = zoneGpus * r.price * 730;
    const regMonthly = regGpus * r.price * 730;
    const peakHourLite = rows[0].lite;
    // Outage hours per month at which idle zone-redundant chips cost the same as failing over to Flash-Lite,
    // capped at the hours in a month: beyond that failover is cheaper at any realistic outage rate.
    let breakeven = 0, never = false;
    if (zoneMonthly > 0) {
      const h = peakHourLite > 0 ? zoneMonthly / peakHourLite : Infinity;
      never = h >= HOURS;
      breakeven = never ? HOURS : Math.round(h * 10) / 10;
    }
    return {
      rows,
      failoverTarget: failKey,
      zoneGpus,
      zoneMonthly,
      regGpus,
      regMonthly,
      breakeven_outage_hours: breakeven,
      breakeven_never: never,
      breakevenHours: breakeven, // deprecated alias (capped)
      ratioLite: liteRate / Math.max(1e-12, openRate),
      ratioFlash: flashRate / Math.max(1e-12, openRate),
    };
  }

  const api = { scenario, fixedFleet, evalFleet, headroomBound, peakGeminiHint, hintText, GEMINI_OVERFLOW_RPS, demand, maasRate, geminiRate, buckets, classPenalties, incidents, getGeminiPrices,
                safeResilience, effectiveOutput, clampOffPeak, baseline: baselineOf, failoverKey, geminiKey, aliasForId, geminiLongRate, geminiMixRate,
                aboveThreshold, longTier, GEMINI_ALIASES, OVERFLOW_TARGETS };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.DilemmaModel = api;
})(this);
