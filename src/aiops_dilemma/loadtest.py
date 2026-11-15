"""Closed-loop concurrent load test against the run's Agent Platform dedicated endpoint (OpenAI chat API).

Every level gets its own sample of prompts with the same mix (`plan_levels`: same shares of short / medium / long
and thinking prompts, no prompt reused across levels while the pool allows), so levels are comparable and prefix
caching only helps where your prompts genuinely share a prefix (e.g. fixed system prompts) — as in production. Requests are
streamed to capture time-to-first-token; records go to runs/<name>/load/c<N>.jsonl.

Extra prompt sets (e.g. `dilemma prompts --long-short-output`) can be run with `prompts_path`; give
them a `tag` so results go to runs/<name>/load/<tag>/c<N>.jsonl and don't overwrite the main sweep
(`dilemma analyze` reads only the untagged files).

Before the first level, a pre-flight check (`preflight`) sends one long streamed generation and one long-prompt
request: an endpoint that cuts long requests (e.g. a serving image whose proxy drops streams after ~30 s) fails in
about two minutes instead of after a full, paid sweep of failed requests. `loadtest.preflight: false` skips it.
"""

import asyncio
import itertools
import json
import math
import time
from collections.abc import Callable
from pathlib import Path

import google.auth.transport.requests
import httpx


class Auth:
    def __init__(self):
        # ADC with the session's project as quota project (a job gets it as $GOOGLE_CLOUD_PROJECT from the app)
        from aiops_dilemma import auth
        self.creds = auth.credentials()

    def token(self) -> str:
        if not self.creds.valid or (self.creds.expiry and self.creds.expiry.timestamp() - time.time() < 300):
            self.creds.refresh(google.auth.transport.requests.Request())
        return self.creds.token


async def _one(client, url, auth, prompt, default_kwargs) -> dict:
    body = {"model": "default", "messages": prompt["messages"], "max_tokens": prompt.get("max_tokens", 1024),
            "temperature": 0.2, "stream": True, "stream_options": {"include_usage": True},
            "skip_special_tokens": False}
    if prompt.get("response_format"):
        body["response_format"] = prompt["response_format"]
    kwargs = {**default_kwargs, **prompt.get("chat_template_kwargs", {})}
    thinking = bool(kwargs.get("enable_thinking"))
    if kwargs:
        body["chat_template_kwargs"] = kwargs
    rec, usage = {"id": prompt.get("id"), "system_tokens": prompt.get("system_tokens")}, {}
    content, finish = [], None
    wants_json = bool(prompt.get("response_format")) or "JSON" in str(prompt["messages"][0].get("content", ""))[:400]
    t0 = time.perf_counter()
    rec["start"] = time.time()
    try:
        async with client.stream("POST", url, json=body, headers={"Authorization": f"Bearer {auth.token()}"}) as r:
            if r.status_code != 200:
                rec.update(status=r.status_code, error=(await r.aread()).decode()[:300])
                return rec
            async for line in r.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                usage = chunk.get("usage") or usage
                for ch in chunk.get("choices", []):
                    d = ch.get("delta", {})
                    if d.get("content"):
                        content.append(d["content"])
                    finish = ch.get("finish_reason") or finish
                    if (d.get("content") or d.get("reasoning_content") or d.get("reasoning")) and "ttft" not in rec:
                        rec["ttft"] = time.perf_counter() - t0
        rec["status"] = 200
        rec["finish_reason"] = finish
        if wants_json:   # a reasoning model with no thinking budget can spend max_tokens before the JSON ends
            text = "".join(content).strip()
            if text.startswith("```"):
                text = text.strip("`").removeprefix("json").strip()
            try:
                json.loads(text)
                rec["json_ok"] = True
            except ValueError:
                rec["json_ok"] = False
    except Exception as e:
        rec.update(status=-1, error=repr(e)[:300])
    finally:
        rec["latency"] = time.perf_counter() - t0
        rec["thinking"] = thinking
    rec.update(prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens"))
    return rec


READY_TIMEOUT_S = 1200
READY_POLL_S = 15


def wait_ready(url: str, timeout_s: float = READY_TIMEOUT_S, poll_s: float = READY_POLL_S,
               progress_cb: Callable[[dict], None] | None = None) -> float:
    """Block until the endpoint answers a tiny chat request with 200; return seconds waited.

    A dedicated endpoint can answer 503 "no healthy upstream" for several minutes after the deploy operation
    completes, while the model server loads and the route comes up. Raises SystemExit with the last error if it
    never becomes ready (the caller tears the endpoint down)."""
    auth, t0, last = Auth(), time.time(), ""
    body = {"model": "default", "messages": [{"role": "user", "content": "Reply with OK."}], "max_tokens": 4}
    while True:
        try:
            r = httpx.post(url, json=body, timeout=120,
                           headers={"Authorization": f"Bearer {auth.token()}"})
            if r.status_code == 200:
                waited = time.time() - t0
                print(f"endpoint ready after {waited:.0f}s", flush=True)
                return waited
            last = f"HTTP {r.status_code}: {r.text[:200]}"
        except Exception as e:   # noqa: BLE001
            last = repr(e)[:200]
        waited = time.time() - t0
        if progress_cb:
            progress_cb({"event": "waiting_ready", "waited_s": round(waited), "last": last})
        if waited > timeout_s:
            raise SystemExit(f"endpoint not ready after {waited / 60:.0f} min — last response: {last}")
        print(f"waiting for the endpoint ({waited:.0f}s): {last}", flush=True)
        time.sleep(poll_s)


# ------------------------------------------------------------------------------------------------ pre-flight check
PREFLIGHT_MIN_S, PREFLIGHT_MAX_S = 45, 90   # long-output check duration on an idle GPU (from the latency budget)
PREFLIGHT_DECODE_TPS = 150                   # single-stream decode guess used to size max_tokens / min_tokens
PREFLIGHT_CONCLUSIVE_S = 40                  # finishing faster than this doesn't prove long requests survive
PREFLIGHT_LONG_PROMPT = 100_000              # long-prompt check size when the workload has prompts over 32K
PREFLIGHT_SHORT_CAP = 32_000
FILLER_CHARS_PER_TOKEN = 4.0                 # rough, for the filler text; the server's usage gives the real count
CONTEXT_MARGIN = 512


def preflight_target_s(cfg: dict) -> float:
    """Seconds the long-output check should stream: half the latency budget, clamped to 45-90 s."""
    budget = float((cfg.get("constraints") or {}).get("latency_budget_s") or 600)
    return float(min(PREFLIGHT_MAX_S, max(PREFLIGHT_MIN_S, budget / 2)))


def _context(cfg: dict) -> int | None:
    m = cfg.get("model") or {}
    v = m.get("max_model_len") or m.get("context_length")
    return int(v) if v else None


def preflight_output_tokens(cfg: dict) -> int:
    lt = cfg.get("loadtest") or {}
    tps = float(lt.get("preflight_decode_tps") or PREFLIGHT_DECODE_TPS)
    n = int(math.ceil(preflight_target_s(cfg) * tps))
    ctx = _context(cfg)
    return max(256, min(n, ctx - CONTEXT_MARGIN)) if ctx else n


def preflight_prompt_tokens(cfg: dict, run_dir: Path | None = None) -> tuple[int | None, str]:
    """(tokens, why) for the long-prompt check; tokens None = skip (the model's context is smaller)."""
    req = cfg.get("requests") or {}
    if float(req.get("pct_over_32k") or 0) > 0:
        n, why = PREFLIGHT_LONG_PROMPT, "the workload has prompts over 32K"
    else:
        largest = 0
        path = Path(run_dir) / "prompts.jsonl" if run_dir else None
        if path and path.exists():
            for line in open(path):
                if line.strip():
                    try:
                        largest = max(largest, int(json.loads(line).get("prompt_tokens") or 0))
                    except (ValueError, TypeError):
                        pass
        n = largest or 2 * int(req.get("avg_input_tokens") or 6000)
        n, why = min(PREFLIGHT_SHORT_CAP, n), "the workload's largest prompts"
    ctx = _context(cfg)
    if ctx and n + CONTEXT_MARGIN > ctx:
        return None, f"skipped: {n:,}-token prompt exceeds the model's {ctx:,}-token context"
    return n, why


def _filler(tokens: int) -> str:
    words = ("the quarterly review notes that inventory levels remained stable while shipping times improved "
             "across most regions and the support queue shrank after the new routing rules went live").split()
    out, i, need = [], 0, int(tokens * FILLER_CHARS_PER_TOKEN)
    size = 0
    while size < need:
        w = words[i % len(words)]
        out.append(w)
        size += len(w) + 1
        i += 1
        if i % 40 == 0:
            out.append(f"(section {i // 40})\n")
    return " ".join(out)


def _http_client(timeout_s: float) -> httpx.Client:     # tests swap in a MockTransport
    return httpx.Client(timeout=httpx.Timeout(timeout_s, connect=30))


def _preflight_stream(url: str, headers: dict, body: dict, timeout_s: float) -> dict:
    """One streamed chat request. Returns status, duration_s, ttft_s, tokens, prompt_tokens, finish_reason, done,
    error. Never raises: a cut stream or timeout is reported in `error`."""
    out = {"status": None, "duration_s": 0.0, "ttft_s": None, "tokens": 0, "prompt_tokens": None,
           "finish_reason": None, "done": False, "error": None}
    t0 = time.perf_counter()
    chunks, usage = 0, {}
    try:
        with _http_client(timeout_s) as client, client.stream("POST", url, json=body, headers=headers) as r:
            out["status"] = r.status_code
            if r.status_code != 200:
                out["error"] = r.read().decode(errors="replace")[:300]
                return out
            for line in r.iter_lines():
                if time.perf_counter() - t0 > timeout_s:
                    out["error"] = f"timeout after {timeout_s:.0f}s"
                    return out
                if not line.startswith("data: "):
                    continue
                if line.strip() == "data: [DONE]":
                    out["done"] = True
                    continue
                try:
                    chunk = json.loads(line[6:])
                except ValueError:
                    continue
                usage = chunk.get("usage") or usage
                for ch in chunk.get("choices", []):
                    d = ch.get("delta", {})
                    if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                        chunks += 1
                        if out["ttft_s"] is None:
                            out["ttft_s"] = round(time.perf_counter() - t0, 2)
                    out["finish_reason"] = ch.get("finish_reason") or out["finish_reason"]
    except Exception as e:   # noqa: BLE001
        out["error"] = repr(e)[:300]
    finally:
        out["duration_s"] = round(time.perf_counter() - t0, 1)
        out["tokens"] = int(usage.get("completion_tokens") or chunks)
        out["prompt_tokens"] = usage.get("prompt_tokens")
    return out


def _check_failure(r: dict) -> str | None:
    """None when the request ended normally, else a short description of how it failed."""
    secs, toks = r.get("duration_s") or 0, r.get("tokens") or 0
    status, err = r.get("status"), str(r.get("error") or "")
    if status is not None and status != 200:
        what = "503 no healthy upstream" if "no healthy upstream" in err else f"HTTP {status}"
        return f"the endpoint answered {what} after {secs:.0f} s ({err[:120].strip()})"
    if "timeout" in err.lower():
        return f"the request timed out after {secs:.0f} s ({toks:,} tokens)"
    if err or not (r.get("finish_reason") or r.get("done")):
        return f"the endpoint cut a streaming request after {secs:.0f} s ({toks:,} tokens)"
    return None


def _context_error(r: dict) -> bool:
    err = str(r.get("error") or "").lower()
    return r.get("status") == 400 and ("context" in err or "too long" in err or "maximum" in err)


def preflight(url: str, cfg: dict, run_dir: Path | None = None, progress_cb: Callable[[dict], None] | None = None,
              image: str | None = None) -> dict:
    """Pre-flight check: (a) one streamed generation forced to run ~45-90 s (min_tokens = max_tokens, ignore_eos),
    (b) one long prompt (the workload's largest, ~100K when it has prompts over 32K) with a short answer.

    Pass = both end normally (finish_reason or [DONE]). Writes run_dir/preflight.json; raises SystemExit with a
    precise message when the endpoint cuts, times out or 5xxs (the job's teardown then removes the endpoint)."""
    auth = Auth()
    image_txt = image or (cfg.get("deployment") or {}).get("image") or "the Model Garden verified recipe image"
    target = preflight_target_s(cfg)
    out_tokens = preflight_output_tokens(cfg)
    prompt_tokens, prompt_why = preflight_prompt_tokens(cfg, run_dir)
    plan = {"target_s": target, "output_tokens": out_tokens, "prompt_tokens": prompt_tokens,
            "prompt_note": prompt_why}
    if progress_cb:
        progress_cb({"event": "preflight_start", **plan})
    print(f"pre-flight: a ~{target:.0f} s streamed generation ({out_tokens:,} tokens) and "
          + (f"a {prompt_tokens:,}-token prompt ({prompt_why})" if prompt_tokens else f"no long prompt ({prompt_why})"),
          flush=True)
    t0 = time.time()
    checks: list[dict] = []
    result = {"ran": True, "passed": False, "image": image or (cfg.get("deployment") or {}).get("image"),
              "started_at": t0, **plan, "checks": checks}

    def finish(passed: bool, message: str) -> dict:
        result.update(passed=passed, message=message, duration_s=round(time.time() - t0, 1))
        if run_dir:
            (Path(run_dir) / "preflight.json").write_text(json.dumps(result, indent=2))
        if progress_cb:
            progress_cb({"event": "preflight_done", "result": result})
        print(f"pre-flight {'passed' if passed else 'FAILED'}: {message}", flush=True)
        return result

    def fail(desc: str):
        msg = f"Pre-flight failed: {desc}. " + preflight_advice(image or (cfg.get("deployment") or {}).get("image"),
                                                                  image_txt)
        finish(False, msg)
        raise SystemExit(msg)

    def call(name: str, body: dict, timeout_s: float) -> dict:
        r = _preflight_stream(url, {"Authorization": f"Bearer {auth.token()}"}, body, timeout_s)
        r["name"] = name
        return r

    # (a) long streamed output
    def long_output_body(n: int) -> dict:
        return {"model": "default", "stream": True, "stream_options": {"include_usage": True},
                "messages": [{"role": "user", "content": "Count upward from 1, one number per line. Do not stop."}],
                "max_tokens": n, "min_tokens": n, "ignore_eos": True, "temperature": 0.0}
    r = call("long_output", long_output_body(out_tokens), target * 3 + 60)
    why = _check_failure(r)
    if why:
        checks.append({**r, "passed": False})
        fail(why)
    ctx = _context(cfg)
    if r["duration_s"] < PREFLIGHT_CONCLUSIVE_S:   # decode was faster than assumed: once more, longer (at most 4x)
        more = int(min(out_tokens * 4, out_tokens * target / max(1.0, r["duration_s"])))
        if ctx:
            more = min(more, ctx - CONTEXT_MARGIN)
        if more > out_tokens * 1.2:
            checks.append({**r, "passed": True, "note": "finished too fast to be conclusive; retried longer"})
            out_tokens = result["output_tokens"] = more
            r = call("long_output", long_output_body(more), target * 3 + 60)
            why = _check_failure(r)
            if why:
                checks.append({**r, "passed": False})
                fail(why)
    r["passed"] = True
    r["conclusive"] = r["duration_s"] >= PREFLIGHT_CONCLUSIVE_S
    checks.append(r)
    notes = [] if r["conclusive"] else [f"the long generation finished in {r['duration_s']:.0f} s, too fast to prove "
                                        "that requests over ~30 s survive"]

    # (b) long prompt, short output
    if prompt_tokens:
        body = {"model": "default", "stream": True, "stream_options": {"include_usage": True}, "max_tokens": 16,
                "temperature": 0.0,
                "messages": [{"role": "user", "content": _filler(int(prompt_tokens * 0.9))
                              + "\n\nReply with the single word OK."}]}
        r2 = call("long_prompt", body, 600)
        if _context_error(r2):
            r2.update(passed=None, note="skipped: the server rejected the prompt as longer than its context")
            notes.append(r2["note"])
        else:
            why = _check_failure(r2)
            if why:
                checks.append({**r2, "passed": False})
                fail(why.replace("a streaming request", "a long-prompt request"))
            r2["passed"] = True
        checks.append(r2)
    else:
        checks.append({"name": "long_prompt", "passed": None, "note": prompt_why})
    lo = next(c for c in reversed(checks) if c["name"] == "long_output")
    lp = checks[-1]
    msg = (f"a {lo['duration_s']:.0f} s stream ({lo['tokens']:,} tokens) completed"
           + (f"; a {lp.get('prompt_tokens') or prompt_tokens:,}-token prompt completed in {lp['duration_s']:.0f} s"
              if lp.get("passed") else "")
           + ("" if not notes else " (" + "; ".join(notes) + ")"))
    return finish(True, msg)


def short_image(entry: dict) -> str:
    """Readable name for a catalog image: 'pytorch-vllm-serve:gemma4 (sha256:3fbc0e08…)' for digest pins."""
    name = entry["image"].rsplit("/", 1)[-1]
    if "@sha256:" in name:
        repo, digest = name.split("@", 1)
        return f"{repo}:{entry['tag']} ({digest[:15]}…)" if entry.get("tag") else f"{repo}@{digest[:15]}…"
    return name


def preflight_advice(image: str | None, image_txt: str) -> str:
    """What to do after a failed pre-flight: switch to the pinned image (named, with its test date)."""
    from aiops_dilemma.catalog import SERVING_IMAGES
    pinned = next((x for x in SERVING_IMAGES if x["key"] == "pinned"), None)
    if pinned and image == pinned["image"]:
        return (f"The endpoint was served by the pinned image {image}, so this is not the known image problem: check "
                "the endpoint's inference timeout and the model server logs, or try another image (deployment.image).")
    pin = (f"the pinned image ({short_image(pinned)}"
           + (f", tested {pinned['tested']}" if pinned.get("tested") else "") + ")") if pinned else "another image"
    return (f"Agent Platform endpoints served by {image_txt} cut long requests; choose {pin} under Serving image in "
            "the benchmark settings (or set deployment.image) and rerun.")


def endpoint_url(state: dict) -> str:
    return f"https://{state['dedicated_dns']}/v1beta1/{state['endpoint']}/chat/completions"


def preflight_enabled(cfg: dict) -> bool:
    return bool((cfg.get("loadtest") or {}).get("preflight", True))


def run_preflight(cfg: dict, run_dir: Path, progress_cb: Callable[[dict], None] | None = None,
                  wait: bool = False) -> dict:
    """Pre-flight against the endpoint recorded in run_dir/endpoint.json (`dilemma preflight`)."""
    path = Path(run_dir) / "endpoint.json"
    if not path.exists():
        raise SystemExit(f"{path} not found: deploy an endpoint first (`dilemma deploy`)")
    state = json.loads(path.read_text())
    url = endpoint_url(state)
    if wait:
        wait_ready(url, progress_cb=progress_cb)
    return preflight(url, cfg, run_dir, progress_cb, image=state.get("image"))


# ------------------------------------------------------------------------------------------- per-level prompt samples
# Every level must see the same prompt mix, or levels aren't comparable: a level whose slice happens to hold more long
# or thinking prompts dips for that reason alone. Prompts are bucketed by (size class, thinking) and each level gets
# the same number from every bucket (largest-remainder quotas of the pool's shares), drawn without overlap between
# levels while the bucket lasts (fresh user content: no prefix-cache hits a production stream wouldn't get).
SIZE_CLASSES = ("short", "medium", "long")       # <= 32K, 32K-128K, > 128K input tokens
SUBBINS, SUBBIN_MIN = 4, 200                     # size quartiles inside buckets of at least 200 prompts


def size_class(tokens: float) -> str:
    return "long" if tokens > 128000 else "medium" if tokens > 32000 else "short"


def prompt_tokens_estimate(p: dict) -> int:
    """Input tokens of a prompt record: the synthetic builder's `input_tokens`, else a characters-per-token estimate
    (precise enough to tell the size classes apart)."""
    for k in ("input_tokens", "target_input_tokens", "prompt_tokens"):
        if p.get(k):
            try:
                return int(p[k])
            except (TypeError, ValueError):
                pass
    chars = sum(len(str(m.get("content") or "")) for m in p.get("messages") or [])
    return int(chars / 3.5)


def prompt_thinking(p: dict, default_kwargs: dict | None = None) -> bool:
    return bool({**(default_kwargs or {}), **(p.get("chat_template_kwargs") or {})}.get("enable_thinking"))


def _quotas(counts: dict, n: int, caps: dict | None = None) -> dict:
    """Largest-remainder split of n requests over buckets in proportion to `counts` (deterministic tie order).
    Rounding up prefers buckets still under `caps` (their fair share of the pool, so they don't run out)."""
    total = sum(counts.values())
    raw = {k: n * v / total for k, v in counts.items()}
    q = {k: int(math.floor(x)) for k, x in raw.items()}
    caps = caps or {}
    order = sorted(raw, key=lambda k: (q[k] + 1 > caps.get(k, math.inf), -(raw[k] - q[k]), -counts[k], str(k)))
    for k in order[: n - sum(q.values())]:
        q[k] += 1
    return q


def plan_levels(prompts: list[dict], sizes: list[int], default_kwargs: dict | None = None,
                seed: str = "levels") -> tuple[list[list[dict]], dict]:
    """Per-level prompt samples with the same (size class, thinking) mix. Returns (samples, info).

    Deterministic: a fixed seed shuffles each bucket once, levels take consecutive non-overlapping slices, and each
    level's order is shuffled with a per-level seed. When a bucket runs out, it wraps around (prompts repeat, marked
    `repeat: true` on the copies) and info["repeats"] says how many."""
    import random
    toks = [prompt_tokens_estimate(p) for p in prompts]
    keys = [(size_class(t), prompt_thinking(p, default_kwargs)) for p, t in zip(prompts, toks)]
    pools: dict = {}
    for i, k in enumerate(keys):
        pools.setdefault(k, []).append(i)
    # Large buckets are split further into size quartiles, so the average prompt size matches between levels too
    # (within the short class, sizes still vary several-fold).
    for k in [k for k, v in pools.items() if len(v) >= SUBBIN_MIN]:
        idx = sorted(pools.pop(k), key=lambda i: (toks[i], i))
        for q in range(SUBBINS):
            pools[(*k, q)] = idx[q * len(idx) // SUBBINS:(q + 1) * len(idx) // SUBBINS]
    for k, idx in pools.items():
        random.Random(f"{seed}-{k[0]}-{int(k[1])}-{k[2] if len(k) > 2 else ''}").shuffle(idx)
    counts = {k: len(v) for k, v in pools.items()}
    cursor = {k: 0 for k in pools}
    used: set = set()
    samples, repeats = [], 0
    for li, n in enumerate(sizes):
        quota = _quotas(counts, n, {k: v * n // max(1, sum(sizes)) for k, v in counts.items()})
        level = []
        for k in sorted(quota, key=lambda k: (SIZE_CLASSES.index(k[0]), k[1], k[2] if len(k) > 2 else -1)):
            for _ in range(quota[k]):
                pool = pools[k]
                i = pool[cursor[k] % len(pool)]
                cursor[k] += 1
                p = dict(prompts[i])
                if i in used:
                    p["repeat"] = True
                    repeats += 1
                used.add(i)
                p["_size_class"], p["_thinking"] = k[0], k[1]
                level.append(p)
        random.Random(f"{seed}-level-{li}-{n}").shuffle(level)
        samples.append(level)
    pool_mix: dict = {}
    for k, v in counts.items():
        name = f"{k[0]}{'+thinking' if k[1] else ''}"
        pool_mix[name] = pool_mix.get(name, 0) + v
    info = {"pool": len(prompts), "requests": sum(sizes), "repeats": repeats, "pool_mix": dict(sorted(pool_mix.items()))}
    if repeats:
        info["note"] = (f"{len(prompts)} prompts for {sum(sizes)} requests: {repeats} requests reuse a prompt "
                        "(prefix caching may flatter those levels; raise prompts.synthetic.count)")
    return samples, info


async def _level(url, prompts, start, n, concurrency, default_kwargs,
                 progress_cb: Callable[[dict], None] | None = None) -> list[dict]:
    auth, records, queue = Auth(), [], itertools.islice(itertools.cycle(prompts), start, None)
    remaining = n
    last_emit = 0.0
    limits = httpx.Limits(max_connections=concurrency + 4, max_keepalive_connections=concurrency + 4)
    async with httpx.AsyncClient(timeout=httpx.Timeout(1800, connect=30), limits=limits) as client:
        async def worker():
            nonlocal remaining, last_emit
            while remaining > 0:
                remaining -= 1
                prompt = next(queue)
                rec = await _one(client, url, auth, prompt, default_kwargs)
                rec["concurrency"] = concurrency
                if prompt.get("repeat"):
                    rec["repeat"] = True
                records.append(rec)
                if progress_cb is not None:
                    now = time.perf_counter()
                    if now - last_emit >= 0.4 or len(records) == n:
                        last_emit = now
                        progress_cb({
                            "event": "request_done",
                            "concurrency": concurrency,
                            "completed": len(records),
                            "total": n,
                            "last_status": rec.get("status"),
                            "last_latency": round(rec.get("latency", 0), 2),
                        })
        await asyncio.gather(*(worker() for _ in range(concurrency)))
    return records


def _error_breakdown(records: list[dict]) -> dict:
    out: dict[str, int] = {}
    for r in records:
        if r.get("status") == 200 and r.get("prompt_tokens"):
            continue
        err = str(r.get("error") or "")
        kind = ("503 no healthy upstream" if "no healthy upstream" in err else
                "connection cut mid-stream" if "peer closed" in err or "RemoteProtocolError" in err else
                "timeout" if "Timeout" in err else f"status {r.get('status')}")
        out[kind] = out.get(kind, 0) + 1
    return out


def run_label(tag: str = "") -> str:
    """Which prompt set a load-test level used: "mixed" (the workload's own mix, the main sweep), "short"
    (prompts <= 32K, the calibration pass), "long" (long prompts only), or the tag itself."""
    t = (tag or "").strip().lower()
    return "mixed" if not t else "short" if t == "short" else "long" if t.startswith("long") else t


def summarize(records: list[dict]) -> dict:
    ok = [r for r in records if r.get("status") == 200 and r.get("prompt_tokens")]
    if not ok:
        return {"ok": 0, "errors": len(records), "error_breakdown": _error_breakdown(records)}
    wall = max(1e-6, max(r["start"] + r["latency"] for r in ok) - min(r["start"] for r in ok))
    lat = sorted(r["latency"] for r in ok)
    pin, pout = sum(r["prompt_tokens"] for r in ok), sum(r["completion_tokens"] for r in ok)
    q = lambda xs, p: xs[min(len(xs) - 1, int(p * len(xs)))]  # noqa: E731
    return {"ok": len(ok), "errors": len(records) - len(ok), "wall_s": wall, "rps": len(ok) / wall,
            "input_tps": pin / wall, "output_tps": pout / wall, "avg_in": pin / len(ok), "avg_out": pout / len(ok),
            "lat_p50": q(lat, 0.5), "lat_p95": q(lat, 0.95),
            "ttft_p50": q(sorted(r.get("ttft", r["latency"]) for r in ok), 0.5),
            "error_breakdown": _error_breakdown(records),
            # answered (HTTP 200) but unusable: cut at max_tokens, or JSON that doesn't parse
            "truncated": sum(1 for r in ok if r.get("finish_reason") == "length"),
            "json_invalid": sum(1 for r in ok if r.get("json_ok") is False),
            "thinking_truncated": sum(1 for r in ok if r.get("finish_reason") == "length" and r.get("thinking")),
            "mix": level_mix(records)}


def level_mix(records: list[dict]) -> dict:
    """The prompt mix a level actually saw (answered requests, server-counted tokens): share of requests and of input
    tokens by size class, thinking share, average input / output tokens, and how many requests reused a prompt."""
    ok = [r for r in records if r.get("status") == 200 and r.get("prompt_tokens")]
    n = len(ok)
    if not n:
        return {}
    req = {c: 0 for c in SIZE_CLASSES}
    tok = {c: 0.0 for c in SIZE_CLASSES}
    for r in ok:
        c = size_class(r["prompt_tokens"])
        req[c] += 1
        tok[c] += r["prompt_tokens"]
    pin = sum(tok.values())
    return {"requests": {c: round(v / n, 4) for c, v in req.items()},
            "tokens": {c: round(v / pin, 4) if pin else 0.0 for c, v in tok.items()},
            "thinking": round(sum(1 for r in ok if r.get("thinking")) / n, 4),
            "avg_in": round(pin / n, 1), "avg_out": round(sum(r.get("completion_tokens") or 0 for r in ok) / n, 1),
            "repeats": sum(1 for r in records if r.get("repeat"))}


def _safe_tag(tag: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in tag.strip()).strip("._")


def load_prompts(run_dir: Path, prompts_path: str | Path | None = None, max_tokens: int | None = None) -> list[dict]:
    path = Path(prompts_path) if prompts_path else run_dir / "prompts.jsonl"
    prompts = [json.loads(line) for line in open(path) if line.strip()]
    if not prompts:
        raise SystemExit(f"{path}: no prompts")
    if max_tokens:
        for p in prompts:
            p["max_tokens"] = max_tokens
    return prompts


def run(cfg: dict, run_dir: Path, concurrency: list[int] | None = None, per_level: int | None = None,
        progress_cb: Callable[[dict], None] | None = None, prompts_path: str | Path | None = None,
        tag: str = "", max_tokens: int | None = None, preflight_check: bool | None = None):
    """Concurrency sweep. The pre-flight check runs after the endpoint is ready, before the first level, unless
    `preflight_check` is False (default: `loadtest.preflight`, on, for untagged runs; tagged runs such as the
    calibration's short-only sweep reuse the main run's check)."""
    state = json.loads((run_dir / "endpoint.json").read_text())
    url = endpoint_url(state)
    prompts = load_prompts(run_dir, prompts_path, max_tokens)
    levels = concurrency or cfg["loadtest"]["concurrency"]
    per_level = per_level or cfg["loadtest"]["requests_per_level"]
    sizes = [max(per_level, 2 * c) for c in levels]
    default_kwargs = cfg["model"].get("chat_template_kwargs") or {}
    samples, sample_info = plan_levels(prompts, sizes, default_kwargs, seed=f"{cfg.get('name', 'run')}-levels")
    if sample_info.get("note"):
        print(f"note: {sample_info['note']}")
    out = run_dir / "load"
    if tag and _safe_tag(tag):
        out = out / _safe_tag(tag)
    out.mkdir(parents=True, exist_ok=True)
    old_files = list(out.glob("c*.jsonl"))
    if old_files:   # keep earlier attempts for diagnosis instead of overwriting them
        arch = out / "archive" / time.strftime("%Y%m%d-%H%M%S")
        arch.mkdir(parents=True, exist_ok=True)
        for f in old_files:
            f.rename(arch / f.name)
        print(f"archived {len(old_files)} earlier level files to {arch}")
    wait_ready(url, progress_cb=progress_cb)
    if preflight_check is None:
        preflight_check = preflight_enabled(cfg) and not tag
    if preflight_check:
        preflight(url, cfg, run_dir, progress_cb, image=state.get("image"))
    run_kind = run_label(tag)
    (out / "levels_plan.json").write_text(json.dumps({**sample_info, "levels": [
        {"concurrency": c, "requests": len(smp), "mix": {f"{k}{'+thinking' if t else ''}": sum(
            1 for p in smp if p["_size_class"] == k and p["_thinking"] == t) for k in SIZE_CLASSES for t in (False, True)
            if any(p["_size_class"] == k and p["_thinking"] == t for p in smp)}} for c, smp in zip(levels, samples)]},
        indent=2))
    summaries = []
    for c, n, sample in zip(levels, sizes, samples):
        print(f"concurrency {c}: {n} requests ...", flush=True)
        if progress_cb:
            progress_cb({"event": "level_start", "concurrency": c, "total": n, "run": run_kind})
        recs = asyncio.run(_level(url, sample, 0, n, c, default_kwargs, progress_cb))
        (out / f"c{c}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
        s = summarize(recs)
        s["concurrency"] = c
        s["run"] = run_kind
        if sample_info.get("repeats"):
            s["sample_note"] = sample_info["note"]
        summaries.append(s)
        if tag:
            (out / "summary.json").write_text(json.dumps({"tag": tag, "prompts": str(prompts_path or "prompts.jsonl"),
                                                          "levels": summaries}, indent=2))
        if progress_cb:
            progress_cb({"event": "level_done", "concurrency": c, "summary": s, "run": run_kind})
        if s.get("ok"):
            print(f"  ok={s['ok']} err={s['errors']} input={s['input_tps']:,.0f} tok/s output={s['output_tps']:,.0f} tok/s "
                  f"p50={s['lat_p50']:.0f}s p95={s['lat_p95']:.0f}s"
                  + (f" errors: {s['error_breakdown']}" if s["errors"] else "")
                  + (f" | answered but unusable: {s['truncated']} hit max_tokens ({s['thinking_truncated']} while thinking), "
                     f"{s['json_invalid']} invalid JSON" if s["truncated"] or s["json_invalid"] else ""), flush=True)
        else:
            errs = sorted({str(r.get("status")) + " " + str(r.get("error"))[:120] for r in recs})[:3]
            print(f"  all {s['errors']} requests failed: {errs}", flush=True)
            raise SystemExit(f"every request at concurrency {c} failed ({'; '.join(errs)}) — stopping the load test")
