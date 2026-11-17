"""Prompt sets for the load test: your own JSONL, custom system prompts, or synthetic prompts built from a short definition.

Your own prompts (prompts.source: file), one JSON object per line:
  {"messages": [{"role": "system", ...}, {"role": "user", ...}],
   "max_tokens": 400, "response_format": {...}, "chat_template_kwargs": {...}}   # last three optional
Use real production prompts if you can: throughput depends on input size, output length and how
much of each prompt is a shared (cacheable) prefix.

Synthetic prompts follow the workload's request profile: N fixed system prompts (prefix-cacheable, or
your uploaded custom system prompts), total input sizes drawn to match Short (<=32K) / Medium (32K-128K) /
Long (>128K) ratios, short JSON outputs, and thinking on for the configured share of requests.
"""

import copy
import json
import math
import random
import urllib.request
from pathlib import Path

from aiops_dilemma import paths, traffic
from aiops_dilemma.config import normalize_buckets

FALLBACK_CHARS_PER_TOKEN = 3.5
WORDS = ("account access alert anomaly approval audit backup billing case change cluster compliance contract "
         "customer dashboard data deadline deployment device document escalation export file finance incident "
         "invoice ledger login migration network order outage partner payment policy priority product project "
         "quota record region release report request review risk role schedule service session shipment "
         "storage subscription supplier support system team ticket token traffic upgrade user vendor workflow").split()


def _tokenizer(hf_id: str | None, cache: Path):
    if not hf_id:
        return None
    try:
        from tokenizers import Tokenizer
        cache.mkdir(parents=True, exist_ok=True)
        path = cache / "tokenizer.json"
        if not path.exists():
            url = f"https://huggingface.co/{hf_id}/resolve/main/tokenizer.json"
            from aiops_dilemma import hftoken
            req = urllib.request.Request(url, headers=hftoken.headers())   # gated repos (Llama) need a token
            path.write_bytes(urllib.request.urlopen(req, timeout=60).read())
        return Tokenizer.from_file(str(path))
    except Exception:
        return None


class TokenCounter:
    def __init__(self, tok):
        self.tok = tok

    def count(self, text: str) -> int:
        return len(self.tok.encode(text).ids) if self.tok else int(len(text) / FALLBACK_CHARS_PER_TOKEN)

    def chars_per_token(self, sample: str) -> float:
        return len(sample) / max(1, self.count(sample))


def _sentence(rng: random.Random, n: int = 14) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(n)).capitalize() + "."


def _record(rng: random.Random, i: int) -> str:
    return json.dumps({"id": f"r{i:07d}", "user": f"user{rng.randint(1, 5000)}", "type": rng.choice(WORDS),
                       "status": rng.choice(["open", "closed", "pending", "flagged"]),
                       "amount": round(rng.lognormvariate(4, 1.2), 2), "note": _sentence(rng, rng.randint(6, 18))},
                      separators=(",", ":"))


def _sample_input(rng: random.Random, req: dict) -> int:
    normalize_buckets(req)
    over32, over128 = req["pct_over_32k"] / 100.0, req["pct_over_128k"] / 100.0
    med_mean = float(req.get("medium_mean") or 69000)
    lrg_mean = float(req.get("large_mean") or 188000)
    u = rng.random()
    if u < over128:
        return int(max(128500, lrg_mean * (0.75 + 0.5 * rng.random())))
    if u < over32:
        return int(min(128000, max(32500, med_mean * (0.65 + 0.7 * rng.random()))))
    # Body (Short <= 32K): lognormal (sigma 0.62) with its mean solved so the overall average matches.
    long_mean = over128 * lrg_mean + (over32 - over128) * med_mean
    body_mean = max(800.0, (req["avg_input_tokens"] - long_mean) / max(1e-9, 1.0 - over32))
    sigma = 0.62
    mu = traffic.clipped_lognormal_mu(body_mean, sigma, 400, 32000)   # the clipped sizes average body_mean
    return int(min(32000, max(400, rng.lognormvariate(mu, sigma))))


def _load_custom_system_prompts(cfg: dict) -> list[str]:
    p_cfg = cfg.get("prompts", {})
    custom = list(p_cfg.get("custom_system_prompts") or [])
    sys_file = p_cfg.get("system_prompts_file")
    if sys_file:
        path = Path(sys_file)
        if not path.is_absolute() and cfg.get("_path"):
            path = Path(cfg["_path"]).parent / path
        path = paths.check_readable(path)   # confined to the session folder on a shared deployment
        if path.exists():
            try:
                data = json.loads(path.read_text())
                if isinstance(data, list):
                    custom.extend(str(x) for x in data if x)
            except Exception:
                txt = path.read_text().strip()
                if txt:
                    custom.append(txt)
    # Deduplicate preserving order
    seen = set()
    out = []
    for s in custom:
        s_clean = s.strip()
        if s_clean and s_clean not in seen:
            seen.add(s_clean)
            out.append(s_clean)
    return out


def _builder(cfg: dict, cache_dir: Path):
    """Return (make, scale, rng): make(n, rng, scale, total=None, max_tokens=None) -> (record, total input tokens)."""
    req, syn = cfg["requests"], cfg["prompts"]["synthetic"]
    tc = TokenCounter(_tokenizer(cfg["model"].get("hf_id"), cache_dir))
    rng = random.Random(f"{cfg['name']}-prompts")

    custom_sys = _load_custom_system_prompts(cfg)
    if custom_sys:
        system_prompts = custom_sys
        sys_sizes = [max(50, tc.count(s)) for s in system_prompts]
        n_sys = len(system_prompts)
    else:
        n_sys = max(1, syn["num_system_prompts"])
        spread = [1.0] if n_sys == 1 else [0.4 + 1.2 * i / (n_sys - 1) for i in range(n_sys)]   # mean 1.0
        sys_sizes = [max(200, int(req["system_prompt_tokens"] * f)) for f in spread]
        cpt_prose = tc.chars_per_token(" ".join(_sentence(rng) for _ in range(50)))
        system_prompts = []
        # match_output_length: ask for an answer of about avg_output_tokens (reference workload); default short answers
        words = int(0.7 * max(64, int(req["avg_output_tokens"])))
        long_answer = bool(syn.get("match_output_length"))
        for i, size in enumerate(sys_sizes):
            r = random.Random(f"sys-{i}")
            if syn["json_output"]:
                style = (f"Reply with compact JSON matching the schema; the reason is a detailed explanation of about "
                         f"{words} words. " if long_answer else
                         "Reply with compact JSON matching the schema; keep the reason under 25 words. ")
            else:
                style = f"Reply in about {words} words. " if long_answer else "Reply concisely. "
            head = f"You are assistant #{i + 1}. Task: {syn['domain'].strip()} " + style
            body = []
            while len(head) + sum(len(b) + 1 for b in body) < size * cpt_prose:
                body.append(f"Guideline {len(body) + 1}: {_sentence(r)}")
            system_prompts.append(head + "\n" + "\n".join(body))

    cpt_data = tc.chars_per_token("\n".join(_record(rng, i) for i in range(40)))
    schema = {"type": "object", "properties": {
        "label": {"type": "string"}, "score": {"type": "integer", "minimum": 0, "maximum": 100},
        "ids": {"type": "array", "items": {"type": "string"}, "maxItems": 3}, "reason": {"type": "string"}},
        "required": ["label", "score", "ids", "reason"]}
    think = req.get("thinking_share", 0.0)
    base_out = max(64, int(req["avg_output_tokens"]))   # answer tokens; thinking is separate
    think_tok = max(0, int(req.get("avg_thinking_tokens") or 0))   # as the cost model (effective_output_tokens)

    def make(n: int, r: random.Random, scale: float, total: int | None = None,
             max_tokens: int | None = None) -> tuple[dict, int]:
        si = r.randrange(n_sys)
        total = total or _sample_input(r, req)
        budget_chars = max(400, int((total - sys_sizes[si]) * cpt_data * scale))
        lines, size, k = [], 0, n * 1_000_000
        while size < budget_chars:
            line = _record(r, k); lines.append(line); size += len(line) + 1; k += 1
        thinking = r.random() < think
        # Long prompts (>32K) use schema-sized output budget (~400 tokens) so long context doesn't artificially inflate output length
        max_tok = min(400, int(base_out * 2)) if total > 32000 and not thinking else int(base_out * 2 + (think_tok * 1.5 if thinking else 0))
        rec = {"id": f"p{n:05d}", "system_prompt": si, "system_tokens": sys_sizes[si], "input_tokens": total,
               "messages": [{"role": "system", "content": system_prompts[si]},
                            {"role": "user", "content": f"Process these {len(lines)} records.\n" + "\n".join(lines)}],
               "max_tokens": max_tokens or max_tok}
        if think:
            rec["chat_template_kwargs"] = {"enable_thinking": thinking}
        # Guided JSON (response_format) only without thinking: on the Model Garden vLLM image for Gemma 4, thinking +
        # a JSON schema delays the first token by ~13 s idle and stalls the engine under load (streams cut at the
        # endpoint's ~30 s first-byte limit, health checks fail). Thinking requests still get the JSON instruction in
        # the system prompt, which is how reasoning apps usually ask for structure.
        if syn["json_output"] and not thinking:
            rec["response_format"] = {"type": "json_schema", "json_schema": {"name": "result", "schema": schema}}
        return rec, total

    scale = 1.0
    if tc.tok:
        pilot = [make(n, random.Random(f"pilot-{n}"), 1.0) for n in range(40)]
        want = sum(t for _, t in pilot)
        got = sum(tc.count(" ".join(m["content"] for m in rec["messages"])) for rec, _ in pilot)
        sys_part = sum(rec["system_tokens"] for rec, _ in pilot)
        scale = max(0.3, min(3.0, (want - sys_part) / max(1, got - sys_part)))
    return make, scale, rng


def level_requests(cfg: dict) -> int:
    """Requests the main load test sends (loadtest.run: max(requests_per_level, 2 x concurrency) per level)."""
    lt = cfg.get("loadtest") or {}
    per = int(lt.get("requests_per_level") or 0)
    return sum(max(per, 2 * int(c)) for c in lt.get("concurrency") or [])


def synthetic(cfg: dict, out: Path) -> Path:
    make, scale, rng = _builder(cfg, out.parent)
    count = int(cfg["prompts"]["synthetic"]["count"])
    need = level_requests(cfg)
    if count < need:
        print(f"note: {count} synthetic prompts for {need} load-test requests: some levels will reuse prompts "
              f"(raise prompts.synthetic.count to ~{math.ceil(1.1 * need)} for fresh prompts at every level)")
    with open(out, "w") as f:
        for n in range(count):
            f.write(json.dumps(make(n, rng, scale)[0]) + "\n")
    return out


LONG_SHORT_START = 500_000   # prompt indices well past the main set: fresh content, no stale prefix-cache hits


def long_short_output(cfg: dict, run_dir: Path, n_medium: int = 100, n_large: int = 50, max_tokens: int = 400,
                      start: int = LONG_SHORT_START) -> Path:
    """Long prompts (32K-128K and >128K) with schema-sized outputs (`max_tokens` cap) on fresh indices.

    Long outputs at long context inflate the measured per-token input cost; production long prompts
    usually return a short structured answer. Sizes follow the workload's medium_mean / large_mean.
    Only the system prompts are shared with the main set (as in production), never the user content.
    Run with: dilemma load -c CONFIG --prompts runs/<name>/prompts_long_shortout.jsonl --tag long_shortout
    """
    req = cfg["requests"]
    med_mean = float(req.get("medium_mean") or 69000)
    lrg_mean = float(req.get("large_mean") or 188000)
    make, scale, _ = _builder(cfg, run_dir)
    rng = random.Random(f"{cfg['name']}-long-shortout-{start}")
    out = run_dir / "prompts_long_shortout.jsonl"
    with open(out, "w") as f:
        for i in range(n_medium + n_large):
            if i < n_medium:
                total = int(min(128000, max(32500, med_mean * (0.65 + 0.7 * rng.random()))))
            else:
                total = int(max(128500, lrg_mean * (0.75 + 0.5 * rng.random())))
            rec, _ = make(start + i, rng, scale, total=total, max_tokens=max_tokens)
            rec["id"] = f"ls{start + i:07d}"
            rec["bucket"] = "medium" if i < n_medium else "large"
            rec["target_input_tokens"] = total
            if "chat_template_kwargs" in rec:   # thinking would blow the output cap
                rec["chat_template_kwargs"] = {"enable_thinking": False}
            f.write(json.dumps(rec) + "\n")
    print(f"{out}: {n_medium} medium + {n_large} large prompts, max_tokens {max_tokens}")
    return out


SHORT_ONLY_START = 800_000   # fresh indices: no prefix-cache hits on the main set's user content


def short_only(cfg: dict, run_dir: Path, n: int | None = None, start: int = SHORT_ONLY_START) -> Path:
    """Prompts <= 32K tokens shaped like the short part of the workload (for per-class calibration).

    Synthetic source: the config's short bucket (its mean, no long tail). File source: the prompts in
    prompts.jsonl that count <= 32K tokens. Writes prompts_short.jsonl.
    """
    out = run_dir / "prompts_short.jsonl"
    if cfg["prompts"]["source"] == "synthetic":
        from aiops_dilemma.costmodel import buckets
        short_cfg = copy.deepcopy(cfg)
        short_cfg["requests"].update(avg_input_tokens=int(buckets(cfg)["small"]["mean"]),
                                     pct_over_32k=0.0, pct_over_128k=0.0, pct_medium=0.0, pct_long=0.0,
                                     pct_short=100.0)
        normalize_buckets(short_cfg["requests"])
        make, scale, _ = _builder(short_cfg, run_dir)
        rng = random.Random(f"{cfg['name']}-short-{start}")
        count = n or cfg["prompts"]["synthetic"]["count"]
        with open(out, "w") as f:
            for i in range(count):
                rec, _ = make(start + i, rng, scale)
                rec["id"] = f"s{start + i:07d}"
                f.write(json.dumps(rec) + "\n")
    else:
        src = run_dir / "prompts.jsonl"
        if not src.exists():
            build(cfg, run_dir)
        tc = None
        kept = 0
        with open(src) as fi, open(out, "w") as fo:
            for line in fi:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("input_tokens"):    # counted at upload (tokenizer, the file's counts, or the estimate)
                    size = int(r["input_tokens"])
                else:
                    tc = tc or TokenCounter(_tokenizer(cfg["model"].get("hf_id"), run_dir))
                    size = tc.count(" ".join(m["content"] for m in r["messages"]))
                if size <= 32000:
                    fo.write(line if line.endswith("\n") else line + "\n")
                    kept += 1
        if not kept:
            raise SystemExit(f"{src}: no prompts of 32K tokens or less to calibrate with")
        print(f"{out}: {kept} of your prompts are 32K tokens or less")
    print(f"{out}: short-prompt set for per-class calibration")
    return out


def from_file(cfg: dict, out: Path) -> Path:
    """Your own prompts (prompts.file, relative to the workload's folder): custom system prompts for prompts without
    one, and the workload's answer length / thinking share where a prompt has none (userprompts.apply_defaults)."""
    from aiops_dilemma import userprompts
    src = Path(cfg["prompts"]["file"])
    if not src.is_absolute() and cfg.get("_path"):
        src = Path(cfg["_path"]).parent / src
    try:
        src = paths.check_readable(src)
    except paths.PathNotAllowed as e:
        raise SystemExit(str(e))
    if not src.exists():
        raise SystemExit(f"{src}: prompt file not found (upload your prompts again, or switch to synthetic prompts)")
    custom_sys = _load_custom_system_prompts(cfg)
    rows = []
    for line in open(src):
        if not line.strip():
            continue
        r = json.loads(line)
        if "messages" not in r:
            raise SystemExit(f"{src}: every line needs a 'messages' list")
        rows.append(r)
    rows = userprompts.apply_defaults(rows, cfg)
    with open(out, "w") as fo:
        for n, r in enumerate(rows):
            if custom_sys and not any(m.get("role") == "system" for m in r["messages"]):
                r["messages"] = [{"role": "system", "content": custom_sys[n % len(custom_sys)]}] + list(r["messages"])
                if r.get("input_tokens"):   # the added system prompt counts as input (estimate: chars / 3.5)
                    r["input_tokens"] = int(r["input_tokens"]) + int(len(r["messages"][0]["content"]) / FALLBACK_CHARS_PER_TOKEN)
            r.setdefault("id", f"p{n:05d}")
            fo.write(json.dumps(r) + "\n")
    print(f"imported {len(rows)} prompts from {src}")
    return out


def build(cfg: dict, run_dir: Path) -> Path:
    out = run_dir / "prompts.jsonl"
    path = synthetic(cfg, out) if cfg["prompts"]["source"] == "synthetic" else from_file(cfg, out)
    stats(cfg, path, run_dir)
    return path


def stats(cfg: dict, path: Path, run_dir: Path, sample: int = 300) -> dict:
    rows = [json.loads(line) for _, line in zip(range(sample), open(path))]
    if cfg["prompts"]["source"] == "file" and rows and all(r.get("input_tokens") for r in rows):
        sizes = sorted(int(r["input_tokens"]) for r in rows)    # counted at upload
    else:
        tc = TokenCounter(_tokenizer(cfg["model"].get("hf_id"), run_dir))
        sizes = sorted(tc.count(" ".join(m["content"] for m in r["messages"])) for r in rows)
    n = max(1, len(sizes))
    over32 = 100 * sum(x > 32000 for x in sizes) / n
    over128 = 100 * sum(x > 128000 for x in sizes) / n
    s = {
        "prompts_sampled": len(sizes),
        "avg_input": sum(sizes) / n,
        "median_input": sizes[len(sizes) // 2],
        "p95_input": sizes[int(0.95 * len(sizes))],
        "max_input": sizes[-1],
        "pct_short": round(100 - over32, 1),
        "pct_medium": round(over32 - over128, 1),
        "pct_long": round(over128, 1),
        "pct_over_32k": round(over32, 1),
        "pct_over_128k": round(over128, 1),
    }
    print("prompt set: " + ", ".join(f"{k}={v:,.0f}" if isinstance(v, float) else f"{k}={v}" for k, v in s.items()))
    return s
