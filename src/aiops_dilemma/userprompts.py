"""Bring your own prompts: parse prompt files or pasted text into the benchmark's prompt set.

Formats (parsed robustly; every file reports what was understood):
  - JSONL: OpenAI chat records (`messages`, also batch records with a `body`), or objects with a prompt / input /
    text / question field (+ optional system, max_tokens, response_format, chat_template_kwargs, input / output
    token counts).
  - JSON: an array of strings, an array of such objects, or {"prompts": [...]} (also data / rows / requests / items).
  - CSV / TSV: a prompt / text / input column (+ optional system / max_tokens / token-count columns).
  - TXT / MD: prompts separated by lines containing only `---`, blank-line separated blocks, or one per line.

The set is stored as `runs/<session>/uploaded_prompts.jsonl` (one chat record per line, `input_tokens` counted with
the model's tokenizer when available, else a characters-per-token estimate) with `uploaded_prompts_meta.json`
(files, what was understood, dropped prompts, how tokens were counted). `prompts.from_file` fills in answer lengths
and thinking from the workload's settings when a prompt has none (`apply_defaults`), so changing "Output tokens"
after the upload still applies. Nothing here sends a prompt anywhere: the benchmark sends them to your endpoint.
"""

import csv
import io
import json
import math
import re
import time
from pathlib import Path

STORE = "uploaded_prompts.jsonl"
META = "uploaded_prompts_meta.json"
SYSTEM_STORE = "uploaded_system_prompts.json"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_PROMPTS = 20_000
PREVIEW_CHARS = 240
CHARS_PER_TOKEN = 3.5
SHORT_MAX, MEDIUM_MAX = 32000, 128000
TEXT_MODES = ("separator", "blocks", "lines")

PROMPT_KEYS = ("prompt", "user_prompt", "input", "text", "question", "query", "content", "instruction")
SYS_KEYS = ("system_prompt", "system", "instructions", "sys_prompt", "system_message")
MAXTOK_KEYS = ("max_tokens", "max_output_tokens", "max_completion_tokens", "max_new_tokens")
IN_TOK_KEYS = ("input_tokens", "prompt_tokens", "in_tokens", "tokens_in", "input_token_count")
OUT_TOK_KEYS = ("output_tokens", "completion_tokens", "out_tokens", "tokens_out", "output_token_count")
TIME_KEYS = ("timestamp", "created_at", "time", "datetime", "request_time", "ts")
LIST_KEYS = ("prompts", "data", "rows", "requests", "items", "examples", "records")
_SEP_LINE = re.compile(r"^[ \t]*-{3,}[ \t]*$", re.M)


class PromptUploadError(ValueError):
    def __init__(self, message: str, status: int = 400, code: str = "bad_request", section: str | None = None):
        super().__init__(message)
        self.status, self.code, self.section = status, code, section


# ------------------------------------------------------------------------------------------------------ parsing
def _lower(obj: dict) -> dict:
    return {str(k).strip().lower(): v for k, v in obj.items() if k is not None}


def _pick(row: dict, keys: tuple[str, ...]):
    for k in keys:
        v = row.get(k)
        if v not in (None, ""):
            return k, v
    return None, None


def _int(v) -> int | None:
    try:
        n = int(float(v))
        return n if n > 0 else None
    except (TypeError, ValueError):
        return None


def _content_text(c) -> str:
    """Message content as text: a string, or OpenAI content parts ([{type: text, text}, ...])."""
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "\n".join(str(p.get("text") or "") if isinstance(p, dict) else str(p) for p in c).strip()
    return "" if c is None else str(c)


def _jsonish(v):
    if isinstance(v, str) and v.strip()[:1] in "[{":
        try:
            return json.loads(v)
        except ValueError:
            return v
    return v


def record_from(obj, fields: set | None = None) -> dict | None:
    """One prompt record from a string or an object (None when it holds no prompt text). `fields` collects what
    was understood (messages, prompt, system, max_tokens, ...)."""
    fields = fields if fields is not None else set()
    if isinstance(obj, str):
        fields.add("text")
        return {"messages": [{"role": "user", "content": obj.strip()}]}    # blank: dropped (and counted) by build
    if not isinstance(obj, dict):
        return None
    row = _lower(obj)
    if isinstance(row.get("body"), dict) and ("messages" in _lower(row["body"]) or "prompt" in _lower(row["body"])):
        row = {**row, **_lower(row["body"])}   # OpenAI batch request lines
    msgs = _jsonish(row.get("messages"))
    rec: dict = {}
    if isinstance(msgs, list) and msgs:
        out = []
        for m in msgs:
            if isinstance(m, dict) and m.get("role"):
                text = _content_text(m.get("content"))
                if text.strip():
                    out.append({"role": str(m["role"]), "content": text})
        if out and any(m["role"] != "system" for m in out):
            rec["messages"] = out
            fields.add("messages")
    elif isinstance(msgs, str) and msgs.strip():
        rec["messages"] = [{"role": "user", "content": msgs.strip()}]
        fields.add("messages")
    if "messages" not in rec:
        k, text = _pick(row, PROMPT_KEYS)
        if text is None or not str(_content_text(text)).strip():
            return None
        fields.add(k)
        rec["messages"] = [{"role": "user", "content": _content_text(text).strip()}]
        sk, sys_txt = _pick(row, SYS_KEYS)
        if sys_txt is not None and str(sys_txt).strip():
            fields.add(sk)
            rec["messages"].insert(0, {"role": "system", "content": str(sys_txt).strip()})
    k, v = _pick(row, MAXTOK_KEYS)
    if _int(v):
        rec["max_tokens"] = _int(v)
        fields.add("max_tokens")
    k, v = _pick(row, IN_TOK_KEYS)
    if _int(v):
        rec["file_input_tokens"] = _int(v)
        fields.add("input token counts")
    k, v = _pick(row, OUT_TOK_KEYS)
    if _int(v):
        rec["output_tokens"] = _int(v)
        fields.add("output token counts")
    rf = _jsonish(row.get("response_format"))
    if isinstance(rf, dict) and rf:
        rec["response_format"] = rf
        fields.add("response_format")
    ck = _jsonish(row.get("chat_template_kwargs"))
    if isinstance(ck, dict) and ck:
        rec["chat_template_kwargs"] = ck
        fields.add("chat_template_kwargs")
    if row.get("id") not in (None, ""):
        rec["source_id"] = str(row["id"])[:80]
    return rec


def split_text(text: str, mode: str = "separator") -> list[str]:
    """Prompts in plain text. separator: lines holding only `---` (no such line = the whole text is one prompt);
    blocks: blank-line separated blocks; lines: one prompt per non-empty line."""
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    if mode == "lines":
        parts = text.split("\n")
    elif mode == "blocks":
        parts = re.split(r"\n[ \t]*\n", text)
    else:
        parts = _SEP_LINE.split(text)
    return [p.strip() for p in parts if p.strip()]


def _decode(raw: bytes) -> str:
    if raw[:3] == b"\xef\xbb\xbf":
        raw = raw[3:]
    return raw.decode("utf-8", errors="replace")


def _rows(fname: str, raw: bytes) -> tuple[list, str, int]:
    """(items, format label, unreadable lines) for structured files; items are strings or dicts."""
    ext = Path(fname).suffix.lower()
    if ext == ".parquet":
        try:
            import pyarrow.parquet as pq
            return pq.read_table(io.BytesIO(raw)).to_pylist(), "Parquet", 0
        except Exception as e:
            raise PromptUploadError(f"{fname}: couldn't read the Parquet file ({e}).")
    text = _decode(raw).strip()
    if not text:
        return [], "empty", 0
    if ext == ".jsonl" or (ext not in (".csv", ".tsv", ".json") and text[:1] == "{" and "\n{" in text):
        items, bad = [], 0
        for line in text.split("\n"):
            if not line.strip():
                continue
            try:
                items.append(json.loads(line))
            except ValueError:
                bad += 1
        if not items and bad:
            raise PromptUploadError(f"{fname}: no line is valid JSON. JSONL needs one JSON object per line.")
        return items, "JSONL", bad
    if ext == ".json" or text[:1] in "[{":
        try:
            data = json.loads(text)
        except ValueError as e:
            raise PromptUploadError(f"{fname}: not valid JSON ({e}).")
        if isinstance(data, dict):
            low = _lower(data)
            lst = next((low[k] for k in LIST_KEYS if isinstance(low.get(k), list)), None)
            data = lst if lst is not None else [data]
        if not isinstance(data, list):
            raise PromptUploadError(f"{fname}: expected an array of prompts or {{\"prompts\": [...]}}.")
        return data, "JSON", 0
    first = text.split("\n", 1)[0]
    delim = "\t" if ext == ".tsv" or (first.count("\t") > first.count(",")) else ","
    try:
        rows = list(csv.DictReader(io.StringIO(text), delimiter=delim))
    except csv.Error as e:
        raise PromptUploadError(f"{fname}: couldn't read the CSV ({e}).")
    return rows, "TSV" if delim == "\t" else "CSV", 0


STRUCTURED_EXT = (".jsonl", ".json", ".csv", ".tsv", ".parquet")
TEXT_EXT = (".txt", ".md")
ACCEPTED_EXT = STRUCTURED_EXT + TEXT_EXT


def parse_file(fname: str, raw: bytes, text_mode: str = "separator") -> tuple[list[dict], dict]:
    """(records, info) for one file. info: filename, format, items read, prompts found, fields understood, skipped."""
    ext = Path(fname).suffix.lower()
    if ext in (".pdf", ".docx"):
        raise PromptUploadError(f"{fname} looks like a workload description: use Upload in Traffic.",
                                code="wrong_section", section="traffic")
    if ext not in ACCEPTED_EXT:
        raise PromptUploadError(f"{fname}: unsupported file type. Use .jsonl, .json, .csv, .tsv, .txt or .md.",
                                status=415, code="unsupported")
    if ext in TEXT_EXT:
        mode = text_mode if text_mode in TEXT_MODES else "separator"
        parts = split_text(_decode(raw), mode)
        label = {"separator": "text, prompts separated by ---", "blocks": "text, blank-line separated prompts",
                 "lines": "text, one prompt per line"}[mode]
        recs = [{"messages": [{"role": "user", "content": p}]} for p in parts]
        return recs, {"filename": fname, "format": label, "items": len(parts), "prompts": len(recs),
                      "fields": ["text"], "skipped": 0}
    items, fmt, bad = _rows(fname, raw)
    fields: set = set()
    recs = []
    for it in items:
        r = record_from(it, fields)
        if r:
            recs.append(r)
    info = {"filename": fname, "format": fmt + (" (chat messages)" if "messages" in fields else ""),
            "items": len(items), "prompts": len(recs), "fields": sorted(fields),
            "skipped": len(items) - len(recs) + bad}
    if items and not recs:
        cols = sorted({str(k) for it in items[:50] if isinstance(it, dict) for k in it})[:12]
        kind = classify(fname, raw)
        if "timeseries" in kind["kinds"] or "hourly_traffic" in kind["kinds"]:
            raise PromptUploadError(f"{fname} looks like a traffic time series: use Upload in Traffic.",
                                    status=422, code="wrong_section", section="traffic")
        raise PromptUploadError(
            f"{fname}: no prompt text found. Expected a 'messages' list or a prompt / input / text / question "
            f"field{(' (found: ' + ', '.join(cols) + ')') if cols else ''}.", status=422, code="no_prompts")
    return recs, info


# ---------------------------------------------------------------------------------- what a file holds (routing)
def classify(fname: str, raw: bytes) -> dict:
    """Which kinds of data a file holds, so each rail section takes its part and points to the others.
    kinds ⊆ {description, text, timeseries, hourly_traffic, request_timestamps, prompts, system_prompts,
    token_counts}; `primary` is the section that should receive it (traffic | system | prompts | unknown)."""
    from aiops_dilemma import ingest
    ext = Path(fname).suffix.lower()
    out = {"filename": fname, "kinds": [], "prompts": 0, "system_prompts": 0, "timestamps": 0, "series": 0,
           "primary": "unknown"}
    if ext in (".pdf", ".docx"):
        out.update(kinds=["description"], primary="traffic")
        return out
    if ext in TEXT_EXT:
        out.update(kinds=["text"], primary="prompts")
        return out
    if ext not in STRUCTURED_EXT:
        return out
    try:
        series = ingest.parse_timeseries(fname, raw) if ext != ".parquet" else []
    except Exception:
        series = []
    if series:
        out.update(kinds=["timeseries"], series=len(series), primary="traffic")
        return out
    try:
        items, _, _ = _rows(fname, raw)
    except PromptUploadError:
        return out
    kinds: list[str] = []
    n_p = n_s = n_t = 0
    tok = traffic_col = False
    for it in items[:MAX_PROMPTS]:
        if isinstance(it, str):
            n_p += it.strip() != ""
            continue
        if not isinstance(it, dict):
            continue
        row = _lower(it)
        r = record_from(it)
        if r:
            n_p += 1
            n_s += any(m["role"] == "system" for m in r["messages"])
        else:
            msgs = _jsonish(row.get("messages"))
            if (isinstance(msgs, list) and any(isinstance(m, dict) and m.get("role") == "system" for m in msgs)) \
                    or _pick(row, SYS_KEYS)[1]:
                n_s += 1
        if _pick(row, TIME_KEYS)[1] is not None:
            n_t += 1
        if _pick(row, IN_TOK_KEYS)[1] is not None:
            tok = True
        if _pick(row, ingest.TRAFFIC_COLS)[1] is not None:
            traffic_col = True
    if traffic_col and not n_p and not tok:
        kinds.append("hourly_traffic")
    if n_t >= 24:
        kinds.append("request_timestamps")
    if n_p:
        kinds.append("prompts")
    if n_s:
        kinds.append("system_prompts")
    if tok:
        kinds.append("token_counts")
    primary = ("prompts" if n_p else "traffic" if ("hourly_traffic" in kinds or "request_timestamps" in kinds or tok)
               else "system" if n_s else "unknown")
    out.update(kinds=kinds, prompts=n_p, system_prompts=n_s, timestamps=n_t, primary=primary)
    return out


def system_prompts_from(fname: str, raw: bytes) -> list[str]:
    """System prompts in a file: text split on `---` lines, a JSON array of strings, or the system messages /
    columns of a request log (most frequent first)."""
    ext = Path(fname).suffix.lower()
    if ext in TEXT_EXT:
        return split_text(_decode(raw), "separator")
    items, _, _ = _rows(fname, raw)
    seen: dict[str, int] = {}
    for it in items:
        texts = []
        if isinstance(it, str):
            texts = [it]
        elif isinstance(it, dict):
            row = _lower(it)
            msgs = _jsonish(row.get("messages"))
            if isinstance(msgs, list):
                texts = [_content_text(m.get("content")) for m in msgs if isinstance(m, dict) and m.get("role") == "system"]
            s = _pick(row, SYS_KEYS)[1]
            if s:
                texts.append(str(s))
            if not texts and set(row) <= {"content", "text", "prompt", "id", "name"}:
                texts = [str(row.get("content") or row.get("text") or row.get("prompt") or "")]
        for t in texts:
            t = t.strip()
            if t:
                seen[t] = seen.get(t, 0) + 1
    return [t for t, _ in sorted(seen.items(), key=lambda kv: -kv[1])]


# ------------------------------------------------------------------------------------------------- the stored set
def _key(rec: dict) -> str:
    return json.dumps(rec["messages"], sort_keys=True)


def token_counter(hf_id: str | None, cache_dir: Path):
    """(TokenCounter, method): the model's tokenizer when it loads, else the characters-per-token estimate."""
    from aiops_dilemma import prompts
    tok = prompts._tokenizer(hf_id, cache_dir) if hf_id else None
    return prompts.TokenCounter(tok), ("tokenizer" if tok else "estimate")


def build(parsed: list[tuple[list[dict], dict]], tc=None, method: str = "estimate", tokenizer: str | None = None
          ) -> tuple[list[dict], dict]:
    """Merge parsed files: drop empty and duplicate prompts, cap the count, count input tokens. Returns
    (records, meta)."""
    from aiops_dilemma import prompts
    tc = tc or prompts.TokenCounter(None)
    recs, files, seen = [], [], set()
    empty = dup = 0
    for rs, info in parsed:
        files.append(info)
        for r in rs:
            if not any(m["content"].strip() for m in r["messages"] if m["role"] != "system"):
                empty += 1
                continue
            k = _key(r)
            if k in seen:
                dup += 1
                continue
            seen.add(k)
            recs.append(r)
    if not recs:
        raise PromptUploadError("No prompts found in what you uploaded.", status=422, code="no_prompts")
    if len(recs) > MAX_PROMPTS:
        raise PromptUploadError(f"{len(recs):,} prompts is more than the limit of {MAX_PROMPTS:,}. Upload a "
                                f"representative sample of up to {MAX_PROMPTS:,}.", status=413, code="too_many")
    counted = {"file": 0, method: 0}
    out = []
    for i, r in enumerate(recs):
        rec = {"id": f"u{i + 1:05d}", "messages": r["messages"]}
        if r.get("file_input_tokens"):
            rec["input_tokens"], src = r["file_input_tokens"], "file"
        else:
            rec["input_tokens"] = max(1, tc.count("\n".join(m["content"] for m in r["messages"])))
            src = method
        counted[src] += 1
        for k in ("max_tokens", "output_tokens", "response_format", "chat_template_kwargs", "source_id"):
            if r.get(k) is not None:
                rec[k] = r[k]
        out.append(rec)
    meta = {"uploaded_at": time.time(), "count": len(out), "files": files,
            "dropped": {"empty": empty, "duplicates": dup},
            "tokens": {"method": method, "tokenizer": tokenizer if method == "tokenizer" else None,
                       "from_file": counted["file"], "counted": counted[method]}}
    return out, meta


def save(run_dir: Path, records: list[dict], meta: dict) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp = run_dir / (STORE + ".tmp")
    with open(tmp, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    tmp.replace(run_dir / STORE)
    (run_dir / META).write_text(json.dumps(meta, indent=2))
    return meta


def store_files(run_dir: Path, files: list[tuple[str, bytes]], text_mode: str = "separator",
                hf_id: str | None = None, tc=None, method: str | None = None) -> dict:
    """Parse files, save the set in the session folder, return its meta."""
    total = sum(len(b) for _, b in files)
    if total > MAX_UPLOAD_BYTES:
        raise PromptUploadError(f"Files are too large ({total / 1e6:.0f} MB; limit {MAX_UPLOAD_BYTES // 2**20} MB). "
                                "Upload a representative sample.", status=413, code="too_large")
    parsed = [parse_file(n, b, text_mode) for n, b in files]
    if tc is None:
        tc, method = token_counter(hf_id, run_dir)
    recs, meta = build(parsed, tc, method or "estimate", hf_id)
    return save(run_dir, recs, meta)


def store_text(run_dir: Path, text: str, mode: str = "separator", hf_id: str | None = None, tc=None,
               method: str | None = None) -> dict:
    if len((text or "").encode()) > MAX_UPLOAD_BYTES:
        raise PromptUploadError("The pasted text is too large (limit 50 MB).", status=413, code="too_large")
    recs = [{"messages": [{"role": "user", "content": p}]} for p in split_text(text, mode)]
    label = {"separator": "pasted text, prompts separated by ---", "blocks": "pasted text, blank-line separated",
             "lines": "pasted text, one prompt per line"}.get(mode, "pasted text")
    if tc is None:
        tc, method = token_counter(hf_id, run_dir)
    recs, meta = build([(recs, {"filename": "pasted", "format": label, "items": len(recs), "prompts": len(recs),
                                "fields": ["text"], "skipped": 0})], tc, method or "estimate", hf_id)
    return save(run_dir, recs, meta)


def remove(run_dir: Path) -> bool:
    found = False
    for n in (STORE, META):
        p = run_dir / n
        if p.exists():
            p.unlink()
            found = True
    return found


def load(run_dir: Path) -> tuple[list[dict], dict] | None:
    p = run_dir / STORE
    if not p.exists():
        return None
    recs = [json.loads(line) for line in open(p) if line.strip()]
    try:
        meta = json.loads((run_dir / META).read_text())
    except (OSError, ValueError):
        meta = {"uploaded_at": p.stat().st_mtime, "count": len(recs), "files": [], "dropped": {},
                "tokens": {"method": "unknown"}}
    return recs, meta


# ----------------------------------------------------------------------------- answer length and thinking defaults
def thinking_flags(n: int, share: float) -> list[bool]:
    """Deterministic: prompt j has thinking on when floor((j+1)·share) > floor(j·share) (evenly spread, exact share)."""
    share = max(0.0, min(1.0, float(share or 0)))
    return [math.floor((j + 1) * share + 1e-9) > math.floor(j * share + 1e-9) for j in range(n)]


def apply_defaults(records: list[dict], cfg: dict) -> list[dict]:
    """Fill each prompt's answer budget and thinking from the workload when the file didn't give them:
    max_tokens = the file's output count, else "Output tokens (answer)"; the configured thinking share of the prompts
    without chat_template_kwargs gets thinking on (deterministic) and max_tokens + "Thinking tokens"."""
    req = cfg.get("requests") or {}
    out_default = max(16, int(req.get("avg_output_tokens") or 400))
    share = float(req.get("thinking_share") or 0)
    think_tok = int(req.get("avg_thinking_tokens") or 2 * out_default)
    free = [i for i, r in enumerate(records) if not r.get("chat_template_kwargs")]
    flags = dict(zip(free, thinking_flags(len(free), share)))
    res = []
    for i, r in enumerate(records):
        r = dict(r)
        thinking = flags.get(i, False)
        if i in flags and share > 0:
            r["chat_template_kwargs"] = {"enable_thinking": bool(thinking)}
        if not r.get("max_tokens"):
            base = int(r["output_tokens"]) if r.get("output_tokens") else out_default
            r["max_tokens"] = base + (think_tok if thinking else 0)
        res.append(r)
    return res


# ------------------------------------------------------------------------------------------------------- summary
def _pctl(sorted_vals: list[int], q: float) -> int:
    if not sorted_vals:
        return 0
    return int(sorted_vals[min(len(sorted_vals) - 1, int(q * len(sorted_vals)))])


def _user_text(rec: dict) -> str:
    users = [m["content"] for m in rec["messages"] if m["role"] != "system"]
    return users[-1] if users else ""


def proposal(records: list[dict], cfg: dict) -> list[dict]:
    """Request sizes measured on the prompts, for the "Update request sizes from my prompts" review list."""
    req = cfg.get("requests") or {}
    toks = [int(r.get("input_tokens") or 0) for r in records]
    n = max(1, len(toks))
    med = [t for t in toks if SHORT_MAX < t <= MEDIUM_MAX]
    lrg = [t for t in toks if t > MEDIUM_MAX]
    items = [("requests.avg_input_tokens", round(sum(toks) / n)),
             ("requests.pct_over_32k", round(100 * (len(med) + len(lrg)) / n, 2)),
             ("requests.pct_over_128k", round(100 * len(lrg) / n, 2))]
    if med:
        items.append(("requests.medium_mean", round(sum(med) / len(med))))
    if lrg:
        items.append(("requests.large_mean", round(sum(lrg) / len(lrg))))
    outs = [int(r["output_tokens"]) for r in records if r.get("output_tokens")]
    if outs and len(outs) >= 0.5 * n:     # only when most prompts came with their answer length
        items.append(("requests.avg_output_tokens", round(sum(outs) / len(outs))))
    return [{"path": p, "value": v, "current": req.get(p.split(".", 1)[1]),
             "same": req.get(p.split(".", 1)[1]) is not None and abs(float(req.get(p.split(".", 1)[1])) - v) < 1e-6}
            for p, v in items]


def summary(run_dir: Path, cfg: dict, context: int | None = None) -> dict | None:
    """What the session's uploaded set holds, how the benchmark will use it, and warnings. None = synthetic."""
    from aiops_dilemma import prompts
    got = load(run_dir)
    if not got:
        return None
    recs, meta = got
    n = len(recs)
    toks = sorted(int(r.get("input_tokens") or 0) for r in recs)
    over32 = sum(t > SHORT_MAX for t in toks)
    over128 = sum(t > MEDIUM_MAX for t in toks)
    with_out = sum(1 for r in recs if r.get("output_tokens"))
    with_max = sum(1 for r in recs if r.get("max_tokens"))
    with_sys = sum(1 for r in recs if any(m["role"] == "system" for m in r["messages"]))
    with_kw = sum(1 for r in recs if r.get("chat_template_kwargs"))
    req = cfg.get("requests") or {}
    out_default = max(16, int(req.get("avg_output_tokens") or 400))
    share = float(req.get("thinking_share") or 0)
    think_tok = int(req.get("avg_thinking_tokens") or 2 * out_default)
    n_default = sum(1 for r in recs if not r.get("max_tokens") and not r.get("output_tokens"))
    n_think = sum(thinking_flags(n - with_kw, share)) if share > 0 else 0
    custom_sys = prompts._load_custom_system_prompts(cfg)
    lt = cfg.get("loadtest") or {}
    requests = prompts.level_requests(cfg)
    warnings = []
    if requests > n:
        factor = requests / n
        conc = ", ".join(str(c) for c in lt.get("concurrency") or [])
        warnings.append(("repeat", f"{n:,} prompts for {requests:,} load-test requests (concurrency {conc}, "
                        f"{int(lt.get('requests_per_level') or 0):,} per level): prompts repeat across levels, each "
                        f"sent about {factor:.1f} times. Prefix caching may flatter repeated requests; upload "
                        f"{requests:,} or more prompts for fresh ones at every level."))
    if context:
        too_long = sum(1 for t in toks if t > context)
        if too_long:
            warnings.append(("context", f"{too_long:,} {'prompt is' if too_long == 1 else 'prompts are'} longer than the model's "
                            f"{context:,}-token context: the endpoint will reject {'it' if too_long == 1 else 'them'}."))
    d = meta.get("dropped") or {}
    if d.get("empty") or d.get("duplicates"):
        bits = [f"{d['empty']:,} empty" if d.get("empty") else "", f"{d['duplicates']:,} duplicate" if d.get("duplicates") else ""]
        warnings.append(("dropped", f"Dropped {' and '.join(b for b in bits if b)} {'prompt' if (d.get('empty', 0) + d.get('duplicates', 0)) == 1 else 'prompts'}."))
    skipped = sum(int(f.get("skipped") or 0) for f in meta.get("files") or [])
    if skipped:
        warnings.append(("skipped", f"Skipped {skipped:,} rows without prompt text or with invalid JSON."))
    method = (meta.get("tokens") or {}).get("method")
    if method == "estimate" and (meta.get("tokens") or {}).get("counted"):
        warnings.append(("estimate", "Token counts are estimates (about 3.5 characters per token): the model's tokenizer wasn't "
                        "available."))
    answer = {"default_max_tokens": out_default, "prompts_using_default": n_default,
              "prompts_with_max_tokens": with_max, "prompts_with_output_tokens": with_out,
              "thinking_share": share, "thinking_tokens": think_tok, "prompts_thinking": n_think,
              "prompts_with_template_kwargs": with_kw}
    if n_default == n:
        answer["text"] = (f"No answer lengths in the file: every prompt gets your \"Output tokens (answer)\" setting, "
                          f"{out_default:,} tokens.")
    elif n_default:
        answer["text"] = (f"{n - n_default:,} prompts bring their own answer length; the other {n_default:,} get your "
                          f"\"Output tokens (answer)\" setting, {out_default:,} tokens.")
    else:
        answer["text"] = "Every prompt brings its own answer length (max_tokens or output token counts from the file)."
    if n_think:
        answer["text"] += f" {n_think:,} of them ({share:.0%}) run with thinking on, +{think_tok:,} tokens."
    return {
        "source": "file", "count": n, "uploaded_at": meta.get("uploaded_at"), "files": meta.get("files") or [],
        "dropped": d, "tokens": meta.get("tokens") or {},
        "input": {"median": _pctl(toks, 0.5), "p95": _pctl(toks, 0.95), "max": toks[-1] if toks else 0,
                  "avg": round(sum(toks) / max(1, n)), "pct_over_32k": round(100 * over32 / max(1, n), 2),
                  "pct_over_128k": round(100 * over128 / max(1, n), 2), "over_32k": over32, "over_128k": over128},
        "with_system": with_sys, "system_added": (n - with_sys) if custom_sys else 0,
        "custom_system_prompts": len(custom_sys),
        "answer": answer,
        "requests": requests, "repeat_factor": round(requests / n, 2) if requests > n else 0,
        "preview": [{"text": (_user_text(r)[:PREVIEW_CHARS] + ("…" if len(_user_text(r)) > PREVIEW_CHARS else "")),
                     "tokens": int(r.get("input_tokens") or 0),
                     "system": any(m["role"] == "system" for m in r["messages"])} for r in recs[:3]],
        "warnings": [{"kind": k, "text": t} for k, t in warnings],
        "proposal": proposal(recs, cfg),
        "context": context,
    }


def describe_source(cfg: dict, run_dir: Path | None) -> dict:
    """{kind: file | synthetic, count, uploaded_at, text} for the Benchmark tab and report.md."""
    p = cfg.get("prompts") or {}
    if p.get("source") == "file":
        got = load(Path(run_dir)) if run_dir and p.get("file") and Path(p["file"]).name == STORE else None
        if got:
            recs, meta = got
            when = time.strftime("%Y-%m-%d", time.localtime(meta.get("uploaded_at") or time.time()))
            return {"kind": "file", "count": len(recs), "uploaded_at": meta.get("uploaded_at"),
                    "text": f"your {len(recs):,} prompts (uploaded {when})"}
        return {"kind": "file", "count": None, "uploaded_at": None, "text": f"your prompts ({p.get('file')})"}
    return {"kind": "synthetic", "count": int((p.get("synthetic") or {}).get("count") or 0),
            "text": "synthetic, shaped like your request sizes"}


def file_has_long_prompts(cfg: dict, run_dir: Path) -> bool | None:
    """For file prompt sets: are any over 32K tokens (prompts.jsonl, else the upload)? None for synthetic sets."""
    if (cfg.get("prompts") or {}).get("source") != "file":
        return None
    from aiops_dilemma import loadtest
    for name in ("prompts.jsonl", STORE):
        path = Path(run_dir) / name
        if path.exists():
            return any(loadtest.prompt_tokens_estimate(json.loads(line)) > SHORT_MAX for line in open(path)
                       if line.strip())
    return False
