"""Smart ingestion for production traffic curves, request logs, specific user prompts, and system prompts.

Supports .csv, .tsv, .jsonl, .json, .txt, .md, and .parquet files.
Automatically classifies uploaded files and derives:
  1. Weekly 168-hour traffic profile (peak / average / floor input tok/s)
  2. Request token distribution across Short (<=32K), Medium (32K-128K), and Long (>128K) buckets
  3. Shared system prompt prefix sizes and cache hit ratio
  4. Normalized OpenAI-format prompts.jsonl and custom system prompts for load testing
"""

import csv
import io
import json
import logging
import math
import os
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from aiops_dilemma import prompts, traffic

TRAFFIC_COLS = ("input_tokens_per_sec", "tps", "input_tps", "tokens_per_sec", "hourly_input_tps")
INPUT_TOK_COLS = ("input_tokens", "prompt_tokens", "in_tokens", "tokens_in", "input_token_count")
OUTPUT_TOK_COLS = ("output_tokens", "completion_tokens", "out_tokens", "tokens_out", "output_token_count")
TIME_COLS = ("timestamp", "created_at", "time", "datetime", "request_time", "ts")
PROMPT_COLS = ("prompt", "user_prompt", "input", "query", "question", "content", "text")
SYS_COLS = ("system_prompt", "system", "instructions", "sys_prompt")


def _first_present(row: dict, candidates: tuple[str, ...]):
    lower_map = {str(k).strip().lower(): v for k, v in row.items() if k is not None}
    for c in candidates:
        if c in lower_map and lower_map[c] not in (None, ""):
            return lower_map[c]
    return None


def _parse_ts(val) -> datetime | None:
    if val is None or val == "":
        return None
    if isinstance(val, (int, float)):
        t = float(val)
        if t > 1e12:
            t /= 1000.0
        return datetime.fromtimestamp(t, tz=timezone.utc)
    s = str(val).strip()
    try:
        if s.replace(".", "", 1).isdigit():
            t = float(s)
            if t > 1e12:
                t /= 1000.0
            return datetime.fromtimestamp(t, tz=timezone.utc)
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _read_records(filename: str, raw_bytes: bytes, role_hint: str = "auto") -> tuple[list[dict], list[str]]:
    """Parse raw file bytes into a list of dict records or raw text blocks (for system prompts)."""
    ext = Path(filename).suffix.lower()
    if role_hint == "system_prompts" or ext in (".txt", ".md"):
        text = raw_bytes.decode("utf-8", errors="replace").strip()
        if ext == ".json":
            data = json.loads(text)
            if isinstance(data, list):
                return [], [str(x["content"] if isinstance(x, dict) and "content" in x else x) for x in data if x]
        # Split on "---" if multiple system prompts in one .txt/.md file, else treat as 1 system prompt
        parts = [p.strip() for p in text.split("\n---\n") if p.strip()] if "\n---\n" in text else ([text] if text else [])
        return [], parts

    if ext == ".parquet":
        try:
            import pyarrow.parquet as pq
            table = pq.read_table(io.BytesIO(raw_bytes))
            return table.to_pylist(), []
        except Exception as e:
            raise ValueError(f"Could not read Parquet file {filename}: {e}")

    text = raw_bytes.decode("utf-8", errors="replace").strip()
    if not text:
        return [], []

    if ext == ".jsonl" or (ext not in (".csv", ".tsv") and text.startswith("{") and "\n{" in text):
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        return rows, []

    if ext == ".json" or text.startswith("[") or text.startswith("{"):
        data = json.loads(text)
        if isinstance(data, list):
            if data and isinstance(data[0], str):
                return [], data
            return [r for r in data if isinstance(r, dict)], []
        if isinstance(data, dict):
            for k in ("rows", "data", "requests", "prompts", "traffic"):
                if isinstance(data.get(k), list):
                    return data[k], []
            return [data], []

    delimiter = "\t" if ext == ".tsv" else ","
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    return list(reader), []


def ingest_files(
    files: list[tuple[str, bytes, str]],
    run_dir: Path,
    hf_id: str | None = None,
    avg_input_fallback: float = 6000.0,
) -> dict:
    """Ingest one or more uploaded files and return extracted traffic, prompt, and bucket statistics.

    Each item in `files` is (filename, raw_bytes, role_hint) where role_hint in ("auto", "traffic", "prompts", "system_prompts").
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    tc = prompts.TokenCounter(prompts._tokenizer(hf_id, run_dir))

    hourly_series: list[float] = []
    prompt_records: list[dict] = []
    system_prompts: list[str] = []
    in_tok_samples: list[int] = []
    out_tok_samples: list[int] = []
    timestamps: list[tuple[datetime, int]] = []
    files_summary: list[dict] = []

    timeseries_files: list[str] = []
    prompt_files: list[tuple[str, bytes]] = []   # stored as the benchmark's prompt set (userprompts, same as /prompts)
    for fname, raw, hint in files:
        # Timestamped series (Metrics Explorer exports, `timestamp,value` CSVs) go through the review flow
        # (parse_timeseries + POST /api/sessions/{name}/traffic/import), never applied silently here.
        if hint not in ("prompts", "system_prompts") and Path(fname).suffix.lower() in (".csv", ".tsv", ".json", ".jsonl"):
            try:
                found = parse_timeseries(fname, raw)
            except Exception:
                found = []
            if found:
                timeseries_files.append(fname)
                files_summary.append({"filename": fname, "detected_type": "timeseries", "series": len(found)})
                continue
        if hint == "prompts" and Path(fname).suffix.lower() in (".txt", ".md"):
            prompt_files.append((fname, raw))      # plain-text prompts (--- separated): userprompts
            files_summary.append({"filename": fname, "detected_type": "prompts", "count": None})
            continue
        rows, sys_texts = _read_records(fname, raw, role_hint=hint)
        if sys_texts:
            system_prompts.extend(sys_texts)
            files_summary.append({"filename": fname, "detected_type": "system_prompts", "count": len(sys_texts)})
            continue
        if not rows:
            continue

        sample0 = rows[0]
        # Case 1: Hourly traffic curve (168 rows or explicit traffic column without messages/prompts)
        t_val0 = _first_present(sample0, TRAFFIC_COLS)
        has_prompt_col = "messages" in sample0 or _first_present(sample0, PROMPT_COLS) is not None
        if hint == "traffic" or (t_val0 is not None and not has_prompt_col and _first_present(sample0, INPUT_TOK_COLS) is None):
            vals = []
            for r in rows:
                v = _first_present(r, TRAFFIC_COLS)
                if v is not None:
                    vals.append(float(v))
                elif "rps" in r or "requests_per_sec" in r:
                    rps = float(r.get("rps") or r.get("requests_per_sec") or 0)
                    vals.append(rps * avg_input_fallback)
            if vals:
                if len(vals) < 168:
                    # Tile / interpolate to 168 hours if daily (24h) or partial series provided
                    reps = (168 + len(vals) - 1) // len(vals)
                    vals = (vals * reps)[:168]
                elif len(vals) > 168:
                    vals = vals[:168]
                hourly_series = vals
                files_summary.append({"filename": fname, "detected_type": "hourly_traffic", "rows": len(rows)})
                continue

        # Case 2: Request / Prompt logs
        f_prompts = 0
        for idx, r in enumerate(rows):
            msgs = r.get("messages")
            sys_txt = _first_present(r, SYS_COLS)
            usr_txt = _first_present(r, PROMPT_COLS)
            if isinstance(msgs, str):
                try:
                    msgs = json.loads(msgs)
                except Exception:
                    msgs = [{"role": "user", "content": msgs}]

            if not msgs and (usr_txt or sys_txt):
                msgs = []
                if sys_txt:
                    msgs.append({"role": "system", "content": str(sys_txt)})
                if usr_txt:
                    msgs.append({"role": "user", "content": str(usr_txt)})

            in_tok = _first_present(r, INPUT_TOK_COLS)
            out_tok = _first_present(r, OUTPUT_TOK_COLS)

            if msgs and isinstance(msgs, list):
                for m in msgs:
                    if hint != "traffic" and m.get("role") == "system" and m.get("content"):
                        system_prompts.append(str(m["content"]))
                calc_in = tc.count(" ".join(str(m.get("content", "")) for m in msgs))
                in_count = int(float(in_tok)) if in_tok is not None else max(1, calc_in)
                out_count = int(float(out_tok)) if out_tok is not None else int(r.get("max_tokens") or 400)
                rec = {
                    "id": str(r.get("id") or f"u{ len(prompt_records) + 1:05d}"),
                    "messages": msgs,
                    "max_tokens": int(r.get("max_tokens") or max(64, out_count)),
                    "input_tokens": in_count,
                }
                if r.get("response_format"):
                    rec["response_format"] = r["response_format"]
                prompt_records.append(rec)
                in_tok_samples.append(in_count)
                out_tok_samples.append(out_count)
                f_prompts += 1
            elif in_tok is not None:
                in_count = int(float(in_tok))
                out_count = int(float(out_tok)) if out_tok is not None else 400
                in_tok_samples.append(in_count)
                out_tok_samples.append(out_count)
                if sys_txt:
                    system_prompts.append(str(sys_txt))
            else:
                continue

            ts_raw = _first_present(r, TIME_COLS)
            dt = _parse_ts(ts_raw)
            if dt is not None:
                timestamps.append((dt, in_count))

        if f_prompts and hint != "traffic":
            prompt_files.append((fname, raw))
        files_summary.append({
            "filename": fname,
            "detected_type": "prompt_logs" if f_prompts else "token_logs",
            "rows": len(rows),
            "prompts_extracted": f_prompts,
        })

    # Derive hourly traffic from timestamps if no explicit hourly series was uploaded
    if not hourly_series and len(timestamps) >= 24:
        timestamps.sort(key=lambda x: x[0])
        span_s = max(3600.0, (timestamps[-1][0] - timestamps[0][0]).total_seconds())
        weeks = max(1.0, span_s / (168.0 * 3600.0))
        buckets_168 = np.zeros(168, dtype=float)
        for dt, tok in timestamps:
            how = dt.weekday() * 24 + dt.hour
            buckets_168[how] += tok
        # Convert hourly token totals into tokens/sec
        hourly_tps = buckets_168 / (3600.0 * weeks)
        if hourly_tps.max() > 0:
            # If fewer than 100 hours had observations, fit a smooth parametric curve from observed peak/avg/floor
            nonzero = hourly_tps[hourly_tps > 0]
            if len(nonzero) < 100:
                pk = float(np.percentile(nonzero, 95))
                fl = max(1.0, float(np.percentile(nonzero, 10)))
                av = float(np.clip(nonzero.mean(), fl * 1.1, pk * 0.9)) if fl * 1.1 <= pk * 0.9 else (pk + fl) / 2
                fl, av = min(fl, pk), min(pk, max(min(fl, pk), av))   # few, even observations: keep floor <= avg <= peak
                hourly_series = [round(float(x), 1) for x in traffic.parametric(pk, av, fl)]
            else:
                hourly_series = [round(float(x), 1) for x in hourly_tps]

    result: dict = {"files": files_summary, "traffic": {}, "requests": {}, "prompts": {},
                    "timeseries_files": timeseries_files}

    if hourly_series:
        csv_path = run_dir / "uploaded_traffic.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["input_tokens_per_sec"])
            for v in hourly_series:
                w.writerow([round(float(v), 2)])
        arr = np.array(hourly_series, dtype=float)
        result["traffic"] = {
            "hourly_csv": str(csv_path),
            "peak_input_tps": round(float(arr.max()), 0),
            "average_input_tps": round(float(arr.mean()), 0),
            "floor_input_tps": round(float(max(1.0, arr.min())), 0),
            "profile": [round(float(x), 1) for x in arr],
        }

    # Deduplicate and analyze system prompts
    unique_sys = []
    if system_prompts:
        counts = Counter(system_prompts)
        total_sys_occurrences = sum(counts.values())
        unique_sys = [s for s, _ in counts.most_common(20)]
        sys_tok_lengths = [tc.count(s) for s in unique_sys]
        weighted_sys_tok = sum(tc.count(s) * c for s, c in counts.items()) / max(1, total_sys_occurrences)
        # Prefix cache hit rate estimation: requests sharing a repeated system prompt
        repeated = sum(c - 1 for c in counts.values() if c > 1)
        est_hit = round(min(0.98, max(0.5, repeated / max(1, total_sys_occurrences))), 2) if total_sys_occurrences > 1 else 0.9
        sys_file = run_dir / "uploaded_system_prompts.json"
        sys_file.write_text(json.dumps(unique_sys, indent=2))
        result["prompts"]["custom_system_prompts"] = unique_sys[:10]
        result["prompts"]["system_prompts_file"] = str(sys_file)
        result["prompts"]["num_system_prompts"] = len(unique_sys)
        result["prompts"]["avg_system_prompt_tokens"] = round(weighted_sys_tok)
        result["requests"]["system_prompt_tokens"] = round(weighted_sys_tok)
        result["requests"]["system_prompt_cache_hit"] = est_hit

    if prompt_files:
        # Same parser, storage and summary as POST /api/sessions/{name}/prompts (userprompts), so both paths agree.
        from aiops_dilemma import userprompts
        try:
            meta = userprompts.store_files(run_dir, prompt_files, hf_id=hf_id, tc=tc,
                                           method="tokenizer" if tc.tok else "estimate")
        except userprompts.PromptUploadError as e:
            raise ValueError(str(e))
        result["prompts"]["source"] = "file"
        result["prompts"]["file"] = userprompts.STORE      # relative to the session folder
        result["prompts"]["count"] = meta["count"]
        result["prompts_upload"] = meta

    if in_tok_samples:
        arr_in = np.array(in_tok_samples, dtype=float)
        arr_out = np.array(out_tok_samples, dtype=float) if out_tok_samples else np.full_like(arr_in, 400.0)
        n_total = len(arr_in)
        short_mask = arr_in <= 32000
        med_mask = (arr_in > 32000) & (arr_in <= 128000)
        long_mask = arr_in > 128000

        pct_short = round(100.0 * float(short_mask.sum()) / n_total, 2)
        pct_med = round(100.0 * float(med_mask.sum()) / n_total, 2)
        pct_long = round(100.0 * float(long_mask.sum()) / n_total, 2)
        pct_over_32k = round(pct_med + pct_long, 2)

        result["requests"].update({
            "avg_input_tokens": round(float(arr_in.mean())),
            "avg_output_tokens": round(float(arr_out.mean())),
            "p50_input_tokens": round(float(np.percentile(arr_in, 50))),
            "p95_input_tokens": round(float(np.percentile(arr_in, 95))),
            "max_input_tokens": round(float(arr_in.max())),
            "pct_short": pct_short,
            "pct_medium": pct_med,
            "pct_long": pct_long,
            "pct_over_32k": pct_over_32k,
            "pct_over_128k": pct_long,
            "short_mean": round(float(arr_in[short_mask].mean())) if short_mask.any() else 5000,
            "medium_mean": round(float(arr_in[med_mask].mean())) if med_mask.any() else 69000,
            "large_mean": round(float(arr_in[long_mask].mean())) if long_mask.any() else 188000,
            "sample_count": n_total,
        })

    return result


# ---------------------------------------------------------------------------------------------------------------------
# Time-series import (Cloud Monitoring / Metrics Explorer CSVs, `timestamp,value` exports, epoch series): parse every
# series with its timestamps, guess what it measures, resample to hourly means by time, build a typical week and
# combine requests/s with token series. Nothing is applied here: the server shows the proposal in a review sheet.
# ---------------------------------------------------------------------------------------------------------------------

TS_ROLES = ("requests", "input_tps", "output_tps", "total_tps", "ignore")
TS_ROLE_LABELS = {"requests": "requests/s", "input_tps": "input tokens/s", "output_tps": "output tokens/s",
                  "total_tps": "total tokens/s (input + output)", "ignore": "ignore", "unknown": "unknown"}
_MONTHS = {m: i + 1 for i, m in enumerate(("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct",
                                            "nov", "dec"))}
# JavaScript Date.toString(): "Sun Nov 08 2026 21:50:00 GMT+0200 (Example Standard Time)" (Metrics Explorer CSVs).
_JS_TS = re.compile(r"^(?:[A-Za-z]{3},?\s+)?([A-Za-z]{3})[a-z]*\.?\s+(\d{1,2}),?\s+(\d{4})\s+(\d{1,2}):(\d{2})(?::(\d{2}))?"
                    r"(?:\.\d+)?\s*(?:(?:GMT|UTC|Z)\s*([+-]\d{2}:?\d{2})?|([+-]\d{4}))?(?:\s*\(.*\))?\s*$")
_TIME_NAMES = {"timestamp", "time", "date", "datetime", "ts", "created_at", "request_time", "start_time", "end_time",
               "interval_end", "interval_start", "time_utc", "timestamp_utc", "hour", "minute", "period", "bucket"}
# Labels that identify an account or project are never shown, stored or echoed.
_PRIVATE_LABEL = re.compile(r"project|resource_container|container|account|billing|email|organization|folder|"
                            r"timeseries id|time_series_id", re.I)
_REQ_COLS = set(INPUT_TOK_COLS) | set(OUTPUT_TOK_COLS) | set(PROMPT_COLS) | set(SYS_COLS) | {"messages"}


def parse_time(val) -> tuple[datetime | None, bool]:
    """Parse one timestamp to an aware UTC datetime. Returns (dt, assumed_utc); assumed_utc = no zone was given.

    JavaScript Date strings (with GMT offset), ISO 8601 / RFC 3339 (Z or offset), `YYYY-MM-DD HH:MM[:SS]` and
    `YYYY/MM/DD ...` (UTC assumed), BigQuery's `... UTC`, RFC 2822, and epoch seconds / ms / µs / ns.
    """
    if val is None:
        return None, False
    if isinstance(val, datetime):
        return (val.astimezone(timezone.utc), False) if val.tzinfo else (val.replace(tzinfo=timezone.utc), True)
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return _epoch(float(val)), False
    s = str(val).strip().strip('"').strip()
    if not s:
        return None, False
    if re.fullmatch(r"\d+(\.\d+)?", s):
        return _epoch(float(s)), False
    m = _JS_TS.match(s)
    if m and m.group(1).lower()[:3] in _MONTHS:
        mon = _MONTHS[m.group(1).lower()[:3]]
        try:
            dt = datetime(int(m.group(3)), mon, int(m.group(2)), int(m.group(4)), int(m.group(5)), int(m.group(6) or 0))
        except ValueError:
            return None, False
        off = m.group(7) or m.group(8)
        if off:
            off = off.replace(":", "")
            sign = -1 if off[0] == "-" else 1
            delta = timedelta(hours=int(off[1:3]), minutes=int(off[3:5]))
            return (dt - sign * delta).replace(tzinfo=timezone.utc), False
        explicit = bool(re.search(r"\b(GMT|UTC)\b|Z$", s))
        return dt.replace(tzinfo=timezone.utc), not explicit
    iso = s
    explicit = False
    if re.search(r"\s*(UTC|GMT)$", iso, re.I):
        iso, explicit = re.sub(r"\s*(UTC|GMT)$", "", iso, flags=re.I), True
    if iso.endswith(("Z", "z")):
        iso, explicit = iso[:-1] + "+00:00", True
    if re.match(r"^\d{4}/\d{1,2}/\d{1,2}", iso):
        iso = iso.replace("/", "-", 2)
    iso = re.sub(r"^(\d{4})-(\d)(?=-)", r"\1-0\2", iso)
    iso = re.sub(r"^(\d{4}-\d{2})-(\d)(?=\D|$)", r"\1-0\2", iso)
    iso = re.sub(r"(\.\d{6})\d+", r"\1", iso)  # nanoseconds -> microseconds
    if re.match(r"^\d{4}-\d{2}-\d{2}", iso):
        try:
            dt = datetime.fromisoformat(iso)
        except ValueError:
            dt = None
        if dt is not None:
            if dt.tzinfo:
                return dt.astimezone(timezone.utc), False
            return dt.replace(tzinfo=timezone.utc), not explicit
    try:
        from email.utils import parsedate_to_datetime

        dt = parsedate_to_datetime(s)
        if dt is not None:
            return (dt.astimezone(timezone.utc), False) if dt.tzinfo else (dt.replace(tzinfo=timezone.utc), True)
    except (TypeError, ValueError, IndexError):
        pass
    return None, False


def _epoch(t: float) -> datetime | None:
    """Epoch seconds / ms / µs / ns between 1980 and 2100; anything else is not a timestamp."""
    for scale in (1.0, 1e3, 1e6, 1e9):
        x = t / scale
        if 3.15e8 <= x <= 4.1e9:
            return datetime.fromtimestamp(x, tz=timezone.utc)
        if x < 3.15e8:
            return None
    return None


def _num_cell(c) -> float | None:
    if c is None:
        return None
    s = str(c).strip().replace(",", "") if isinstance(c, str) else c
    try:
        f = float(s)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _is_ts_cell(c, allow_epoch: bool) -> bool:
    if c is None or str(c).strip() == "":
        return False
    if _num_cell(c) is not None:
        return allow_epoch and parse_time(c)[0] is not None
    return parse_time(c)[0] is not None


def _grid(fname: str, raw: bytes) -> list[list[str]]:
    """Rows of cells from CSV/TSV, or from JSON / JSONL records that carry a time field."""
    ext = Path(fname).suffix.lower()
    text = raw.decode("utf-8-sig", errors="replace").strip()
    if not text:
        return []
    if ext in (".json", ".jsonl") or text[:1] in "[{":
        try:
            recs = ([json.loads(ln) for ln in text.splitlines() if ln.strip()] if (ext == ".jsonl" or
                    (text.startswith("{") and "\n{" in text)) else json.loads(text))
        except ValueError:
            return []
        if isinstance(recs, dict):
            recs = next((recs[k] for k in ("points", "data", "rows", "series", "traffic") if isinstance(recs.get(k), list)), [])
        recs = [r for r in recs if isinstance(r, dict)] if isinstance(recs, list) else []
        if not recs:
            return []
        keys = list(dict.fromkeys(k for r in recs[:200] for k in r))
        return [keys] + [["" if r.get(k) is None else str(r.get(k)) for k in keys] for r in recs]
    first = text.split("\n", 1)[0]
    delim = "\t" if ext == ".tsv" or ("\t" in first and "," not in first) else ";" if first.count(";") > first.count(",") else ","
    return [row for row in csv.reader(io.StringIO(text), delimiter=delim)]


def _safe_labels(labels: dict) -> dict:
    return {k: v for k, v in labels.items() if k and v not in (None, "") and not _PRIVATE_LABEL.search(str(k))}


def _regular(times: list[datetime]) -> tuple[bool, float]:
    """(looks like a regular series, median step in seconds). Request logs have irregular, duplicate timestamps."""
    if len(times) < 2:
        return len(times) == 1, 3600.0
    diffs = np.diff(np.array([t.timestamp() for t in times]))
    pos = diffs[diffs > 0]
    if not len(pos) or len(pos) < 0.7 * len(diffs):
        return False, 0.0
    step = float(np.median(pos))
    mult = pos / step
    on_grid = np.abs(mult - np.round(mult)) <= 0.05
    return bool(on_grid.mean() >= 0.8), step


def parse_timeseries(fname: str, raw: bytes) -> list[dict]:
    """Every time series in an export, or [] when the file isn't one (no timestamps, or per-request logs).

    Handles Cloud Monitoring Metrics Explorer CSVs (`label,value` rows such as `TimeSeries ID,...`,
    `model_user_id,...` above `timestamp,value` rows; several series = several value columns or column blocks),
    header-row CSVs (`timestamp,value`, `Time,<metric name>`), epoch timestamps, and JSON / JSONL records.
    Each series: {id, file, name, labels (safe: no project/account labels), points [(utc datetime, value)], step_s,
    assumed_utc}.
    """
    rows = [[c.strip() for c in r] for r in _grid(fname, raw) if any(str(c).strip() for c in r)]
    if len(rows) < 2:
        return []
    # First data row: a timestamp in the first non-empty cell and a number somewhere after it.
    start = None
    for i, r in enumerate(rows[:200]):
        j0 = next((j for j, c in enumerate(r) if c), None)
        if j0 is None:
            continue
        hdr_time = i > 0 and str(rows[i - 1][j0] if j0 < len(rows[i - 1]) else "").lower() in _TIME_NAMES
        if _is_ts_cell(r[j0], allow_epoch=hdr_time or i == 0 or j0 == 0) and any(
                _num_cell(c) is not None for c in r[j0 + 1:]):
            if _num_cell(r[j0]) is not None and not hdr_time:
                # A bare number in column 0 is only a timestamp when the whole column is epoch-like and increasing.
                col = [_num_cell(x[j0]) for x in rows[i:i + 50] if j0 < len(x)]
                if not all(c is not None and _epoch(c) for c in col) or col != sorted(col):
                    continue
            start = i
            break
    if start is None:
        return []
    header, data = rows[:start], rows[start:]
    ncols = max(len(r) for r in data)
    hdr_cell = lambda h, j: h[j] if j < len(h) else ""  # noqa: E731
    ts_cols = []
    for j in range(ncols):
        cells = [r[j] for r in data if j < len(r) and r[j]]
        if not cells:
            continue
        named_time = any(hdr_cell(h, j).lower() in _TIME_NAMES for h in header)
        allow_epoch = named_time or j == 0
        hits = sum(1 for c in cells[:300] if _is_ts_cell(c, allow_epoch))
        if hits >= 0.8 * min(len(cells), 300) and (named_time or _num_cell(cells[0]) is None or j == 0):
            ts_cols.append(j)
    if not ts_cols:
        return []
    series: list[dict] = []
    stem = Path(fname).stem
    for j in range(ncols):
        if j in ts_cols or j < ts_cols[0]:
            continue
        t = max(c for c in ts_cols if c < j)
        cells = [r[j] for r in data if j < len(r) and r[j]]
        if not cells or sum(_num_cell(c) is not None for c in cells) < 0.5 * len(cells):
            continue
        labels: dict = {}
        name = ""
        for h in header:
            key = (hdr_cell(h, t) or hdr_cell(h, 0)).strip()
            val = hdr_cell(h, j).strip()
            if not key and not val:
                continue
            if key.lower() in _TIME_NAMES or (not key and val):
                name = val or name
            elif val:
                labels[key] = val
        if name.strip().lower() in _REQ_COLS:
            return []  # per-request logs (timestamp, input_tokens, ...): handled by ingest_files
        pts: dict[datetime, float] = {}
        assumed = False
        for r in data:
            if j >= len(r) or t >= len(r):
                continue
            v = _num_cell(r[j])
            dt, a = parse_time(r[t]) if _num_cell(r[t]) is None else (_epoch(_num_cell(r[t])), False)
            if v is None or dt is None:
                continue
            assumed = assumed or a
            pts[dt] = pts.get(dt, 0.0) + v if dt in pts else v
        if not pts:
            continue
        times = sorted(pts)
        regular, step = _regular(times)
        if not regular and len(times) > 3:
            return []  # irregular timestamps: request logs, not a series
        metric_name = name or labels.get("metric") or labels.get("metric_type") or labels.get("__name__") or ""
        series.append({
            "id": f"s{len(series) + 1}", "file": fname, "name": metric_name or stem, "column": name,
            "labels": _safe_labels(labels), "points": [(dt, pts[dt]) for dt in times],
            "step_s": int(round(step)) if len(times) > 1 else 3600, "assumed_utc": assumed,
        })
    return series


def _norm_text(s) -> str:
    return " " + re.sub(r"[_\-\.:/]+", " ", str(s or "")).lower().replace("(", " ").replace(")", " ") + " "


def _guess_text(text: str) -> str | None:
    t = text
    tok = re.search(r"\btokens?\b|\btps\b|token count|tokens per s|\bthroughput\b", t) and not re.search(r"\bqps\b|\brps\b", t)
    inp = re.search(r"\b(input|prompt|in|cache read input|cache write input)\b", t)
    out = re.search(r"\b(output|completion|generated|generation|out)\b", t)
    if tok:
        if inp and not out:
            return "input_tps"
        if out and not inp:
            return "output_tps"
        if re.search(r"\b(input tps|tokens per sec|input tokens per sec|hourly input tps)\b", t):
            return "input_tps"
        if re.search(r"^\s*tps\s*$", t):
            return "input_tps"
        return "total_tps"
    if re.search(r"\b(qps|rps|req s|requests?|invocations?|invocation count|predictions?|prediction count|calls|"
                 r"queries|response count|request count)\b", t):
        return "requests"
    return None


def classify_series(s: dict) -> tuple[str, str, str]:
    """(metric guess, confidence high/medium/low, reason) from the label `type`, column / series name, other labels
    and the file name, in that order."""
    labels = s.get("labels") or {}
    typ = str(labels.get("type") or labels.get("token_type") or "").lower()
    if typ:
        if typ == "output" or "output" in typ or "completion" in typ:
            return "output_tps", "high", f"label type={typ}"
        if "input" in typ or "prompt" in typ:
            return "input_tps", "high", f"label type={typ}"
    sources = [("column", s.get("column") or ""), ("series name", s.get("name") if s.get("name") != Path(s.get("file") or "").stem else ""),
               ("labels", " ".join(f"{k} {v}" for k, v in labels.items() if k in ("metric", "metric_type", "__name__"))),
               ("file name", Path(s.get("file") or "").stem)]
    for where, text in sources:
        if not str(text).strip() or str(text).strip().lower() in ("value", "values", "y", "count"):
            continue
        g = _guess_text(_norm_text(text))
        if g:
            conf = "high" if where != "file name" else "medium"
            if g == "total_tps":
                conf = "medium"
                why = (f"{where} '{str(text)[:60]}' is a token count without an input/output split, so it's treated "
                       "as input + output")
            else:
                why = f"{where} '{str(text)[:60]}' looks like {TS_ROLE_LABELS[g]}"
            return g, conf, why
    return "unknown", "low", "no column, label or file name says what this series measures"


def _per_interval(s: dict) -> bool:
    """True when values are counts per interval (divide by the step), not per-second rates."""
    t = _norm_text(" ".join([s.get("column") or "", s.get("name") or "", Path(s.get("file") or "").stem]))
    if re.search(r"per second|per sec|\bper s\b|\bqps\b|\brps\b|\btps\b|\brate\b|\bs\b\s*$|req s|tok s", t):
        return False
    return bool(re.search(r"\b(count|total|sum|number)\b", t))


def hourly_means(points: list, step_s: float) -> dict[datetime, tuple[float, float]]:
    """Time-weighted hourly means: each point covers the `step_s` before its timestamp (Monitoring's aligned points
    are labelled with the end of their interval). Returns {hour start (UTC): (mean, covered seconds)}; hours no point
    covers are absent (missing, not zero)."""
    step = max(1.0, float(step_s or 3600))
    acc: dict[datetime, list[float]] = {}
    for dt, v in points:
        end = dt.timestamp()
        a = end - step
        while a < end - 1e-6:
            h0 = math.floor(a / 3600) * 3600
            b = min(end, h0 + 3600)
            w = b - a
            key = datetime.fromtimestamp(h0, tz=timezone.utc)
            slot = acc.setdefault(key, [0.0, 0.0])
            slot[0] += float(v) * w
            slot[1] += w
            a = b
    return {k: (s / w, w) for k, (s, w) in sorted(acc.items()) if w > 0}


def _circ_fill(arr: np.ndarray) -> np.ndarray:
    """Fill NaNs by circular linear interpolation (a day or a week wraps around)."""
    a = np.array(arr, dtype=float)
    n = len(a)
    ok = np.where(~np.isnan(a))[0]
    if not len(ok):
        return a
    if len(ok) == n:
        return a
    xs = np.concatenate([ok - n, ok, ok + n])
    ys = np.concatenate([a[ok]] * 3)
    miss = np.where(np.isnan(a))[0]
    a[miss] = np.interp(miss, xs, ys)
    return a


def _fmt_utc(dt: datetime) -> str:
    return dt.strftime("%a %H:%M")


def _gaps(hours: list[datetime]) -> list[dict]:
    out = []
    for a, b in zip(hours, hours[1:]):
        missing = int(round((b - a).total_seconds() / 3600)) - 1
        if missing >= 1:
            g0 = a + timedelta(hours=1)
            out.append({"start": g0.isoformat(), "end": b.isoformat(), "hours": missing})
    return out


def _context_defaults(ctx: dict | None) -> dict:
    ctx = dict(ctx or {})
    req = ctx.get("requests") or {}
    avg_in = _num(req.get("avg_input_tokens")) or 6000.0
    from aiops_dilemma.config import effective_output_tokens
    eff_out = effective_output_tokens(req) if req else 300.0
    wf = _num(ctx.get("weekend_factor"))
    cur = ctx.get("current_profile")
    cur = [max(0.0, float(x)) for x in cur] if isinstance(cur, list) and len(cur) == 168 and all(
        _num(x) is not None for x in cur) else None
    return {"avg_in": max(1.0, avg_in), "eff_out": max(0.0, eff_out or 0.0),
            "weekend_factor": 0.15 if wf is None else min(1.0, max(0.0, wf)), "current_profile": cur}


def _wmean(vals: list[tuple[float, float]]) -> float:
    """Coverage-weighted mean of (value, seconds covered); NaN when empty."""
    w = sum(x[1] for x in vals)
    return sum(v * x for v, x in vals) / w if vals and w > 0 else float("nan")


def _ratio_sum(num: dict, den: dict) -> float | None:
    common = [h for h in num if h in den and den[h] > 0]
    d = sum(den[h] for h in common)
    return (sum(num[h] for h in common) / d) if common and d > 0 else None


def build_traffic_proposal(series: list[dict], mapping: dict | None = None, context: dict | None = None,
                           option: str | None = None, missing_is_zero: bool = False,
                           window: tuple[datetime, datetime] | None = None, gaps_as_zero: bool = False) -> dict:
    """Combine mapped series into a proposed week. Returns the import response (without applying anything):
    {series, proposal, coverage, options, warnings, option}. `mapping` = {series id: role}; missing ids use the guess.
    `missing_is_zero` (Cloud Monitoring: counters report nothing in hours without calls) fills the whole `window`;
    `gaps_as_zero` fills only the hours missing between the first and last data point (the user's choice)."""
    ctx = _context_defaults(context)
    mapping = mapping or {}
    warnings: list[str] = []
    rows, by_role = [], {r: {} for r in TS_ROLES if r != "ignore"}
    cached_shares: list[float] = []         # Cloud Monitoring: share of input tokens that were cache reads
    hour_w: dict[datetime, float] = {}      # seconds of each hour the data covers (weights for the averages)
    for s in series:
        guess, conf, why = s.get("metric_guess"), s.get("confidence"), s.get("reason")
        if not guess:
            guess, conf, why = classify_series(s)
        role = mapping.get(s["id"]) or (guess if guess in TS_ROLES else "ignore")
        if role not in TS_ROLES:
            role = "ignore"
        if role == "input_tps" and s.get("cached_share"):
            cached_shares.append(float(s["cached_share"]))
        pts = s["points"]
        hm = hourly_means(pts, s.get("step_s") or 3600)
        div = (s.get("step_s") or 3600) if s.get("per_interval", _per_interval(s)) else 1.0
        if div != 1.0 and role != "ignore":
            warnings.append(f"{s['name'][:60]}: values look like counts per {int(div)} s interval; divided by the "
                            "interval to get a per-second rate.")
        rows.append({"id": s["id"], "file": s.get("file"), "name": s.get("name"), "metric_guess": guess,
                     "confidence": conf, "reason": why, "role": role, "labels": _safe_labels(s.get("labels") or {}),
                     "points_n": len(pts), "first": pts[0][0].isoformat() if pts else None,
                     "last": pts[-1][0].isoformat() if pts else None, "step_s": s.get("step_s"),
                     "coverage_h": round(sum(w for _, w in hm.values()) / 3600, 1)})
        if role == "ignore":
            continue
        tgt = by_role[role]
        # An hour touched by a single short point (e.g. the edge of the export) is too thin to stand for the hour.
        min_w = 600 if (s.get("step_s") or 3600) < 600 else 0
        for h, (m, w) in hm.items():
            if w < min_w:
                continue
            tgt[h] = tgt.get(h, 0.0) + m / div
            hour_w[h] = max(hour_w.get(h, 0.0), w)
        if s.get("assumed_utc"):
            warnings.append(f"{s.get('file')}: timestamps have no time zone; read as UTC.")
        if s.get("cached_share"):
            warnings.append(f"{s['cached_share']:.1%} of the input tokens were cache reads or writes (prompt caching). "
                            "They are counted as input: a self-hosted model still has to serve that context, though "
                            "prefix caching makes it cheaper.")
    if missing_is_zero and window:
        h = window[0].replace(minute=0, second=0, microsecond=0)
        while h < window[1]:
            for r, d in by_role.items():
                if d or any(x["role"] == r for x in rows):
                    d.setdefault(h, 0.0)
            hour_w[h] = 3600.0
            h += timedelta(hours=1)
    if gaps_as_zero and not missing_is_zero:
        for d in by_role.values():
            if len(d) > 1:
                h, last = min(d), max(d)
                while h < last:
                    d.setdefault(h, 0.0)
                    hour_w.setdefault(h, 3600.0)
                    h += timedelta(hours=1)
    req, inp, out, tot = by_role["requests"], by_role["input_tps"], by_role["output_tps"], by_role["total_tps"]
    r_cfg = ctx["avg_in"] / (ctx["avg_in"] + ctx["eff_out"]) if ctx["avg_in"] + ctx["eff_out"] > 0 else 1.0
    basis, derived_input = "", False
    if inp:
        basis = "input tokens/s series"
    elif tot and out:
        inp = {h: max(0.0, v - out[h]) if h in out else v * r_cfg for h, v in tot.items()}
        basis = "total − output tokens/s"
    elif tot:
        inp = {h: v * r_cfg for h, v in tot.items()}
        basis = f"total tokens/s × {r_cfg:.1%} (your {ctx['avg_in']:,.0f} in / {ctx['eff_out']:,.0f} out per request)"
        derived_input = True
        warnings.append(f"The token series counts input + output tokens. Input ≈ total × {r_cfg:.1%}, from your "
                        f"configured {ctx['avg_in']:,.0f} input / {ctx['eff_out']:,.0f} output tokens per request "
                        "(derived).")
    elif req:
        inp = {h: v * ctx["avg_in"] for h, v in req.items()}
        basis = f"requests/s × your {ctx['avg_in']:,.0f} input tokens per request"
        derived_input = True
        warnings.append(f"Only requests/s: input tok/s = requests/s × your configured {ctx['avg_in']:,.0f} input "
                        "tokens per request (derived). Add a token series to measure it.")
    elif out:
        inp = {h: v * ctx["avg_in"] / max(1.0, ctx["eff_out"]) for h, v in out.items()}
        basis = "output tokens/s × your input/output ratio"
        derived_input = True
        warnings.append("Only output tokens: input tok/s derived from your configured input/output ratio.")
    result = {"series": rows, "warnings": warnings, "options": [], "proposal": None, "coverage": None,
              "derived": [], "basis": basis}
    if not inp or max(inp.values(), default=0) <= 0:
        result["error"] = ("No usable traffic: map at least one series to requests/s or tokens/s (with non-zero values)."
                           if inp is not None else "No series mapped.")
        return result

    # Derived per-request figures (time-aligned: only hours where both series have data).
    derived = []
    avg_in_new = None
    if req and (by_role["input_tps"] or tot):
        if by_role["input_tps"]:
            v = _ratio_sum(by_role["input_tps"], req)
            why, dep = "input tokens/s ÷ requests/s over the hours both cover", False
        elif out:
            v = _ratio_sum({h: max(0.0, t - out.get(h, 0.0)) for h, t in tot.items()}, req)
            why, dep = "(total − output tokens/s) ÷ requests/s over the hours both cover", False
        else:
            tpr = _ratio_sum(tot, req)
            v = tpr * r_cfg if tpr else None
            why = (f"{tpr:,.0f} total tokens per request (token count ÷ requests/s) × {r_cfg:.1%} input share from "
                   "your configured output tokens" if tpr else "")
            dep = True
        if v and v > 0:
            avg_in_new = v
            derived.append({"path": "requests.avg_input_tokens", "label": "Input tokens per request",
                            "value": int(round(v)), "now": int(round(ctx["avg_in"])), "reason": why,
                            "derived_from_config": dep, "checked": True})
    shares = cached_shares
    if shares and (avg_in_new or ctx["avg_in"]):
        a_in = avg_in_new or ctx["avg_in"]
        derived.append({"path": "requests.cached_input_tokens", "label": "Cached input tokens per request",
                        "value": int(round(a_in * shares[0])), "now": None,
                        "reason": f"{shares[0]:.1%} of input tokens were cache reads (resent conversation or shared "
                                  "context): priced at the cached rate on MaaS, and mostly skipped by prefix caching on GPUs",
                        "derived_from_config": avg_in_new is None, "checked": True})
    if req and out:
        v = _ratio_sum(out, req)
        if v and v > 0:
            derived.append({"path": "requests.avg_output_tokens", "label": "Output tokens per request",
                            "value": int(round(v)), "now": int(round(ctx["eff_out"])),
                            "reason": "output tokens/s ÷ requests/s (includes thinking tokens)",
                            "derived_from_config": False, "checked": True})
    elif out and (by_role["input_tps"] or tot):
        base_in = by_role["input_tps"] or {h: max(0.0, t - out.get(h, 0.0)) for h, t in tot.items()}
        ratio = _ratio_sum(out, base_in)
        if ratio:
            a_in = avg_in_new or ctx["avg_in"]
            derived.append({"path": "requests.avg_output_tokens", "label": "Output tokens per request",
                            "value": int(round(a_in * ratio)), "now": int(round(ctx["eff_out"])),
                            "reason": f"output ÷ input tokens = {ratio:.3f} × {a_in:,.0f} input tokens per request",
                            "derived_from_config": avg_in_new is None, "checked": True, "output_ratio": ratio})
    output_ratio = None
    if out and (by_role["input_tps"] or tot):
        base_in = by_role["input_tps"] or {h: max(0.0, t - out.get(h, 0.0)) for h, t in tot.items()}
        output_ratio = _ratio_sum(out, base_in)

    # Coverage
    hours = sorted(h for h, v in inp.items() if v is not None)
    first_pt = min((p[0][0] for p in (s["points"] for s, r in zip(series, rows) if r["role"] != "ignore") if p),
                   default=hours[0])
    last_pt = max((p[-1][0] for p in (s["points"] for s, r in zip(series, rows) if r["role"] != "ignore") if p),
                  default=hours[-1])
    if missing_is_zero and window:
        first_pt, last_pt = window[0], window[1]
    covered = len(hours)
    step_used = min((s.get("step_s") or 3600 for s, r in zip(series, rows) if r["role"] != "ignore"), default=3600)
    span_h = max(1, int(round(((last_pt - first_pt).total_seconds() + (0 if missing_is_zero else step_used)) / 3600)))
    gaps = _gaps(hours)
    gap_h = sum(g["hours"] for g in gaps)
    span_txt = (f"{span_h} h of data ({_fmt_utc(first_pt)} – {_fmt_utc(last_pt)} UTC)" if span_h < 72 else
                f"{span_h / 24:.1f} days of data ({first_pt:%a %d %b %H:%M} – {last_pt:%a %d %b %H:%M} UTC)")
    cov = {"hours": covered, "days": round(covered / 24, 1), "span_hours": span_h, "first": first_pt.isoformat(),
           "last": last_pt.isoformat(), "gaps": gaps[:50], "gap_hours": gap_h, "gaps_as_zero": bool(gaps_as_zero),
           "label": span_txt + (f"; {gap_h} h inside it have no points" if gap_h else "")}
    raw_gaps = _gaps(sorted(set().union(*[set(hourly_means(s["points"], s.get("step_s") or 3600))
                                          for s, r in zip(series, rows) if r["role"] != "ignore"])))
    if raw_gaps and not missing_is_zero:
        longest = max(raw_gaps, key=lambda g: g["hours"])
        g0 = datetime.fromisoformat(longest["start"])
        cov["raw_gap_hours"] = sum(g["hours"] for g in raw_gaps)
        if not gaps_as_zero:
            warnings.append(f"{cov['raw_gap_hours']} h inside the data have no points (longest {longest['hours']} h from "
                            f"{_fmt_utc(g0)} UTC); they are treated as missing and interpolated. Monitoring exports "
                            "skip intervals without requests, so if those were quiet hours, treat gaps as zero traffic.")

    wf = ctx["weekend_factor"]
    slot = {h: h.weekday() * 24 + h.hour for h in hours}
    options = []

    def opt(key, label, desc, week, extra=None):
        w = np.maximum(0.0, np.array(week, dtype=float))
        d = {"key": key, "label": label, "description": desc, "custom_hourly": [round(float(x), 2) for x in w],
             "peak_input_tps": round(float(w.max()), 1), "average_input_tps": round(float(w.mean()), 1),
             "floor_input_tps": round(float(w.min()), 1),
             "monthly_input_tokens": round(float(w.mean()) * 3600 * traffic.HOURS_PER_MONTH)}
        d.update(extra or {})
        options.append(d)

    if covered >= 168 and span_h >= 168:
        acc = [[] for _ in range(168)]
        for h in hours:
            acc[slot[h]].append((inp[h], hour_w.get(h, 3600.0)))
        weeks = span_h / 168
        wk = _circ_fill(np.array([_wmean(a) for a in acc]))
        n_missing = sum(1 for a in acc if not a)
        opt("typical_week", "Typical week",
            f"Mean of each weekday-hour over {weeks:.1f} weeks of data"
            + (f"; {n_missing} weekday-hours never seen, interpolated" if n_missing else "") + ".", wk)
    else:
        day_all = [[] for _ in range(24)]
        wkd, wke = [[] for _ in range(24)], [[] for _ in range(24)]
        for h in hours:
            vw = (inp[h], hour_w.get(h, 3600.0))
            day_all[h.hour].append(vw)
            (wke if h.weekday() >= 5 else wkd)[h.hour].append(vw)
        mean24 = lambda a: np.array([_wmean(x) for x in a])  # noqa: E731
        if span_h >= 24:
            d_wkd, d_wke = mean24(wkd), mean24(wke)
            have_wkd, have_wke = int((~np.isnan(d_wkd)).sum()), int((~np.isnan(d_wke)).sum())
            notes = []
            if have_wkd >= 6:
                d_wkd = _circ_fill(d_wkd)
            if have_wke >= 6:
                d_wke = _circ_fill(d_wke)
            if have_wkd < 6 and have_wke >= 6:
                d_wkd = d_wke / wf if wf > 0 else d_wke
                notes.append(f"no weekday data: weekdays = weekend ÷ your weekend factor {wf:.0%}")
            if have_wke < 6:
                d_wke = d_wkd * wf
                notes.append(f"no weekend data: weekend = weekdays × your weekend factor {wf:.0%}")
            opt("weekday_weekend", "Weekday and weekend curves",
                f"Hourly curves from {span_h / 24:.1f} days of data, repeated over the week"
                + (f" ({'; '.join(notes)})" if notes else "") + ".", np.concatenate([d_wkd] * 5 + [d_wke] * 2))
        else:
            d = mean24(day_all)
            n_have = int((~np.isnan(d)).sum())
            d = _circ_fill(d)
            opt("daily_shape", "Use it as the daily shape",
                f"Every weekday follows these {n_have} hours; the other {24 - n_have} hours of the day are "
                f"interpolated, and the weekend = weekdays × your weekend factor ({wf:.0%}).",
                np.concatenate([d] * 5 + [d * wf] * 2))
            warnings.insert(0, f"Only {span_h} h of data ({_fmt_utc(first_pt)} – {_fmt_utc(last_pt)} UTC)"
                               + (f", {covered} h of it with points" if covered < span_h - 1 else "")
                               + ": less than a day, so any week is an extrapolation. Choose how to use it.")
    cur = ctx["current_profile"]
    if cur is not None and max(cur) > 0:
        cur_arr = np.array(cur, dtype=float)
        obs = sum(inp[h] for h in hours)
        ref = sum(cur_arr[slot[h]] for h in hours)
        if ref > 0 and obs > 0:
            k = obs / ref
            opt("level_only", "Only use the level",
                f"Keep your current week's shape and scale it ×{k:.2f} so it matches the measured traffic over the "
                f"same {covered} hours of the week.", cur_arr * k, {"scale": round(k, 4)})
    keys = [o["key"] for o in options]
    choice = option if option in keys else keys[0]
    chosen = options[keys.index(choice)]
    proposal = {k: chosen[k] for k in ("custom_hourly", "peak_input_tps", "average_input_tps", "floor_input_tps",
                                       "monthly_input_tokens")}
    proposal["avg_input_tokens"] = next((d["value"] for d in derived if d["path"] == "requests.avg_input_tokens"), None)
    proposal["avg_output_tokens"] = next((d["value"] for d in derived if d["path"] == "requests.avg_output_tokens"), None)
    proposal["output_ratio"] = round(output_ratio, 4) if output_ratio else None
    proposal["input_derived"] = derived_input
    proposal["basis"] = basis
    result.update({"proposal": proposal, "coverage": cov, "options": options, "option": choice, "derived": derived})
    return result


def series_to_json(series: list[dict]) -> list[dict]:
    out = []
    for s in series:
        d = {k: v for k, v in s.items() if k != "points"}
        d["labels"] = _safe_labels(d.get("labels") or {})
        d["points"] = [[dt.isoformat(), v] for dt, v in s["points"]]
        out.append(d)
    return out


def series_from_json(items: list[dict]) -> list[dict]:
    out = []
    for d in items or []:
        s = dict(d)
        s["points"] = [(datetime.fromisoformat(t), float(v)) for t, v in d.get("points") or []]
        out.append(s)
    return out


def import_timeseries_files(files: list[tuple[str, bytes]]) -> tuple[list[dict], list[str]]:
    """Parse uploaded files; returns (series with guesses, names of files that are not time series)."""
    series, other = [], []
    for fname, raw in files:
        try:
            found = parse_timeseries(fname, raw)
        except Exception as e:  # a malformed file shouldn't break the others
            log.info("parse_timeseries(%s) failed: %s", fname, e)
            found = []
        if not found:
            other.append(fname)
            continue
        for s in found:
            s["id"] = f"s{len(series) + 1}"
            s["metric_guess"], s["confidence"], s["reason"] = classify_series(s)
            s["per_interval"] = _per_interval(s)
            series.append(s)
    return series, other


# ---------------------------------------------------------------------------------------------------------------------
# Workload from a document: Gemini on Agent Platform reads a free-form description (PDF / DOCX / text) and returns
# the workload fields it supports, each with a confidence label and evidence. The document is processed in memory
# only: nothing is written to disk.
# ---------------------------------------------------------------------------------------------------------------------

log = logging.getLogger(__name__)

# Newest first; the first model that answers wins (override with $DILEMMA_DESCRIBE_MODEL).
DESCRIBE_MODELS = (
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
)
DESCRIBE_MAX_BYTES = 20 * 1024 * 1024
DESCRIBE_MAX_TEXT_CHARS = 2_000_000
DESCRIBE_EXTS = (".pdf", ".txt", ".md", ".docx")
TASK_TYPES = ("extraction", "summarization", "reasoning", "coding", "chat")
CONFIDENCE = ("stated", "derived", "assumed")

# path -> (kind, min, max). kind: num | share | pct | hours | hourly | task | text
DESCRIBE_FIELDS: dict[str, tuple[str, float | None, float | None]] = {
    "traffic.peak_input_tps": ("num", 0.0, None),
    "traffic.average_input_tps": ("num", 0.0, None),
    "traffic.floor_input_tps": ("num", 0.0, None),
    "traffic.busy_hours_utc": ("hours", 0, 24),
    "traffic.weekend_factor": ("share", 0.0, 1.0),
    "traffic.monthly_input_tokens": ("num", 0.0, None),
    "traffic.custom_hourly": ("hourly", 0.0, None),
    "requests.avg_input_tokens": ("num", 0.0, None),
    "requests.avg_output_tokens": ("num", 0.0, None),
    "requests.pct_over_32k": ("pct", 0.0, 100.0),
    "requests.pct_over_128k": ("pct", 0.0, 100.0),
    "requests.system_prompt_tokens": ("num", 0.0, None),
    "requests.thinking_share": ("share", 0.0, 1.0),
    "requests.avg_thinking_tokens": ("num", 0.0, None),
    "maas.max_rps": ("num", 0.0, None),
    "maas.input_per_m": ("num", 0.0, None),
    "maas.output_per_m": ("num", 0.0, None),
    "maas.cached_per_m": ("num", 0.0, None),
    "constraints.latency_budget_s": ("num", 0.0, None),
    "use_case.task_type": ("task", None, None),
    "use_case.description": ("text", None, None),
}
# Zero is a meaningful value for these (no quiet-hour traffic, no long prompts, no thinking, ...); others must be > 0.
_ZERO_OK = {"traffic.floor_input_tps", "traffic.weekend_factor", "requests.pct_over_32k", "requests.pct_over_128k",
            "requests.system_prompt_tokens", "requests.thinking_share", "requests.avg_thinking_tokens",
            "maas.cached_per_m", "maas.max_rps"}
_INT_FIELDS = {"requests.avg_input_tokens", "requests.avg_output_tokens", "requests.system_prompt_tokens",
               "requests.avg_thinking_tokens", "traffic.monthly_input_tokens"}

DESCRIBE_PROMPT = """You turn a free-form description of an LLM workload into the inputs of a cost model that compares
self-hosting an open-weight LLM on GPUs/TPUs with per-token APIs. Read the attached document and return one entry in
`fields` for every value below that the document supports. Omit any field the document gives no basis for: never
invent values to fill the list.

Fields (units matter):
- traffic.peak_input_tps: busiest hour of a typical week, averaged over that hour, in INPUT tokens per second.
- traffic.average_input_tps: weekly average input tokens per second.
- traffic.floor_input_tps: quietest hour of the week, input tokens per second (0 allowed).
- traffic.busy_hours_utc: [start_hour, end_hour] of the weekday busy plateau in UTC, 0-24, start < end (use `numbers`).
- traffic.weekend_factor: weekend traffic above the floor as a fraction of weekday traffic above the floor, 0-1.
- traffic.monthly_input_tokens: total input tokens per month (730 hours).
- traffic.custom_hourly: ONLY if the document describes an hour-by-hour (or finer) weekly or daily pattern: exactly
  168 hourly input tok/s values, Monday 00:00 UTC first (use `numbers`). A daily pattern repeats for each day,
  adjusted for any weekday/weekend difference the document states. Average 10-minute data into hours.
- requests.avg_input_tokens / requests.avg_output_tokens: mean tokens per request (output = visible answer tokens).
- requests.pct_over_32k / requests.pct_over_128k: PERCENT (0-100) of requests whose input exceeds 32K / 128K tokens.
- requests.system_prompt_tokens: tokens of the shared system prompt / instructions prefix.
- requests.thinking_share: FRACTION (0-1) of requests with reasoning/thinking enabled.
- requests.avg_thinking_tokens: extra output tokens per request when thinking is on.
- maas.max_rps: rate limit (requests per second) of the per-token open-model API (MaaS) they use or plan to use.
- maas.input_per_m / maas.output_per_m / maas.cached_per_m: that API's USD price per million input / output /
  cached input tokens.
- constraints.latency_budget_s: maximum acceptable end-to-end latency per request, seconds.
- use_case.task_type: one of extraction, summarization, reasoning, coding, chat (use `text`).
- use_case.description: one or two neutral sentences describing what the workload does (use `text`).

Conversions (show the arithmetic briefly in `evidence` and mark the field `derived`):
- requests/s x average input tokens per request = input tok/s. Requests/min or /hour: divide accordingly.
- Monthly or daily volumes: tokens per month / (730 x 3600) = average tok/s; requests per day x tokens per request
  / 86400 = average tok/s. Also fill monthly_input_tokens when a monthly volume is given or derivable.
- If only output or total tokens are given, split using the stated input/output ratio; otherwise say so in notes.
- Percentiles or charts: use the busiest-hour level (or p95-p99 of hourly values) for peak, the mean for average and
  the quietest hour (or p1-p5) for floor. Per-minute or per-second peaks are burstier than hourly averages; prefer
  hourly figures and mention the difference in notes.
- Weekday/weekend patterns: busy_hours_utc from the stated business hours (convert the stated time zone to UTC) and
  weekend_factor from the weekend-to-weekday ratio.
- Token counts from characters or words: about 4 characters or 0.75 words per token; mark `derived`.

Confidence: `stated` = the number appears in the document in these units; `derived` = computed from stated numbers
(give the arithmetic); `assumed` = a reasonable default you inferred without a direct number (give the reasoning).
Keep `evidence` short (a brief quote or one line of reasoning). Put conflicts between figures, ambiguities, caveats,
and anything relevant you could not map to a field in `notes` (short sentences). Use `number` for numeric fields,
`numbers` for lists, `text` for strings."""


def _describe_schema() -> dict:
    return {
        "type": "OBJECT",
        "properties": {
            "fields": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "path": {"type": "STRING", "enum": list(DESCRIBE_FIELDS)},
                        "number": {"type": "NUMBER", "nullable": True},
                        "numbers": {"type": "ARRAY", "items": {"type": "NUMBER"}, "nullable": True},
                        "text": {"type": "STRING", "nullable": True},
                        "confidence": {"type": "STRING", "enum": list(CONFIDENCE)},
                        "evidence": {"type": "STRING"},
                    },
                    "required": ["path", "confidence", "evidence"],
                    "propertyOrdering": ["path", "number", "numbers", "text", "confidence", "evidence"],
                },
            },
            "notes": {"type": "ARRAY", "items": {"type": "STRING"}},
        },
        "required": ["fields", "notes"],
        "propertyOrdering": ["fields", "notes"],
    }


class DescribeError(Exception):
    """A user-facing failure of describe_workload; `status` is the HTTP status the server should return."""

    def __init__(self, message: str, status: int = 400, code: str = "bad_request"):
        super().__init__(message)
        self.status = status
        self.code = code


DOCX_XML_MAX_BYTES = 20_000_000   # uncompressed word/document.xml


def _docx_document_xml(data: bytes) -> bytes:
    """word/document.xml of a .docx, refusing one that would decompress beyond DOCX_XML_MAX_BYTES (a "zip bomb" would
    otherwise exhaust memory): the declared size is checked first and the read is bounded in case it lies."""
    import zipfile
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        try:
            info = z.getinfo("word/document.xml")
        except KeyError:
            raise DescribeError("This .docx has no document body (word/document.xml).", 400, "bad_docx")
        if info.file_size > DOCX_XML_MAX_BYTES:
            raise DescribeError("This .docx is too large to read (its text part is over "
                                f"{DOCX_XML_MAX_BYTES // 1_000_000} MB uncompressed).", 413, "docx_too_large")
        with z.open(info) as fh:
            xml_bytes = fh.read(DOCX_XML_MAX_BYTES + 1)
    if len(xml_bytes) > DOCX_XML_MAX_BYTES:
        raise DescribeError("This .docx is too large to read.", 413, "docx_too_large")
    return xml_bytes


def _docx_text(data: bytes) -> str:
    """Paragraph and table text of a .docx, via python-docx when installed, else straight from the XML. The size
    check runs first either way. ElementTree resolves no external entities and expat limits entity expansion."""
    xml_bytes = _docx_document_xml(data)
    try:
        import docx  # type: ignore

        d = docx.Document(io.BytesIO(data))
        parts = [p.text for p in d.paragraphs]
        for table in d.tables:
            for row in table.rows:
                parts.append("\t".join(c.text for c in row.cells))
        return "\n".join(p for p in parts if p.strip())
    except ImportError:
        pass
    import xml.etree.ElementTree as ET

    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    root = ET.fromstring(xml_bytes)
    # Paragraphs inside tables are read once, as part of their row (cells tab-separated), not again on their own.
    in_tables = {id(p) for tbl in root.iter(f"{ns}tbl") for p in tbl.iter(f"{ns}p")}
    lines = []
    for el in root.iter():
        if el.tag == f"{ns}p" and id(el) not in in_tables:
            lines.append("".join(t.text or "" for t in el.iter(f"{ns}t")))
        elif el.tag == f"{ns}tr":
            lines.append("\t".join("".join(t.text or "" for t in tc.iter(f"{ns}t")) for tc in el.iter(f"{ns}tc")))
    return "\n".join(ln for ln in lines if ln.strip())


def _document_part(data: bytes | None, filename: str | None, text: str | None):
    """Return ("pdf", bytes) or ("text", str); raise DescribeError for unsupported, empty or oversized input."""
    if text is not None and not data:
        text = str(text)
        if len(text.encode("utf-8", errors="ignore")) > DESCRIBE_MAX_BYTES:
            raise DescribeError("Text is too large (limit 20 MB).", 413, "too_large")
        if not text.strip():
            raise DescribeError("The description is empty.", 400, "empty")
        return "text", text[:DESCRIBE_MAX_TEXT_CHARS]
    if not data:
        raise DescribeError("Upload a .pdf, .docx, .txt or .md file, or send {\"text\": ...}.", 400, "empty")
    if len(data) > DESCRIBE_MAX_BYTES:
        raise DescribeError(f"File is too large ({len(data) / 1e6:.1f} MB; limit 20 MB).", 413, "too_large")
    ext = Path(filename or "").suffix.lower()
    if not ext:  # sniff
        ext = ".pdf" if data[:5] == b"%PDF-" else ".docx" if data[:2] == b"PK" else ".txt"
    if ext not in DESCRIBE_EXTS:
        raise DescribeError(f"Unsupported file type '{ext}'. Use .pdf, .docx, .txt or .md.", 415, "unsupported_type")
    if ext == ".pdf":
        if data[:5] != b"%PDF-":
            raise DescribeError("This file is not a valid PDF.", 415, "unsupported_type")
        return "pdf", data
    if ext == ".docx":
        try:
            body = _docx_text(data)
        except DescribeError:
            raise
        except Exception:
            raise DescribeError("Could not read this .docx file.", 415, "unsupported_type")
    else:
        body = data.decode("utf-8", errors="replace")
    if not body.strip():
        raise DescribeError("The document contains no text.", 400, "empty")
    return "text", body[:DESCRIBE_MAX_TEXT_CHARS]


def _genai_client(project: str, location: str):
    from google import genai
    from google.genai import types

    from aiops_dilemma import auth

    from aiops_dilemma import regions
    return genai.Client(vertexai=True, project=project, location=regions.require(location, allow_global=True),
                        credentials=auth.credentials(project),
                        http_options=types.HttpOptions(timeout=180_000))


def _is_auth_error(e: Exception) -> bool:
    name = type(e).__name__
    msg = str(e)
    return (name in ("DefaultCredentialsError", "RefreshError", "TransportError")
            or getattr(e, "code", None) == 401 or "UNAUTHENTICATED" in msg
            or "Reauthentication is needed" in msg or "default credentials were not found" in msg.lower())


def _is_model_unavailable(e: Exception) -> bool:
    """Errors that mean "try the next model": not found / not allowed for this model / overloaded."""
    code = getattr(e, "code", None)
    msg = str(e).lower()
    if code in (404, 429, 500, 503, 504):
        return True
    return code in (400, 403) and "model" in msg and any(
        s in msg for s in ("not found", "not supported", "does not have access", "not available", "unsupported"))


def _num(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f and f not in (float("inf"), float("-inf")) else None


def _validate_value(path: str, item: dict, notes: list[str]):
    """Coerce one model-proposed field to the tool's units; return None (and add a note) when unusable."""
    kind, lo, hi = DESCRIBE_FIELDS[path]
    if kind == "task":
        t = str(item.get("text") or "").strip().lower()
        return t if t in TASK_TYPES else None
    if kind == "text":
        t = " ".join(str(item.get("text") or "").split())
        return t[:500] or None
    if kind == "hours":
        nums = [n for n in (_num(x) for x in (item.get("numbers") or [])) if n is not None]
        if len(nums) != 2:
            return None
        a, b = (int(round(min(max(n, 0), 24))) for n in nums)
        if a >= b:
            notes.append(f"Ignored busy hours {nums}: the window must start before it ends within one UTC day.")
            return None
        return [a, b]
    if kind == "hourly":
        nums = [_num(x) for x in (item.get("numbers") or [])]
        if not nums or any(n is None for n in nums):
            return None
        if len(nums) == 24:
            nums = nums * 7
            notes.append("The hourly pattern covers one day; it was repeated for every day of the week.")
        elif len(nums) == 1008:  # 10-minute points -> hourly averages
            nums = [sum(nums[i:i + 6]) / 6 for i in range(0, 1008, 6)]
        if len(nums) != 168:
            notes.append(f"Ignored the hourly pattern: expected 168 hourly values, got {len(nums)}.")
            return None
        vals = [round(max(0.0, n), 1) for n in nums]
        return vals if max(vals) > 0 else None
    n = _num(item.get("number"))
    if n is None and item.get("numbers"):
        n = _num((item.get("numbers") or [None])[0])
    if n is None or n < 0 or (n == 0 and path not in _ZERO_OK):
        return None
    if kind == "share" and 1 < n <= 100:  # a percentage where a fraction was asked for
        n /= 100.0
    if hi is not None and n > hi:
        notes.append(f"{path} was {n:g}, above the maximum {hi:g}; clamped.")
        n = hi
    if path in _INT_FIELDS:
        return int(round(n))
    return round(n, 6) if n < 1 else round(n, 3)


def validate_description(raw: dict) -> dict:
    """Turn Gemini's {fields, notes} into {workload, fields, notes}: known paths only, units and ranges enforced."""
    notes = [" ".join(str(n).split())[:400] for n in (raw.get("notes") or []) if str(n).strip()][:20]
    workload: dict[str, dict] = {}
    fields: list[dict] = []
    seen: set[str] = set()
    for item in raw.get("fields") or []:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or "")
        if path not in DESCRIBE_FIELDS or path in seen:
            continue
        value = _validate_value(path, item, notes)
        if value is None:
            continue
        seen.add(path)
        conf = str(item.get("confidence") or "").lower()
        fields.append({"path": path, "value": value, "confidence": conf if conf in CONFIDENCE else "assumed",
                       "evidence": " ".join(str(item.get("evidence") or "").split())[:300]})
        sec, key = path.split(".", 1)
        workload.setdefault(sec, {})[key] = value

    t, r = workload.get("traffic", {}), workload.get("requests", {})
    if "pct_over_128k" in r and r["pct_over_128k"] > r.get("pct_over_32k", 100.0):
        r["pct_over_128k"] = r.get("pct_over_32k", 0.0)
        for f in fields:
            if f["path"] == "requests.pct_over_128k":
                f["value"] = r["pct_over_128k"]
        notes.append("Share over 128K exceeded the share over 32K; capped to it.")
    pk, av, fl = t.get("peak_input_tps"), t.get("average_input_tps"), t.get("floor_input_tps")
    if (pk is not None and av is not None and pk < av) or (fl is not None and av is not None and fl > av) or \
            (pk is not None and fl is not None and fl > pk):
        notes.append("Peak, average and quietest-hour traffic are inconsistent (need quietest ≤ average ≤ peak); "
                     "check them before estimating.")
    return {"workload": workload, "fields": fields, "notes": notes}


def describe_workload(data: bytes | None = None, filename: str | None = None, text: str | None = None,
                      project: str | None = None, location: str = "global") -> dict:
    """Ask Gemini on Agent Platform to extract workload inputs from a document; nothing is persisted.

    Returns {workload, fields: [{path, value, confidence, evidence}], notes, model}. Raises DescribeError.
    """
    kind, payload = _document_part(data, filename, text)
    if not project or project == "my-gcp-project":
        from aiops_dilemma import auth

        project = auth.app_project()
    if not project or project == "my-gcp-project":
        raise DescribeError("No Google Cloud project is configured. Sign in and choose a project to read documents "
                            "with Gemini on Agent Platform.", 401, "not_signed_in")

    from google.genai import types

    doc = (types.Part.from_bytes(data=payload, mime_type="application/pdf") if kind == "pdf"
           else types.Part.from_text(text=f"<document>\n{payload}\n</document>"))
    contents = [types.Content(role="user", parts=[doc, types.Part.from_text(
        text="Extract the workload fields this document supports, following the instructions.")])]
    cfg = types.GenerateContentConfig(
        system_instruction=DESCRIBE_PROMPT,
        temperature=0.1,
        response_mime_type="application/json",
        response_schema=_describe_schema(),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    try:
        client = _genai_client(project, location)
    except Exception as e:
        if _is_auth_error(e):
            raise DescribeError("Not signed in to Google Cloud. Sign in to read documents with Gemini on "
                                "Agent Platform.", 401, "not_signed_in")
        raise DescribeError(f"Could not reach Gemini on Agent Platform: {str(e)[:200]}", 502, "model_error")

    override = os.environ.get("DILEMMA_DESCRIBE_MODEL", "").strip()
    candidates = ([override] if override else []) + [m for m in DESCRIBE_MODELS if m != override]
    last_err: Exception | None = None
    for model_id in candidates:
        try:
            resp = client.models.generate_content(model=model_id, contents=contents, config=cfg)
        except Exception as e:
            if _is_auth_error(e):
                raise DescribeError("Not signed in to Google Cloud (or the credentials expired). Sign in to read "
                                    "documents with Gemini on Agent Platform.", 401, "not_signed_in")
            if _is_model_unavailable(e):
                log.info("describe_workload: %s unavailable (%s), trying the next model", model_id, str(e)[:120])
                last_err = e
                continue
            if getattr(e, "code", None) == 403:
                raise DescribeError(f"Permission denied calling Gemini on Agent Platform in project {project}: "
                                    f"{str(e)[:200]}", 403, "permission_denied")
            raise DescribeError(f"Gemini on Agent Platform could not read the document: {str(e)[:200]}",
                                502, "model_error")
        raw = getattr(resp, "parsed", None)
        if not isinstance(raw, dict):
            try:
                raw = json.loads(getattr(resp, "text", None) or "")
            except (TypeError, ValueError):
                raise DescribeError("Gemini returned an unreadable answer; try again.", 502, "bad_response")
        log.info("describe_workload answered by %s", model_id)
        out = validate_description(raw if isinstance(raw, dict) else {})
        out["model"] = model_id
        return out
    raise DescribeError("No Gemini Flash model is available on Agent Platform in project "
                        f"{project} ({str(last_err)[:160]}).", 503, "model_unavailable")
