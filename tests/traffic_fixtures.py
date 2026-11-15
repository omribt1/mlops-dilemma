"""Synthetic traffic exports for the time-series importer tests and UI checks (no real data).

Shapes mimic Cloud Monitoring Metrics Explorer's "Download CSV": `label,value` header rows (fake TimeSeries ID,
project `my-gcp-project`), then JavaScript Date.toString() timestamps with a non-UTC offset and 5-minute points.

    python tests/traffic_fixtures.py /tmp/out-dir     # writes the synthetic CSVs used by the browser check
"""

import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
OFFSET_H = 2                      # the exporter's browser time zone (GMT+0200)
TOKENS_PER_REQUEST = 3000         # synthetic: 2,800 input + 200 output
INPUT_PER_REQUEST = 2800


def js_time(dt_utc: datetime, offset_h: int = OFFSET_H) -> str:
    loc = dt_utc + timedelta(hours=offset_h)
    sign = "+" if offset_h >= 0 else "-"
    return (f"{DOW[loc.weekday()]} {MON[loc.month - 1]} {loc.day:02d} {loc.year} {loc:%H:%M:%S} "
            f"GMT{sign}{abs(offset_h):02d}00 (Example Standard Time)")


def rps_at(dt_utc: datetime) -> float:
    """Requests/s: a smooth weekday bump around 15:00 UTC, weekends at 20%."""
    x = (dt_utc.hour + dt_utc.minute / 60 - 15) / 3.0
    v = 0.2 + 1.8 * math.exp(-x * x)
    return v * (0.2 if dt_utc.weekday() >= 5 else 1.0)


def points(start_utc: datetime, hours: float, step_min: int = 5, gaps=()):
    """[(utc time, rps)] every step; `gaps` = [(from_hour, to_hour)] offsets with no points."""
    out = []
    n = int(hours * 60 / step_min)
    for i in range(n):
        t = start_utc + timedelta(minutes=step_min * i)
        off_h = (t - start_utc).total_seconds() / 3600
        if any(a <= off_h < b for a, b in gaps):
            continue
        out.append((t, rps_at(t)))
    return out


def metrics_explorer_csv(kind: str, pts, model: str = "example-model", offset_h: int = OFFSET_H,
                         label_order=("TimeSeries ID", "project_id", "location", "model_user_id")) -> str:
    """kind: 'qps' (requests/s) or 'tokens' (total tokens/s, no input/output split)."""
    labels = {"TimeSeries ID": "00000000-1111-2222-3333-444444444444-0-0" if kind == "qps"
              else "99999999-8888-7777-6666-555555555555-0-0",
              "project_id": "my-gcp-project", "location": "global", "model_user_id": model}
    rows = [f"{k},{labels[k]}" for k in label_order]
    mult = 1.0 if kind == "qps" else TOKENS_PER_REQUEST
    rows += [f"{js_time(t, offset_h)},{v * mult!r}" for t, v in pts]
    return "\n".join(rows)


def multi_series_csv(pts) -> str:
    """Two series (two locations) in one export: one timestamp column, one value column per series."""
    rows = ["TimeSeries ID,aaaa-0-0,bbbb-0-0", "project_id,my-gcp-project,my-gcp-project",
            "location,us-central1,europe-west4", "model_user_id,example-model,example-model"]
    rows += [f"{js_time(t)},{v * 0.75!r},{v * 0.25!r}" for t, v in pts]
    return "\n".join(rows)


def block_series_csv(pts) -> str:
    """Two series as side-by-side column blocks (timestamp, value, timestamp, value)."""
    rows = ["TimeSeries ID,aaaa-0-0,TimeSeries ID,bbbb-0-0", "type,input,type,output"]
    rows += [f"{js_time(t)},{v * INPUT_PER_REQUEST!r},{js_time(t)},{v * 200!r}" for t, v in pts]
    return "\n".join(rows)


def iso_csv(pts, column: str = "value") -> str:
    rows = [f"timestamp,{column}"] + [f"{t.strftime('%Y-%m-%dT%H:%M:%SZ')},{v!r}" for t, v in pts]
    return "\n".join(rows)


START_PARTIAL = datetime(2026, 11, 1, 19, 50, tzinfo=timezone.utc)   # a Sunday evening (UTC)
START_WEEKS = datetime(2026, 10, 5, 0, 0, tzinfo=timezone.utc)       # a Monday


def partial_files() -> dict[str, str]:
    """< 24 h (16 h, Sun 19:50 - Mon 11:40 UTC) with a 3-hour gap: the QPS + token count pair."""
    pts = points(START_PARTIAL, 15 + 55 / 60, gaps=[(5, 8)])   # last point Mon 11:40 UTC
    return {"Model_requests_per_second_(QPS)_example.csv": metrics_explorer_csv("qps", pts),
            "Token_count_by_model_(per_second)_example.csv": metrics_explorer_csv(
                "tokens", pts, label_order=("TimeSeries ID", "location", "model_user_id", "project_id"))}


def weeks_files(days: int = 14) -> dict[str, str]:
    pts = points(START_WEEKS, days * 24, step_min=5, gaps=[(30, 33)])
    return {"Model_requests_per_second_(QPS)_weeks.csv": metrics_explorer_csv("qps", pts)}


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/ux-traffic/files")
    out.mkdir(parents=True, exist_ok=True)
    for name, text in {**partial_files(), **weeks_files()}.items():
        (out / name).write_text(text)
    print(out)
