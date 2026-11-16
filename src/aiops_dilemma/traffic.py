"""Weekly traffic profile: 168 hourly-average input tokens/sec (Monday 00:00 UTC first).

Sources, in order of precedence: your own series (`hourly_csv`), a user-drawn or document-inferred week
(`custom_hourly`, 168 values), a named traffic shape (`preset`, see PRESETS), or a parametric week from peak / average / floor, a weekday busy window and a
weekend factor. The parametric shape is solved so that the busiest hour, the quietest hour and the weekly
average all match what you specify.

`monthly_input_tokens` (optional) anchors the volume: the parametric week is solved for the average it implies
(peak and floor kept; it overrides `average_input_tps`), and a CSV / custom / preset week is rescaled to that total.
"""

import csv
import math

import numpy as np


def _window(busy: tuple[int, int], ramp: float = 5.0) -> np.ndarray:
    """0..1 weekday shape: plateau over the busy window, raised-cosine ramps on each side."""
    start, end = busy
    h = np.arange(24) + 0.5
    s = np.zeros(24)
    for i, x in enumerate(h):
        if start <= x <= end:
            s[i] = 1.0
        elif start - ramp < x < start:
            s[i] = 0.5 - 0.5 * math.cos(math.pi * (x - (start - ramp)) / ramp)
        elif end < x < end + ramp:
            s[i] = 0.5 + 0.5 * math.cos(math.pi * (x - end) / ramp)
    return s


def parametric(peak: float, average: float, floor: float, busy=(13, 20), weekend_factor=0.15) -> np.ndarray:
    if not floor <= average <= peak:
        raise ValueError("need floor <= average <= peak")
    base = _window(tuple(busy))

    def week(p: float) -> np.ndarray:
        weekday = floor + (peak - floor) * base ** p
        weekend = floor + weekend_factor * (weekday - floor)
        return np.concatenate([weekday] * 5 + [weekend] * 2)

    lo, hi = 1e-3, 50.0                       # shape exponent: small = fat plateau, large = spiky
    for _ in range(80):
        mid = math.sqrt(lo * hi)
        if week(mid).mean() > average:
            lo = mid
        else:
            hi = mid
    w = week(math.sqrt(lo * hi))
    if abs(w.mean() - average) / average > 0.02:
        raise ValueError(f"can't reach average {average:,.0f} with this peak/floor/weekend factor "
                         f"(achievable range ~{week(50).mean():,.0f}–{week(1e-3).mean():,.0f})")
    return w


def from_csv(path: str) -> np.ndarray:
    from aiops_dilemma import paths
    with open(paths.check_readable(path)) as f:   # confined to the session folder on a shared deployment
        rows = [float(r["input_tokens_per_sec"]) for r in csv.DictReader(f)]
    if len(rows) != 168:
        raise ValueError(f"{path}: expected 168 hourly rows, got {len(rows)}")
    return np.array(rows)


HOURS_PER_MONTH = 730


def monthly_tokens(profile: np.ndarray) -> float:
    """Monthly input tokens of a weekly hourly-average tok/s profile (730 h per month)."""
    return float(np.mean(profile)) * 3600 * HOURS_PER_MONTH


def from_custom(values) -> np.ndarray:
    from aiops_dilemma.config import clean_custom_hourly
    clean, err = clean_custom_hourly(values)
    if err or clean is None:
        raise ValueError(err or "traffic.custom_hourly is empty")
    return np.array(clean, dtype=float)


def _monthly(traffic: dict) -> float | None:
    try:
        m = float(traffic.get("monthly_input_tokens") or 0)
    except (TypeError, ValueError):
        return None
    return m if m > 0 else None


def rescale_to_monthly(profile: np.ndarray, monthly: float | None) -> np.ndarray:
    if not monthly:
        return profile
    cur = monthly_tokens(profile)
    return profile * (monthly / cur) if cur > 0 else profile


# Named traffic shapes ------------------------------------------------------------------------------------------------
# Each region is busy over the same local hours (a smooth bump), weighted by its share of users; nights sit at
# `floor` x the peak and weekends at `weekend` x the weekday shape. Same monthly volume, very different peaks.

_REGION_UTC_OFFSET = {"us_east": -5, "us_west": -8, "eu": 1, "apac": 8}

PRESETS = {
    "us_core_hours": dict(mix={"us_east": .8, "us_west": .2}, floor=.015, weekend=.04, hours=(10, 16.5)),
    "us_europe": dict(mix={"us_east": .35, "us_west": .2, "eu": .45}, floor=.08, weekend=.2, hours=(8.5, 18)),
    "follow_the_sun": dict(mix={"us_east": .3, "us_west": .1, "eu": .3, "apac": .3}, floor=.15, weekend=.35,
                           hours=(8.5, 18)),
    "steady": dict(mix={"us_east": .25, "us_west": .25, "eu": .25, "apac": .25}, floor=.7, weekend=.9,
                   hours=(8.5, 18)),
}

PRESET_INFO = {
    "us_core_hours": ("US business hours",
                      "US users only, busy during the working day; quiet nights and almost empty weekends."),
    "us_europe": ("US + Europe", "Users in the US and Europe: a long working day across both, quieter weekends."),
    "follow_the_sun": ("Follow the sun (Americas, Europe, Asia)",
                       "Users on three continents: someone is always working, weekends still dip."),
    "steady": ("Steady, machine-driven", "Batch jobs and agents running around the clock: nearly flat."),
}


def _bump(start: float, end: float, h: float, ramp: float = 1.5) -> float:
    def s(x):
        return 1 / (1 + math.exp(-x / (ramp / 3)))
    h = h % 24
    return max(s(h - start) * s(end - h), s(h + 24 - start) * s(end - h - 24), s(h - 24 - start) * s(end - h + 24))


def _preset_shape(mix: dict, floor: float, weekend: float, hours: tuple) -> np.ndarray:
    day = np.array([sum(w * _bump(hours[0], hours[1], h + _REGION_UTC_OFFSET[r]) for r, w in mix.items())
                    for h in range(24)])
    day = day / day.max()
    wkd = floor + (1 - floor) * day
    wke = floor + (1 - floor) * day * weekend
    return np.concatenate([wkd] * 5 + [wke] * 2)      # Monday 00:00 UTC first


def preset_week(name: str, average_tps: float) -> np.ndarray:
    """168 hourly input tok/s for a named traffic shape, scaled so its mean equals `average_tps`."""
    if name not in PRESETS:
        raise ValueError(f"unknown traffic preset {name!r} (choose from {', '.join(PRESETS)})")
    shape = _preset_shape(**PRESETS[name])
    return shape * (float(average_tps) / shape.mean())


def preset_list() -> list[dict]:
    """Presets for the UI: key, name, one-line description, peak / average."""
    out = []
    for key in PRESETS:
        w = _preset_shape(**PRESETS[key])
        name, desc = PRESET_INFO[key]
        out.append({"key": key, "name": name, "description": desc, "peak_to_average": round(float(w.max() / w.mean()), 2),
                    "average_to_floor": round(float(w.mean() / w.min()), 1)})
    return out


def weekly_profile(traffic: dict) -> np.ndarray:
    """168 hourly input tok/s. Precedence: hourly_csv > custom_hourly > preset > parametric;
    monthly_input_tokens anchors volume."""
    monthly = _monthly(traffic)
    if traffic.get("hourly_csv"):
        return rescale_to_monthly(from_csv(traffic["hourly_csv"]), monthly)
    if traffic.get("custom_hourly"):
        return rescale_to_monthly(from_custom(traffic["custom_hourly"]), monthly)
    if traffic.get("preset"):
        avg = monthly / (3600 * HOURS_PER_MONTH) if monthly else float(traffic.get("average_input_tps") or 0)
        if avg <= 0:
            raise ValueError("traffic.preset needs average_input_tps or monthly_input_tokens")
        return preset_week(traffic["preset"], avg)
    shape = (traffic.get("busy_hours_utc", (13, 20)), traffic.get("weekend_factor", 0.15))
    if monthly:
        avg = monthly / (3600 * HOURS_PER_MONTH)
        try:   # the monthly total overrides the average; peak and floor stay
            return parametric(traffic["peak_input_tps"], avg, traffic["floor_input_tps"], *shape)
        except ValueError:
            if not traffic.get("average_input_tps"):
                raise
        # Not reachable with this peak / floor: keep the shape of the stated week, scaled to the monthly total.
        return rescale_to_monthly(parametric(traffic["peak_input_tps"], traffic["average_input_tps"],
                                             traffic["floor_input_tps"], *shape), monthly)
    return parametric(traffic["peak_input_tps"], traffic["average_input_tps"], traffic["floor_input_tps"], *shape)


def raise_off_peak(demand: np.ndarray, factor: float) -> np.ndarray:
    """Off-peak hours grow `factor`x (e.g. other time zones; clamped to 1-10); the busiest hour stays the same."""
    factor = min(10.0, max(1.0, float(factor)))
    return np.minimum(demand * factor, demand.max())


def clipped_lognormal_mu(target_mean: float, sigma: float, lo: float, hi: float) -> float:
    """mu such that a lognormal(mu, sigma) clipped to [lo, hi] has mean `target_mean`.

    Clipping changes the mean (the upper tail is cut at `hi`, the lower raised to `lo`), so mu = ln(mean) - sigma^2/2
    undershoots when the target is a sizeable share of `hi`. E[clip(X)] = lo P(X<lo) + hi P(X>hi) + E[X; lo<X<hi],
    increasing in mu, solved by bisection. Targets outside (lo, hi) are clamped just inside.
    """
    def phi(z: float) -> float:
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    def clipped_mean(mu: float) -> float:
        a, b = (math.log(lo) - mu) / sigma, (math.log(hi) - mu) / sigma
        mid = math.exp(mu + sigma * sigma / 2) * (phi(b - sigma) - phi(a - sigma))
        return lo * phi(a) + hi * (1.0 - phi(b)) + mid

    t = min(max(float(target_mean), lo * 1.001), hi * 0.999)
    a, b = math.log(lo) - 6 * sigma, math.log(hi) + 6 * sigma
    for _ in range(100):
        m = (a + b) / 2
        if clipped_mean(m) < t:
            a = m
        else:
            b = m
    return (a + b) / 2
