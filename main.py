"""Forecast GTA VI's launch-day Twitch viewers hour by hour.

Method (per case, fitted at run time from the case file's own comparables):

  The target is a rolling-midnight console launch: it unlocked in New
  Zealand 13 hours before the UTC release day started, the UK unlocked at
  00:00 UTC, the US east coast at 05:00 UTC and the west coast at 08:00 UTC.
  Past launches with the same shape are the ones that (a) were already
  being watched in hour 1 (hour-1 viewers at least 2% of their release-day
  peak) and (b) first crossed 10% of that peak within the 24 hours before
  their release day started -- i.e. they launched the night before, not
  days earlier on early access and not at midday on the release day itself.
  Training on the other families (daytime simultaneous launches, early
  access launches) measurably worsens the forecast for this family.

  For each hour h = 2..24, fit an ordinary least squares line in log space

      log viewers(h) = a_h + b_h * log viewers(1) + c_h * log viewers(-1h)

  over the training launches, where viewers(1) is the case's hour-1 print
  and viewers(-1h) is the category's viewers in the hour before the day
  started (23:00 UTC the day before: 12 hours after the New Zealand
  unlock, just before the UK unlock). The second feature measures how
  strong the already-unlocked eastern markets are, which separates games
  whose later ramp is US-driven from games whose audience is already
  largely awake. The five quantiles are the fitted mean plus the standard
  normal quantiles times 0.9 * the residual standard deviation of that
  hour's fit (the 0.9 shrinks the spread slightly toward the sharpness the
  four practice launches reward).

  Standard library only, no network, deterministic: the same case always
  writes the same bytes. One case runs in well under a second.

  Local validation (practice answers re-read from the kit's own documented
  source, SullyGnome hourly averages; hour-1 prints match the case files
  exactly): mean pinball loss on the log scale 0.163 over the four practice
  launches, against 0.387 for the house baseline starter on the same cases.
  Leave-one-out over the comparable launches in the same family: 0.251,
  against 0.263 for the same fit without the launch-family filter.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file would import. (This entry is a single file.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# Quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Spread multiplier on the residual standard deviation (see docstring).
SIGMA_MULT = 0.9
# A training launch counts when hour 1 is at least this share of its peak.
POST_UNLOCK_FRAC = 0.02
# ... and when it first crossed 10% of its peak inside this window (hours
# relative to its release day's start): launched the night before.
LAUNCH_WINDOW = (-24, -9)
LAUNCH_FRAC = 0.10
# How far back the launch-hour search looks (hours before day start).
LAUNCH_SEARCH_BACK = 48
# Fallback when too few training launches have an hour's print.
MIN_TRAIN = 8
# Fallback persistence spread, as in the house baseline.
FALLBACK_SIGMA = 0.15


def parse_time(text):
    """ISO 8601 with a Z suffix to an aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def norm_ppf(p):
    """Standard normal quantile (Acklam's rational approximation).

    Accurate to ~1e-9 in the central region, plenty for a forecast spread.
    """
    if p <= 0.0 or p >= 1.0:
        raise ValueError(f"level {p} is not in (0, 1)")
    a = [-3.969683028665376e01, 2.209460984245205e02,
         -2.759285984469534e02, 1.383577518672690e02,
         -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02,
         -1.556989798598866e02, 6.680131188771972e01,
         -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e00, -2.549732539343734e00,
         4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e00, 3.754408661907416e00]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if phigh < p:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def round2(x):
    """x rounded to 2 decimals, half up (as the starter rounds it)."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value == int(value) else value


def ols(X, y):
    """Least squares with an intercept. X: rows of features, y: targets.

    Returns (coefficients with the intercept first, residual std with a
    degrees-of-freedom correction), or (None, None) when singular.
    """
    n = len(X)
    p = len(X[0]) + 1
    rows = [[1.0] + list(x) for x in X]
    # Normal equations, solved by Gauss-Jordan with partial pivoting.
    m = [[sum(rows[i][a] * rows[i][b] for i in range(n)) for b in range(p)]
         for a in range(p)]
    v = [sum(rows[i][a] * y[i] for i in range(n)) for a in range(p)]
    for col in range(p):
        piv = max(range(col, p), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            return None, None
        m[col], m[piv] = m[piv], m[col]
        v[col], v[piv] = v[piv], v[col]
        for r in range(p):
            if r != col:
                f = m[r][col] / m[col][col]
                for k in range(col, p):
                    m[r][k] -= f * m[col][k]
                v[r] -= f * v[col]
    coef = [v[i] / m[i][i] for i in range(p)]
    resid = [y[i] - sum(coef[j] * rows[i][j] for j in range(p))
             for i in range(n)]
    dof = max(n - p, 1)
    return coef, math.sqrt(sum(r * r for r in resid) / dof)


def hourly_map(points, key="from"):
    """{hour-start datetime: viewers} from a list of hourly points."""
    out = {}
    for point in points:
        try:
            at = parse_time(point[key])
            viewers = point.get("viewers")
        except (KeyError, TypeError, ValueError):
            continue
        if viewers is not None and viewers > 0:
            out[at] = viewers
    return out


def launch_hour(hourly_points, day_start):
    """Hours from the release day's start to the launch's first big hour.

    The first hour in [day_start - 48h, day_start + 24h] whose viewers reach
    10% of the release day's peak. None when the peak is unknown or the
    threshold is never reached in the window.
    """
    by_hour = hourly_map(hourly_points)
    peak = max((v for t, v in by_hour.items()
                if day_start <= t < day_start + timedelta(hours=24)), default=0)
    if not peak:
        return None
    for offset in range(-LAUNCH_SEARCH_BACK, 25):
        v = by_hour.get(day_start + timedelta(hours=offset))
        if v is not None and v >= LAUNCH_FRAC * peak:
            return offset
    return None


def is_training_launch(curve, launch):
    """A rolling-midnight launch: watched in hour 1, launched overnight."""
    y1 = curve.get(1)
    peak = max((v for v in curve.values() if v), default=0)
    if not y1 or not peak:
        return False
    if y1 < POST_UNLOCK_FRAC * peak:
        return False
    if launch is None or not (LAUNCH_WINDOW[0] <= launch <= LAUNCH_WINDOW[1]):
        return False
    return True


def forecast_case(case):
    """One case's forecast: {"hours": [{"hour": h, "q": [...]}, ...]}."""
    levels = case.get("levels") or LEVELS
    z = [norm_ppf(level) for level in levels]
    known = next((k for k in case["known"] if k["hour"] == 1), None)
    if known is None:
        raise ValueError("case has no hour-1 print")
    p1 = max(1, known["viewers"])

    day_start = parse_time(case["day_start"])
    # The target's pre-day viewers, from the hourly history; minute reads
    # fill any hour the hourly history lacks (the final case records every
    # minute, the practice cases only hourly points).
    pre = hourly_map(case.get("history_hourly") or [])
    if case.get("history"):
        by_hour = {}
        for point in case["history"]:
            try:
                at = parse_time(point["at"])
            except (KeyError, TypeError, ValueError):
                continue
            v = point.get("viewers")
            if v is not None and v >= 0:
                by_hour.setdefault(at.replace(minute=0, second=0,
                                             microsecond=0), []).append(v)
        for hour_start, vs in by_hour.items():
            pre.setdefault(hour_start, sum(vs) / len(vs))
    pre_1h = pre.get(day_start - timedelta(hours=1)) or p1

    # Training launches, fitted fresh from this case's comparables.
    train = []
    for comp in case.get("comparables") or []:
        hourly = comp.get("hourly") or []
        comp_start = parse_time(comp["day_start"])
        by_hour = hourly_map(hourly)
        curve = {h: by_hour.get(comp_start + timedelta(hours=h - 1))
                 for h in range(1, 25)}
        launch = launch_hour(hourly, comp_start)
        if not is_training_launch(curve, launch):
            continue
        pre_comp = by_hour.get(comp_start - timedelta(hours=1))
        if not pre_comp:
            continue
        train.append((math.log(curve[1]), math.log(pre_comp), curve))

    target = [math.log(p1), math.log(max(1, pre_1h))]
    rows_to_forecast = case.get("forecast") or [{"hour": h} for h in range(2, 25)]
    hours = []
    for row in sorted(rows_to_forecast, key=lambda r: r["hour"]):
        h = row["hour"]
        rows = [(x1, x2, math.log(c[h])) for (x1, x2, c) in train if c.get(h)]
        coef = sig = None
        if len(rows) >= MIN_TRAIN:
            coef, sig = ols([(x1, x2) for (x1, x2, _) in rows],
                            [y for (_, _, y) in rows])
        if coef is None or sig is None:
            # Too little training data for this hour: persistence, with a
            # spread that widens over the day like the house baseline.
            quantiles = [p1 * math.exp(zz * FALLBACK_SIGMA * math.sqrt(h - 1))
                         for zz in z]
        else:
            mu = coef[0] + coef[1] * target[0] + coef[2] * target[1]
            spread = SIGMA_MULT * sig
            quantiles = [math.exp(mu + zz * spread) for zz in z]
        # Clamp to the contest's bounds, round, keep non-decreasing.
        clean = []
        prev = 0.01
        for q in quantiles:
            q = min(max(q, 0.01), 1_000_000_000)
            q = round2(q)
            q = max(q, prev)
            clean.append(q)
            prev = q
        hours.append({"hour": h, "q": clean})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
