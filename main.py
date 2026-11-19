"""GTA VI trophy rarity forecaster (Python 3.13, standard library only).

Model: earners-over-tracked decomposition with a refresh-aware spread.

The page Exophase prints is not a live feed: it is frozen between Exophase's
own refreshes, then jumps.  A trophy's earn rate is earners / players-tracked,
and a refresh dilutes every rate at once (launch night: 638 -> 3,868
players tracked, every rate but the 100% one down), so the forecast is
built around those refresh steps rather than a smooth trend:

* Distinct snapshots.  Consecutive identical reads are one Exophase state,
  so the history is collapsed to the states that actually changed.

* Trajectory branch (two or more distinct states, the shape the final case
  has): per trophy, logit(rate) is fitted against log(1 + hours since
  release) by least squares on the recent window (the last 24 hours of
  states, where the current regime lives), the slope clipped and shrunk
  toward the pooled slope across trophies, and blended with the fit over
  the full history; the projection is damped toward the last observed rate
  in proportion to how far the target lies past the observed range, so a
  4.5-day extrapolation cannot run away on a short-window slope.  The
  spread grows with the horizon and with each fit's residual, so a 4.5-day
  projection is honestly much wider than a 90-minute one.

* Flat branch (one distinct state, the shape the practice cases have):
  every read is the same launch-night page, so no trend exists to fit.
  Instead the model expects the next Exophase refresh inside the horizon:
  the tracked base multiplies (launch-night prior, calibrated on the one
  refresh the practice answers show: 638 -> 3,868 players) and the new
  players earn each trophy at a rate that depends on what kind of trophy it
  is -- story missions are earned by almost none of the newcomers, the
  first chapter and the quick early activities (a fishing challenge, an
  arcade game, discovering vehicles) by far more.  The median is the old
  earners plus the newcomers' earners, over the new tracked base.

* The trophy every tracked player holds (100% in every read) is forecast
  at 100% flat: it is the intro mission, and a share that is already
  universal cannot dilute.

Quantiles are logit-normal around the median with the z-values
(-1.645, -0.674, 0, 0.674, 1.645).  Everything is deterministic: no
randomness, no wall-clock use, no network.  The target and horizon come
from the case file itself.
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# --- flat-branch (refresh) prior, calibrated on the practice refresh -------
# One Exophase refresh on launch night multiplied the tracked base ~6x
# (638 -> 3,868 players).  The flat branch assumes one such refresh lands
# inside the horizon.
REFRESH_MULTIPLIER = 6.0
# Share of newly tracked players who earn a trophy of each kind, in percent.
# Story missions: almost none of the newcomers have played that far yet.
# Chapter 1 and the quick early activities: far more of them have.
NEW_EARN_PCT = {"story": 0.06, "ch1": 0.34, "quick": 0.40, "other": 0.03}
# Spread of the flat branch in logit units (tuned on the practice backtest).
FLAT_SIGMA = 0.20

# --- trajectory-branch constants -------------------------------------------
# Spread floor, per-square-root-day growth of the spread, and its cap.
TRAJ_SIGMA_FLOOR = 0.20
TRAJ_SIGMA_PER_SQRT_DAY = 0.30
TRAJ_SIGMA_CAP = 1.60

_STORY_RE = re.compile(
    r"^Complete (Chapter \d+|the final mission|Breakfast Run|The Cartel Heist"
    r"|Child's Play|Headliners|VCIA Takedown|Operation Savage Glory)",
    re.IGNORECASE,
)
_QUICK_RES = ("fishing challenge", "arcade", "discover 100 vehicles")


def clamp(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if float(value).is_integer() else value


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def trophy_kind(description):
    """What kind of trophy this is, from its own description."""
    d = (description or "").lower()
    if "chapter 1" in d:
        return "ch1"
    if any(q in d for q in _QUICK_RES):
        return "quick"
    if _STORY_RE.match(description or ""):
        return "story"
    return "other"


def ols(xs, ys):
    """Least-squares slope and intercept of ys on xs (lists, same length)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return 0.0, my
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    return b, my - b * mx


def distinct_states(case):
    """Collapse consecutive identical reads into one state each.

    Returns a list of (time, players_tracked, rates) in order, one per
    distinct Exophase state, using the first read that shows each state.
    """
    states = []
    prev_key = None
    for read in case["history"]:
        key = (read["players_tracked"], tuple(read["rates"]))
        if key != prev_key:
            states.append((parse_time(read["at"]), read["players_tracked"], list(read["rates"])))
            prev_key = key
    return states


def quantiles(median, sigma):
    if median >= 99.995:
        return [100, 100, 100, 100, 100]
    lg = logit(median)
    qs = []
    for z in Z:
        qs.append(round4(clamp(inv_logit(lg + z * sigma))))
    # Rounding can break monotonicity at the extremes; repair it.
    for i in range(1, len(qs)):
        if qs[i] < qs[i - 1]:
            qs[i] = qs[i - 1]
    return qs


def flat_forecast(case, states):
    """One distinct state: expect a refresh inside the horizon."""
    _, n_last, rates_last = states[-1]
    horizon_h = case["horizon_minutes"] / 60.0
    # A refresh is only expected if the horizon is long enough for one.
    mult = 1.0 + (REFRESH_MULTIPLIER - 1.0) * min(1.0, horizon_h)
    n_pred = n_last * mult
    new_players = n_pred - n_last
    sigma = FLAT_SIGMA * math.sqrt(1.0 + case["horizon_minutes"] / 1440.0)
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate = rates_last[i]
        if rate is None:
            rate = 0.0
        if rate >= 99.995:
            rows.append({"id": trophy["id"], "q": [100, 100, 100, 100, 100]})
            continue
        earners = rate / 100.0 * n_last
        kind = trophy_kind(trophy.get("description", ""))
        earners += NEW_EARN_PCT[kind] / 100.0 * new_players
        median = earners / n_pred * 100.0
        rows.append({"id": trophy["id"], "q": quantiles(median, sigma)})
    return {"trophies": rows}


def trajectory_forecast(case, states):
    """Two or more distinct states: fit the rate curve and project it.

    Logit(rate) is fitted on log(1 + hours since release), on the recent
    window first (the last 24 hours of states) and on the full history
    second; the two projections are blended, then damped toward the last
    observed rate for the part of the horizon that lies past the data.
    """
    release = datetime(2026, 11, 19, tzinfo=timezone.utc)
    try:
        rd = case.get("game", {}).get("release_date", "")
        release = datetime.fromisoformat(rd + "T00:00:00+00:00")
    except (ValueError, TypeError):
        pass
    target = parse_time(case["target"])
    checkpoint = parse_time(case["at"])
    x_target = math.log1p(max(0.0, (target - release).total_seconds() / 3600.0))
    horizon_days = case["horizon_minutes"] / 1440.0

    xs = [math.log1p(max(0.0, (t - release).total_seconds() / 3600.0)) for t, _, _ in states]
    times = [t for t, _, _ in states]
    # Recent window: states from the last 24 hours before the checkpoint;
    # fall back to the last four states, then to all states, if too few.
    recent_idx = [j for j, t in enumerate(times) if (checkpoint - t).total_seconds() <= 24 * 3600]
    if len(recent_idx) < 3:
        recent_idx = list(range(max(0, len(states) - 4), len(states)))
    if len(recent_idx) < 2:
        recent_idx = list(range(len(states)))
    x_max = max(xs)
    span_recent = max(xs[j] for j in recent_idx) - min(xs[j] for j in recent_idx)
    overshoot = max(0.0, x_target - x_max)
    damp = span_recent / (span_recent + overshoot) if span_recent > 0 else 1.0

    def fit_logit(idxs, i):
        ex, ey = [], []
        for j in idxs:
            r = states[j][2][i]
            if r is not None and 0 < r < 100:
                ex.append(xs[j])
                ey.append(logit(r))
        if len(ex) >= 2 and len(set(ex)) >= 2:
            b, a = ols(ex, ey)
            resid = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(ex, ey)) / len(ex))
            return min(2.0, max(-1.5, b)), a, min(resid, 0.8)
        if ey:
            return 0.0, ey[-1], 0.0
        return 0.0, None, 0.0

    n_trophies = len(case["trophies"])
    recent_fits = [fit_logit(recent_idx, i) for i in range(n_trophies)]
    full_fits = [fit_logit(list(range(len(states))), i) for i in range(n_trophies)]
    pooled = sorted(f[0] for f in recent_fits)[n_trophies // 2]
    k = len(recent_idx)
    w_self = k / (k + 3.0)

    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rates_last = states[-1][2][i]
        if rates_last is not None and rates_last >= 99.995:
            rows.append({"id": trophy["id"], "q": [100, 100, 100, 100, 100]})
            continue
        b_r, a_r, resid = recent_fits[i]
        b_f, a_f, _ = full_fits[i]
        b_r = w_self * b_r + (1.0 - w_self) * pooled
        if a_r is None:
            # Never earned in any state: a tiny rate, honestly uncertain.
            rows.append({"id": trophy["id"], "q": quantiles(0.01, TRAJ_SIGMA_CAP)})
            continue
        pred_recent = a_r + b_r * x_target
        pred_full = (a_f + b_f * x_target) if a_f is not None else pred_recent
        pred = 0.7 * pred_recent + 0.3 * pred_full
        last_lg = logit(rates_last) if rates_last is not None and rates_last > 0 else pred
        # Damp the move from the last observed rate for the extrapolated part.
        pred = last_lg + damp * (pred - last_lg)
        median = inv_logit(pred)
        sigma = math.sqrt(
            TRAJ_SIGMA_FLOOR ** 2
            + resid ** 2
            + (TRAJ_SIGMA_PER_SQRT_DAY * math.sqrt(horizon_days)) ** 2
        )
        sigma = min(TRAJ_SIGMA_CAP, max(0.2, sigma))
        rows.append({"id": trophy["id"], "q": quantiles(median, sigma)})
    return {"trophies": rows}


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    states = distinct_states(case)
    if len(states) <= 1:
        return flat_forecast(case, states)
    return trajectory_forecast(case, states)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
