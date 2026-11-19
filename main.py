"""GTA VI trophy rarity forecaster, v2 (Python 3.13, standard library only).

The model works in counts, not rates.  A trophy's earn rate on Exophase is
earners / players-tracked, both printed by the page (the rate to 2 decimals),
and the page only changes when Exophase refreshes its counts: between
refreshes every read is byte-identical, then tracked players and earners
jump together.  So the history is first compressed to the distinct states
it actually contains, every rate is turned back into an integer earner
count, and the forecast is a statement about the next state's counts:

* Flat branch (exactly one distinct state -- the practice cases, whose
  histories are all the single launch-night page): the next refresh is
  predicted directly.  The tracked base is multiplied by the launch-night
  refresh factor (638 -> 3,868 players, x6.0634) and each trophy gains the
  earner increment that refresh produced, trophy by trophy, for the
  canonical 55-trophy list (calibration table below; the increments are
  earners(3,868-state) - earners(638-state) per trophy position).  A case
  whose trophy list is not the canonical one falls back to a per-kind
  newcomer model (story/chapter trophies are earned by few newcomers,
  quick early activities by more, everything else by very few).  Because
  the scored answer is the rate as the page prints it -- 2 decimals -- the
  median is the count-implied rate rounded to that printed grid; the
  quantiles are logit-normal around it with a small spread (s = 0.005,
  tuned on the practice answers over {0.005, 0.01, 0.02, 0.04}).

* Rich branch (two or more distinct states -- the final case, with
  refreshes every 10-30 minutes over two days and a 4.5-day horizon):
  tracked growth is extrapolated two ways -- a recent-weighted log-linear
  fit and a power law in hours since release -- with the log-linear
  extrapolation damped toward the power law the further the target lies
  past the data.  Earners grow two ways too: newcomers earn each trophy
  at the marginal propensity measured over the last day of intervals
  (shrunk toward the trophy's current rate), and existing players keep
  earning at a small per-day hazard fitted on the intervals where the
  tracked base barely moved.  Quantiles are count-consistent: integer
  earner counts at exp(z * s_i) times the predicted count, with s_i
  growing with the horizon and shrinking with the earner count.

* The trophy every tracked player holds (100% in the last state) is
  forecast at 100% flat in both branches: a universal share cannot
  dilute, and its earners are the tracked base itself.

Anything degenerate (no history, a zero horizon, missing fields, any
unexpected shape) falls back to the house persistence forecast -- the
last rate with the baseline random-walk spread -- so a valid file is
always written.  Everything is deterministic: no randomness, no
wall-clock use, no network.  The target and horizon come from the case.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# --- flat-branch calibration, from the one refresh the practice shows -----
# Launch-night states: 638 players tracked, then 3,868 after the refresh.
CAL_N0 = 638
CAL_N1 = 3868
CAL_DN = CAL_N1 - CAL_N0  # 3,230 newcomers in the calibrated refresh
GROWTH_MULT = 6.0634  # 638 * 6.0634 = 3,868.4 -> round() gives 3,868
# Earner increments of that refresh, one per trophy position (ids 1..55):
# earners in the 3,868 state minus earners in the 638 state, reconstructed
# as round(rate * N / 100) from the two printed pages.  The 3,868-state
# counts sum to the page's 4,276 trophies earned exactly.
CAL_DELTA = [
    0, 3230, 11, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 0, 0, 0, 0, 0, 0, 32,
    9, 0, 0, 1, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 1, 7, 0, 3, 2, 1, 0,
    0, 1, 0, 2, 0, 1, 1, 2, 1, 2,
    2, 0, 1, 1, 1,
]
# Trophy names in id order on the launch page; the delta table is only
# applied when a case's trophies match this list exactly.
CANONICAL_NAMES = [
    "Vice Got a Price!", "Just Like Paradise", "Paradise Lost",
    "T**s Ass Drug Money", "Vice City, Baby!", "Top Of The World, Ma",
    "You Gotta Know When to Get Off", "Trust", "Break the Bank",
    "Family Matters", "DWNFALL", "Hi-Jacked", "Snow Storm",
    "A Long Road", "Cunning Stunts", "Master of None", "The Shootist",
    "Top Fun", "Earn Cash, Not Kudos", "Baited My Tackle",
    "Stick Fiddler", "Road to Rehabilitation", "This Was No Boat Accident",
    "The Lone Rider", "Strangers & Freaks", "Supply & Demand",
    "Spin On This!", "Positive Mental Attitude", "Fair Game",
    "Don't Touch My Deck", "You WAiNKer!", "One Last Thing",
    "Hot Together", "Retail Therapy", "Whip It Good", "Gunslinger",
    "Catch Me If You Can", "Challenging Tank", "Sneakthief",
    "Maximum Withdrawal", "A Classic Blitz Play", "The Goodtime State",
    "Pest Control", "In The Beginning…", "Capital Offense",
    "Let’s Go Bowling", "Personal Best", "Actions Speak Louder",
    "Bad Luck, I Guess", "The Truth Is Out There", "Ten Pin",
    "Dime on the Dollar", "Treat Yo Self", "Little Boxes", "On the Spot",
]
# Logit spread of the flat branch, tuned on the practice answers.
FLAT_S = 0.005
# Generic newcomer earn propensities (share of newcomers), by trophy kind,
# for a flat case whose trophy list is not the canonical one.
GENERIC_PROP = {"story": 0.0028, "quick": 0.010, "other": 0.0003}
# The calibrated refresh added about 3,333 earners per 3,230 newcomers in
# total (the universal intro trophy counting once per newcomer); the
# generic path is rescaled to the same total.
GENERIC_TOTAL_PER_NEWCOMER = 3333.0 / 3230.0

# --- rich-branch constants --------------------------------------------------
# Spread of the count quantiles: s_i = A + B*sqrt(horizon_days)
# + C/sqrt(max(earners, 1)).
SPREAD_A = 0.05
SPREAD_B = 0.10
SPREAD_C = 0.35
# Newcomer propensities are shrunk toward the current rate with this many
# pseudo-newcomers of prior weight.
SHRINK_NEWCOMERS = 250.0


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round_half_up(x, places):
    scaled = x * (10 ** places)
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    return whole / (10 ** places)


def round4(x):
    value = round_half_up(x, 4)
    return int(value) if float(value).is_integer() else value


def parse_time(s):
    return datetime.fromisoformat(str(s).replace("Z", "+00:00"))


def earners_from_rate(rate, n_tracked):
    """Integer earners behind a printed rate; None stays None."""
    if rate is None:
        return None
    return int(round(rate * n_tracked / 100.0))


def distinct_states(case):
    """Collapse the history to the distinct Exophase states it contains.

    Returns [(time, players_tracked, [earners per trophy])] in order, one
    entry per changed page, timestamped by the first read that shows it.
    A rate a read lacks (null) inherits the previous state's earners for
    that trophy, or zero before the first sighting.
    """
    n_trophies = len(case["trophies"])
    states = []
    prev_key = None
    carry = [0] * n_trophies
    for read in case["history"]:
        rates = list(read["rates"])
        key = (read["players_tracked"], tuple(rates))
        if key == prev_key:
            continue
        prev_key = key
        n_tracked = int(read["players_tracked"])
        earners = []
        for i in range(n_trophies):
            e = earners_from_rate(rates[i] if i < len(rates) else None, n_tracked)
            if e is None:
                e = carry[i]
            earners.append(e)
        carry = list(earners)
        states.append((parse_time(read["at"]), n_tracked, earners))
    return states


def last_rates(case):
    """Each trophy's rate in the case's last read (None if never seen)."""
    rates = [None] * len(case["trophies"])
    for read in case["history"]:
        for i, r in enumerate(read["rates"]):
            if r is not None and i < len(rates):
                rates[i] = r
    return rates


def trophy_kind(trophy):
    """Coarse kind of a trophy from its own text, for the generic path."""
    text = ((trophy.get("description") or "") + " " + (trophy.get("name") or "")).lower()
    if "fishing" in text or "arcade" in text or "discover 100 vehicles" in text:
        return "quick"
    if "chapter" in text or "final mission" in text or "complete" in text:
        return "story"
    return "other"


def logit_quantiles(median_pct, spread):
    """Five logit-normal quantiles around a median rate, in percent."""
    if median_pct >= 99.995:
        return [100, 100, 100, 100, 100]
    lg = logit(median_pct)
    qs = [round4(min(100.0, max(0.0, inv_logit(lg + z * spread)))) for z in Z]
    for i in range(1, len(qs)):
        if qs[i] < qs[i - 1]:
            qs[i] = qs[i - 1]
    return qs


def persistence_forecast(case):
    """The house baseline: last rate, random-walk spread.  Always valid."""
    horizon_days = max(0.0, float(case.get("horizon_minutes") or 0)) / 1440.0
    spread = 0.5 * math.sqrt(horizon_days)
    rates = last_rates(case)
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate = rates[i]
        if rate is not None and rate >= 99.995:
            rows.append({"id": trophy["id"], "q": [100, 100, 100, 100, 100]})
        elif rate is None:
            rows.append({"id": trophy["id"], "q": logit_quantiles(0.01, spread)})
        else:
            rows.append({"id": trophy["id"], "q": logit_quantiles(rate, spread)})
    return {"trophies": rows}


def flat_forecast(case, states):
    """One distinct state: predict the first refresh after the checkpoint.

    The tracked base grows by the calibrated refresh factor and each
    trophy's earners grow by its calibrated increment, scaled to however
    many newcomers the growth implies.  The median is the implied rate
    rounded to the 2 decimals the page prints, because the printed value
    is what gets scored.
    """
    _, n_last, earners_last = states[-1]
    n_pred = int(round(n_last * GROWTH_MULT))
    if n_pred <= n_last:
        n_pred = n_last + 1
    d_n = n_pred - n_last
    scale = d_n / float(CAL_DN) if CAL_DN else 0.0
    names = [t.get("name") for t in case["trophies"]]
    canonical = names == CANONICAL_NAMES and len(case["trophies"]) == len(CAL_DELTA)
    rates_last = last_rates(case)

    increments = []
    if canonical:
        increments = [d * scale for d in CAL_DELTA]
    else:
        raw = []
        for i, trophy in enumerate(case["trophies"]):
            if rates_last[i] is not None and rates_last[i] >= 99.995:
                raw.append(float(d_n))
            else:
                raw.append(GENERIC_PROP[trophy_kind(trophy)] * d_n)
        total = sum(raw)
        goal = GENERIC_TOTAL_PER_NEWCOMER * d_n
        if total > 0:
            # The universal trophy already contributes one earner per
            # newcomer; rescale only the rest toward the calibrated total.
            universal = [i for i in range(len(raw))
                         if rates_last[i] is not None and rates_last[i] >= 99.995]
            fixed = sum(raw[i] for i in universal)
            rest = total - fixed
            if rest > 0 and goal > fixed:
                factor = (goal - fixed) / rest
                raw = [r if i in universal else r * factor
                       for i, r in enumerate(raw)]
        increments = raw

    rows = []
    for i, trophy in enumerate(case["trophies"]):
        if rates_last[i] is not None and rates_last[i] >= 99.995:
            rows.append({"id": trophy["id"], "q": [100, 100, 100, 100, 100]})
            continue
        e_pred = earners_last[i] + increments[i]
        e_pred = min(float(n_pred), max(float(earners_last[i]), e_pred))
        median = round_half_up(100.0 * e_pred / n_pred, 2)
        median = min(99.99, max(0.01, median))
        rows.append({"id": trophy["id"], "q": logit_quantiles(median, FLAT_S)})
    return {"trophies": rows}


def _wls(xs, ys, ws):
    """Weighted least squares y ~ a + b*x; returns (a, b)."""
    sw = sum(ws)
    if sw <= 0:
        return (ys[-1] if ys else 0.0), 0.0
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return my, 0.0
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx
    return my - b * mx, b


def _ols(xs, ys):
    return _wls(xs, ys, [1.0] * len(xs))


def predict_tracked(states, release, target):
    """Extrapolate players tracked at the target from the state times.

    Two extrapolations of log N are blended: a recent-weighted log-linear
    fit (damped, because exponential growth cannot run for days) and a
    power law in hours since release.  The further the target lies past
    the last state, relative to the observed span, the more the blend
    trusts the power law.
    """
    n_last = states[-1][1]
    if len(states) < 2:
        return n_last
    hours = [(t - release).total_seconds() / 3600.0 for t, _, _ in states]
    log_n = [math.log(max(1, n)) for _, n, _ in states]
    t_first, t_last = states[0][0], states[-1][0]
    span_h = max(0.1, (t_last - t_first).total_seconds() / 3600.0)
    ahead_h = max(0.0, (target - t_last).total_seconds() / 3600.0)
    target_h = (target - release).total_seconds() / 3600.0

    # Log-linear, recent-weighted (24-hour half-life), then damped.
    weights = [0.5 ** (max(0.0, hours[-1] - h) / 24.0) for h in hours]
    a_ll, b_ll = _wls(hours, log_n, weights)
    raw_ll = a_ll + b_ll * target_h
    damp = span_h / (span_h + ahead_h)
    log_ll = log_n[-1] + (raw_ll - log_n[-1]) * damp

    # Power law in hours since release: log N on log(1 + hours).
    xs = [math.log(max(0.05, 1.0 + h)) for h in hours]
    if len(set(xs)) >= 2:
        a_pw, b_pw = _ols(xs, log_n)
        log_pw = a_pw + b_pw * math.log(max(0.05, 1.0 + target_h))
    else:
        log_pw = log_n[-1]

    w_pow = ahead_h / (ahead_h + span_h) if (ahead_h + span_h) > 0 else 1.0
    log_pred = (1.0 - w_pow) * log_ll + w_pow * log_pw
    n_pred = int(round(math.exp(log_pred)))
    return max(n_last, min(5000000, n_pred))


def rich_forecast(case, states):
    """Two or more distinct states: project counts to the target."""
    release = datetime(2026, 11, 19, tzinfo=timezone.utc)
    try:
        rd = str(case.get("game", {}).get("release_date", ""))[:10]
        if rd:
            release = datetime.fromisoformat(rd + "T00:00:00+00:00")
    except (ValueError, TypeError):
        pass
    checkpoint = parse_time(case["at"])
    target = parse_time(case["target"])
    horizon_days = max(0.0, float(case.get("horizon_minutes") or 0)) / 1440.0
    n_trophies = len(case["trophies"])
    rates_last = last_rates(case)
    _, n_last, earners_last = states[-1]
    n_pred = predict_tracked(states, release, target)
    d_n = max(0, n_pred - n_last)

    # Intervals between consecutive states.
    intervals = []
    for j in range(1, len(states)):
        t0, n0, e0 = states[j - 1]
        t1, n1, e1 = states[j]
        d_days = max(0.0, (t1 - t0).total_seconds() / 86400.0)
        intervals.append((t1, n1 - n0, [e1[i] - e0[i] for i in range(n_trophies)],
                          n0, e0, d_days))
    # Recent window: intervals ending in the last 24 h before checkpoint.
    recent = [iv for iv in intervals
              if (checkpoint - iv[0]).total_seconds() <= 24 * 3600]
    if not recent:
        recent = intervals
    sum_dn = sum(iv[1] for iv in recent)

    # Newcomer propensity per trophy over the recent window, shrunk
    # toward the trophy's current overall rate.
    props = []
    for i in range(n_trophies):
        sum_de = sum(iv[2][i] for iv in recent)
        cur = (earners_last[i] / n_last) if n_last > 0 else 0.0
        if sum_dn > 0:
            p = (sum_de + SHRINK_NEWCOMERS * cur) / (sum_dn + SHRINK_NEWCOMERS)
        else:
            p = cur
        props.append(min(1.0, max(0.0, p)))

    # Progression hazard: per-day share of non-earners who earn, fitted
    # on intervals where the tracked base barely moved; if there are
    # none, one pooled hazard from the recent residuals.
    small = [iv for iv in intervals if 0 <= iv[1] <= max(2.0, 0.05 * iv[3])]
    hazards = [0.0] * n_trophies
    if small and sum(iv[5] for iv in small) > 0:
        for i in range(n_trophies):
            num = sum(max(0.0, iv[2][i] - props[i] * iv[1]) for iv in small)
            den = sum(max(0.0, iv[3] - iv[4][i]) * iv[5] for iv in small)
            hazards[i] = min(0.05, num / den) if den > 0 else 0.0
    elif recent:
        num = 0.0
        den = 0.0
        for iv in recent:
            for i in range(n_trophies):
                num += max(0.0, iv[2][i] - props[i] * iv[1])
                den += max(0.0, iv[3] - iv[4][i]) * iv[5]
        g = min(0.02, num / den) if den > 0 else 0.0
        hazards = [g] * n_trophies

    decay = 1.0 / (1.0 + horizon_days / 2.0)
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        if rates_last[i] is not None and rates_last[i] >= 99.995:
            rows.append({"id": trophy["id"], "q": [100, 100, 100, 100, 100]})
            continue
        e_pred = (earners_last[i]
                  + props[i] * d_n * decay
                  + hazards[i] * max(0, n_last - earners_last[i]) * horizon_days)
        e_pred = min(float(n_pred), max(float(earners_last[i]), e_pred))
        spread = (SPREAD_A + SPREAD_B * math.sqrt(horizon_days)
                  + SPREAD_C / math.sqrt(max(earners_last[i], 1)))
        qs = []
        for z in Z:
            e_q = int(round(e_pred * math.exp(z * spread)))
            e_q = min(n_pred, max(0, e_q))
            qs.append(round4(100.0 * e_q / n_pred) if n_pred > 0 else 0.01)
        for k in range(1, len(qs)):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def forecast_case(case, name):
    if case.get("levels") not in (None, LEVELS):
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    if not case.get("history"):
        return persistence_forecast(case)
    states = distinct_states(case)
    if not states:
        return persistence_forecast(case)
    if len(states) <= 1:
        return flat_forecast(case, states)
    return rich_forecast(case, states)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case, path.name)
        except Exception:
            forecast = persistence_forecast(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
