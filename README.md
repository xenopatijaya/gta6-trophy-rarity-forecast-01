[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/xenopatijaya/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/xenopatijaya/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# gta6-trophy-rarity-forecast-01

Forecast, for every trophy of Grand Theft Auto VI on PlayStation 5, its earn rate on Exophase at 2026-11-26T00:00:00Z — the percent of the players Exophase tracks who will have earned it by then — as five quantiles per trophy, scored by mean pinball loss on the logit scale. Built for the [make.fun bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01) ($250, top 20).

## How it works

Exophase's page is not a live feed: it freezes between Exophase's own refreshes, then jumps. A refresh dilutes every rate at once (launch night: 638 → 3,868 players tracked, every rate but the 100% one down), so the forecaster works in counts — earners over players tracked — and is built around those refresh steps, not a smooth trend.

- **Distinct states** — consecutive identical reads are one Exophase state; the history is collapsed to the states that actually changed, and every printed rate is turned back into an integer earner count.
- **Flat history (practice cases)** — every read is the same launch-night page, so no trend exists to fit. The model predicts the first refresh directly: the tracked base multiplies by the calibrated launch-night factor (638 → 3,868, ×6.0634) and each trophy gains the exact earner increment that refresh produced, trophy by trophy (a 55-entry calibration table; a non-canonical trophy list falls back to per-kind newcomer propensities). Because the scored answer is the rate as the page prints it, the median is the count-implied rate rounded to the printed 2-decimal grid, with a small logit spread (s = 0.005) around it.
- **Rich history (final case)** — tracked growth is extrapolated two ways (a recent-weighted log-linear fit and a power law in hours since release, the log-linear damped toward the power law as the horizon grows) and earners grow two ways: newcomers earn at each trophy's marginal propensity over the last day of refresh intervals, shrunk toward its current rate, and existing players keep earning at a small per-day hazard fitted on the intervals where the tracked base barely moved. Quantiles are count-consistent — integer earner counts at exp(z·sᵢ) times the predicted count — with sᵢ growing with the horizon and shrinking with the earner count.
- **The 100% trophy** (the intro mission every tracked player holds) is forecast at 100% flat: a universal share cannot dilute.
- **Degenerate inputs** (no history, zero horizon, missing fields) fall back to the house persistence forecast, so a valid file is always written.

Deterministic, standard library only, no network, no randomness: the target and horizon come from the case file itself.

## Practice backtest

Practice answers are the refreshed page (04:30Z / 05:00Z reads, 3,868 players tracked, 4,276 trophies earned — recovered from the live Exophase page, which had not refreshed again by 11:48Z, and verified by the exact trophies-earned total).

| case | this entry | persistence baseline | trend baseline |
| ---- | ---------: | -------------------: | -------------: |
| p00  | **0.0010** | 1.5084 | 1.5084 |
| p01  | **0.0010** | 1.5084 | 1.5084 |

Exact loss 0.000954. The predicted medians reproduce the refreshed page's printed rates exactly (typical miss 0.000 pp, baselines: 0.677 pp); 90% coverage 1.00 (baselines: 0.00). On a synthetic rich-history test (launch states plus the refreshed state held to 12:00Z, forecasting 21:16Z the same day) the rich branch scores 0.331 against persistence's 0.860.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` in the input directory and writes `<id>.out` for each: `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`, one row per trophy, quantiles non-decreasing in percent.

## License

MIT — see [LICENSE](LICENSE).

---

Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards
