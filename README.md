[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/xenopatijaya/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/xenopatijaya/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# gta6-trophy-rarity-forecast-01

Forecast, for every trophy of Grand Theft Auto VI on PlayStation 5, its earn rate on Exophase at 2026-11-26T00:00:00Z — the percent of the players Exophase tracks who will have earned it by then — as five quantiles per trophy, scored by mean pinball loss on the logit scale. Built for the [make.fun bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01) ($250, top 20).

## How it works

Exophase's page is not a live feed: it freezes between Exophase's own refreshes, then jumps. A refresh dilutes every rate at once (launch night: 638 → 3,868 players tracked, every rate but the 100% one down), so the forecaster is built around those refresh steps, not a smooth trend.

- **Distinct snapshots** — consecutive identical reads are one Exophase state; the history is collapsed to the states that actually changed.
- **Flat history (practice cases)** — every read is the same launch-night page, so no trend exists to fit. The model expects the next refresh inside the horizon: the tracked base multiplies ~6× (launch-night prior) and the new players earn each trophy at a rate that depends on its kind — story missions almost none, Chapter 1 and the quick early activities (fishing challenge, arcade game, discovering vehicles) far more. The median is the old earners plus the newcomers' earners, over the new tracked base.
- **Rich history (final case)** — logit(rate) is fitted on log(1 + hours since release) over the last 24 hours of states, slope clipped and shrunk toward the pooled slope, blended with the full-history fit, then damped toward the last observed rate for the part of the horizon past the data.
- **The 100% trophy** (the intro mission every tracked player holds) is forecast at 100% flat.
- **Spread** — logit-normal quantiles; the width grows with the horizon and with each fit's residual, so a 4.5-day projection is honestly much wider than a 90-minute one.

Deterministic, standard library only, no network, no randomness: the target and horizon come from the case file itself.

## Practice backtest

Practice answers are the refreshed page (04:30Z / 05:00Z reads, 3,868 players tracked, 4,276 trophies earned — recovered from the live Exophase page, which had not refreshed again by 11:48Z, and verified by the exact trophies-earned total).

| case | this entry | persistence baseline | trend baseline |
| ---- | ---------: | -------------------: | -------------: |
| p00  | **0.0958** | 1.5084 | 1.5084 |
| p01  | **0.0958** | 1.5084 | 1.5084 |

Typical miss 0.032 pp (baselines: 0.677 pp), 90% coverage 0.96 (baselines: 0.00). Worst trophy is #20 (fishing challenge, loss 0.40); no catastrophic misses.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` in the input directory and writes `<id>.out` for each: `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`, one row per trophy, quantiles non-decreasing in percent.

## License

MIT — see [LICENSE](LICENSE).

---

Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards
