# PoC 2: evolving forecasting programs

2026-10-24 · run `evolve-poc2` · engine described in `docs/design/evolution-engine.md` · reproduce with
`uv run laplace-research evolve init|run|report --run-id poc2`

## Setup
- **Data:** hourly Coinbase candles for BTC, ETH, SOL, XRP, DOGE, AVAX and LINK from January 2025 to 24 October 2026
  (15,872 hours; 10 hours of exchange outage filled flat).
- **Questions:** at each sampled hour, ten-bin distributions of the log return over 1, 4 and 24 hours for every asset
  (21 questions per forecast time).
- **Windows:** search February 2025 to May 2026 (every 23rd hour), validation June to 15 August 2026 (every 5th hour),
  holdout 16 August to 24 October 2026 (every 3rd hour: 547 forecast times over 69 days, 11,487 questions).
- **Population:** 8 islands, each seeded with the same one-line naive program and steered by its own hypothesis;
  25 generations, 2 children per island per generation; 400 programs in total.
- **Cost:** $2.35 of mutation calls (GLM 5.3 Flash, DeepSeek V4.1 Flash, GLM 5.3), about 75 minutes on 8 cores.

## Results on the holdout window
Gains are mean log-likelihood gains in nats per question; exp(gain) − 1 is the average extra probability a forecaster
put on what actually happened. Champions were chosen on validation; the holdout played no part in any choice.

| forecaster | vs naive | vs GARCH (95% CI) | vs HAR (95% CI) | 1h | 4h | 24h |
|---|---:|---|---|---:|---:|---:|
| **pool of evolved champions** | **+0.0427** | +0.0443 (+0.0170…+0.0697) | **+0.0158 (+0.0002…+0.0306)** | +0.0384 | +0.0385 | +0.0513 |
| best champion (tails island) | +0.0414 | +0.0429 (+0.0208…+0.0663) | +0.0145 (+0.0002…+0.0292) | +0.0324 | +0.0345 | +0.0572 |
| champion chosen on validation (spillover) | +0.0345 | +0.0360 (+0.0029…+0.0657) | +0.0076 (−0.0146…+0.0274) | +0.0369 | +0.0332 | +0.0334 |
| HAR-RV | +0.0269 | +0.0284 (+0.0152…+0.0414) | — | +0.0172 | +0.0164 | +0.0470 |
| EWMA, Student-t | +0.0092 | +0.0108 | −0.0176 | +0.0099 | +0.0066 | +0.0111 |
| naive (the seed) | 0 | | −0.0269 | 0 | 0 | 0 |
| GARCH(1,1), Student-t | −0.0016 | — | −0.0284 | −0.0220 | −0.0142 | +0.0315 |

1. **The pool beats both classical baselines out of sample.** Against GARCH clearly; against HAR by +0.016 nats per
   question (1.6% more probability on the outcome), with the 95% interval just clear of zero. Six of eight champions
   beat HAR on the holdout, one of them significantly.
2. **The gain is at short horizons.** At 1 and 4 hours the pool more than doubles HAR's gain over naive; at 24 hours
   they are close. That is where intraday structure lives.
3. **The pool beats the single champion we would have picked.** On validation the best single champion was the
   spillover island's; on the holdout the pool beat it by 0.008 nats per question.
4. **Classical models add nothing to the pool.** Fitted on validation, the pool gives HAR, EWMA and GARCH a combined
   weight of 0.004; the evolved champions already contain what they know.
5. **No sign of overfitting yet.** Champions score about the same on the holdout as on the search window (+0.02 to
   +0.04). One holdout window is one regime, though; see the caveats.
6. **Contribution is noisy at this size.** The rank correlation of champions' pool contributions between validation
   and holdout was 0.36 (8 members). Rewards based on contribution will need longer windows or a lower-variance rule.

## What evolved
Every champion descends from the same naive seed through 7 to 14 mutations and is 180 to 300 lines long. Read in
order, the titles in each lineage show the search rediscovering standard volatility modelling and then combining it
in ways no single textbook model does:
- **an hour-of-day and weekday volatility profile pooled across all seven assets**, with a separate weekend profile
  because session spikes vanish on Saturdays and Sundays (seasonality island);
- **high-low range estimators** (Parkinson, Garman-Klass, Rogers-Satchell, a Yang-Zhang-style gap term) blended with
  close-to-close returns (ranges, tails);
- **fast/slow volatility blends whose weights depend on the horizon**, and GARCH-style aggregation over the horizon
  (clustering, horizons);
- **tail fatness estimated from recent kurtosis**, regime-dependent tail shapes, volume shocks that fatten tails rather
  than only scale (tails, regimes, activity);
- **market-wide shocks led by BTC and ETH**, deseasonalized before pooling (spillover).

## Mutators
| mutator | calls | valid | improved on parent | cost | $ per improvement |
|---|---:|---:|---:|---:|---:|
| GLM 5.3 Flash | 172 | 88% | 42% | $0.34 | $0.005 |
| DeepSeek V4.1 Flash | 172 | 48% | 27% | $0.61 | $0.013 |
| GLM 5.3 (high reasoning) | 48 | 88% | 35% | $1.40 | $0.082 |

GLM 5.3 Flash was the most efficient by far. Most DeepSeek failures were infrastructure, not bad programs: its token
budget ran out while reasoning once prompts grew (the budget is now doubled). All three models appear in every
champion's lineage.

## Caveats
- **One holdout window** of 69 days. A rolling-origin evaluation over several windows is the next check.
- **The HAR baseline is a plain one** (levels OLS, no seasonality). A seasonal HAR and the options market's implied
  volatility for BTC and ETH are the next rungs of the ladder.
- **Multiple comparisons:** the pool was the pre-declared product; the individual champion intervals are not
  corrected for the eight comparisons.
- **The mutating models know published volatility research.** That is legitimate prior knowledge, not leakage: the
  programs never see dates, and every judgement is on data after the search window.

## Next
Stronger baselines (seasonal HAR, implied volatility), rolling-origin holdouts, a longer run with more islands, and the
live loop: forecasts committed on-chain every hour, so the record that counts starts accumulating.
