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
- **Cost:** $2.35 of recorded mutation calls (GLM 5.3 Flash, DeepSeek V4.1 Flash, GLM 5.3), about 75 minutes on 8
  cores. 86 replies without a usable program were not recorded; including them, about $2.9–3.5 (see Corrections).

## Corrections (re-run after an independent review)
An independent review of the engine found problems in how PoC 2 was measured. Each was reproduced, fixed with
regression tests, and the report was re-run on the same programs and windows. The programs themselves are unchanged.
1. **Future candles were reachable.** Contexts were views into the full price panel, and a view's `.base` led to
   candles after the forecast time. None of the 284 valid programs used it, but the judge must not depend on that.
   Programs now run in fresh processes that have never loaded a candle after the forecast time.
2. **The pool was smoothed twice when scored**, which flattered it by about 0.0006 nats per question, **and its
   weights were not fully fitted** (EM stopped after 500 iterations). Every forecast is now finalized once, and pool
   weights are fitted to a certified optimality gap of 1e-8 nats.
3. **Window boundaries leaked.** A few labels at the end of the search and validation windows settled after the next
   window's first forecast. Windows are now purged so every label settles before the next window starts.
4. **Intervals were too narrow.** Resampling independent calendar days cuts through 24-hour outcomes that overlap
   midnight. Intervals now come from a circular block bootstrap over forecast issues (72-hour blocks, with 24 to
   168 hours as a sensitivity check), each issue weighted equally.
5. **Cost was under-recorded.** 86 replies that contained no usable program were stored at $0. Bounded by their token
   caps, the run cost about $2.9–3.5 in total, not $2.35.

Re-running also exposed a sandbox defect: numpy's C code imports some of its own submodules lazily, and inside a fresh
worker those imports were refused. One PoC 2 child failed this way during the run and was discarded; the defect is
fixed.

### Corrected results on the holdout window
| forecaster | vs naive | vs GARCH (95% CI) | vs HAR (95% CI) | 1h | 4h | 24h |
|---|---:|---|---|---:|---:|---:|
| **pool of evolved champions** | **+0.0411** | +0.0426 (+0.0095…+0.0723) | **+0.0143 (−0.0069…+0.0331)** | +0.0377 | +0.0376 | +0.0478 |
| best champion in hindsight (tails island) | +0.0414 | +0.0429 (+0.0185…+0.0678) | +0.0146 (−0.0017…+0.0311) | +0.0324 | +0.0345 | +0.0572 |
| champion chosen on validation (spillover) | +0.0345 | +0.0360 (−0.0030…+0.0707) | +0.0077 (−0.0207…+0.0317) | +0.0369 | +0.0332 | +0.0334 |
| HAR-RV | +0.0268 | +0.0283 (+0.0143…+0.0422) | — | +0.0171 | +0.0163 | +0.0469 |
| EWMA, Student-t | +0.0092 | +0.0108 | −0.0176 | +0.0099 | +0.0066 | +0.0111 |
| naive (the seed) | 0 | | −0.0268 | 0 | 0 | 0 |
| GARCH(1,1), Student-t | −0.0016 | — | −0.0283 | −0.0220 | −0.0142 | +0.0315 |

| pool of evolved champions | mean | 24h blocks | 48h blocks | 72h blocks | 168h blocks |
|---|---:|---|---|---|---|
| vs GARCH | +0.0426 | +0.0154…+0.0692 | +0.0099…+0.0723 | +0.0095…+0.0723 | +0.0143…+0.0673 |
| vs HAR | +0.0143 | −0.0025…+0.0307 | −0.0061…+0.0321 | −0.0069…+0.0331 | −0.0049…+0.0302 |

What the corrected evidence supports:
1. **The pool beats GARCH**, with intervals clear of zero at every block length.
2. **Against HAR the estimate is positive but unresolved.** +0.014 nats per question, and every interval includes
   zero. The original claim that the interval was "just clear of zero" is withdrawn.
3. **The pool beats the champion we would have picked** on validation by +0.0066 nats per question. The best champion
   in hindsight matches the pool, but could not have been picked in advance.
4. **Classical models get zero weight** in a fully fitted pool of champions and classical models.
5. **Contribution ranks are noisy:** their rank correlation between validation and holdout is 0.48 for eight members.
6. **Reading the numbers:** exp(gain) − 1 is a geometric-mean relative likelihood per question, not an average
   increase in the probability assigned. (The original text below says the latter.)
7. **How long resolution takes:** the daily mean of the pool-minus-HAR gain has a standard deviation of about 0.068.
   Even if the true edge were as large as observed, confirming it at 80% power would take roughly six months of
   daily data; an edge of 0.005 nats per question would take about four years (more once autocorrelation is
   counted). Seven live days can show that the system works and is honest; thirty can give a direction, not a
   verdict.

This holdout has now been inspected repeatedly, so it serves as development evidence. Stronger rungs on the baseline
ladder (seasonal HAR, a range-informed volatility model, hour- and weekday-conditioned bin frequencies, filtered
historical simulation, implied volatility for BTC and ETH) and fresh, later windows come before any stronger claim.

## Original results (24 October 2026, superseded)
*Kept unchanged for the record. These numbers were produced by the original evaluator; see the corrections above.*

### Results on the holdout window (original)
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
*The programs are unchanged by the corrections.*

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
Stronger baselines (see Corrections), fresh evaluation windows, a matched-budget comparison of hypothesis-steered
islands against generic ones, a longer run, and the live loop: forecasts committed on-chain every hour, so the record
that counts starts accumulating.
