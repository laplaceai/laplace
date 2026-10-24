# The evolution engine

Laplace searches for forecasting strategies instead of asking a model for forecasts. Language models write and rewrite
short programs; realized outcomes decide which programs survive. This note describes the research version in
`research/src/laplace_research/evolve/`.

## Why programs, not prompts
In PoC 1 (`docs/research/2026-10-23-poc1-majors.md`) language models turning numeric cards into probabilities lost to
a one-line volatility model, and different prompts on one model made almost the same errors. Models are better at
writing the code that does the statistics than at doing the statistics in their heads. Programs are also cheap to run,
deterministic, inspectable and easy to verify after the fact.

## The question
Every hour, for each of seven assets (BTC, ETH, SOL, XRP, DOGE, AVAX, LINK) and three horizons (1, 4 and 24 hours), a
program gives the probability of each of ten log-return bins. The bins are fixed in percent so thresholds read
directly (`feed.py`):

| horizon | inner edges (±%) |
|---|---|
| 1 hour | 0.3, 0.7, 1.5, 3 |
| 4 hours | 0.6, 1.4, 3, 6 |
| 24 hours | 1.5, 3.5, 7, 14 |

plus an edge at zero. The outcome is the bin of the realized log return between hourly closes.

## A program
```python
def forecast(ctx):
    ...  # return ten non-negative numbers, one per bin
```
`ctx` (`context.py`) holds the last 720 hourly candles of the asset and of every other asset, the hour and weekday,
the bin edges, and distribution helpers (`lib.py`: normal, Laplace, Student-t and mixtures, all parameterized by
standard deviation). The arrays are read-only views of the past; a program cannot see the future because the future is
not in memory.

## Safety
Programs come from language models, so they are treated as untrusted.
- **The future is structurally out of reach** (`history.py`, `evaluate.py`). Each evaluation runs in a freshly spawned
  worker that reads a time-major history file forward into a zero-filled buffer. When a program forecasts at hour `t`,
  its process has never loaded a candle after `t`, so nothing it can reach, a view's `.base` included, holds the
  future. A test rewrites every candle after a cutoff and checks that forecasts before it do not change.
- **Static checks** (`sandbox.py`) as defence in depth: imports limited to `math` and `numpy`, and a module may only
  be used as `module.attribute` (never bound to another name); attributes that reach files, raw memory, pickling or
  randomness are refused on any object; no private or dunder names, attribute assignment, dynamic code, classes or
  global state.
- Restricted builtins and a restricted `__import__` (numpy's own lazy internal imports pass; program source cannot
  ask for them).
- A CPU-time alarm and an address-space limit per worker; any exception, timeout, write attempt or malformed output
  rejects the program, and the reason is shown to the next mutation.

Research runs only execute programs proposed by our own mutation calls. Production runs will execute them in isolated
containers without network access.

## The loop
- **Islands** (`islands.py`): eight populations, each steered by a hypothesis about where predictability lives:
  volatility clustering, the clock (sessions, weekends), spillover between assets, tail shape, high-low ranges,
  trading activity, mean reversion across horizons, regimes.
- **Selection:** a parent is drawn from the island's best programs; sometimes a strong program from another island is
  shown as inspiration.
- **Mutation** (`mutate.py`): a model gets the interface, the rules, the hypothesis, the parent with its scores by
  horizon, any inspiration and the island's recent failures, and returns a title, a one-line rationale and a complete
  new program. Cheap models (GLM 5.3 Flash, DeepSeek V4.1 Flash) propose most children; a stronger reasoning model
  (GLM 5.3) proposes every fourth generation.
- **Evaluation:** every child runs on a search window of forecast times; its fitness is the mean log-likelihood gain
  over the naive baseline, in nats per question.
- **Record** (`db.py`): every program, failed or not, with its parent, mutator, cost, rationale and scores.

## Judging honestly
- **Proper scoring.** A program's output is normalized and mixed with the uniform distribution at weight 0.001 exactly
  once; that final forecast is what is scored, pooled and published (a mixture of final forecasts is never smoothed
  again). Fitness is the binned log score of final forecasts, which is strictly proper for them; for an arbitrary true
  distribution the smoothing costs at most 0.0009 nats per question. The ranked probability score is reported
  alongside.
- **Three purged windows.** Programs evolve on a search window (February 2025 to May 2026). Each island's champion is
  chosen on a validation window (June to mid-August 2026) from its three best search programs. Everything is then
  reported on a holdout window (mid-August 2026 onward) that played no part in any choice. A forecast belongs to a
  window only if its labels at every horizon settle before the window ends, so no window is fitted or selected on
  information from the next.
- **A baseline ladder** (`baselines.py`), all fitted on the search window only: naive (a Laplace distribution scaled by
  the last week's volatility, which is also the seed program), EWMA with Student-t tails, GARCH(1,1) with Student-t
  innovations fitted by maximum likelihood, and a HAR model of realized variance. Beating naive is easy; GARCH and HAR
  are the current bar, and stronger rungs (seasonal and range-informed models, implied volatility) come next.
- **Intervals** come from a circular block bootstrap over forecast issues: questions issued a few hours apart overlap,
  so blocks of consecutive issues (72 hours by default, 24 to 168 hours as a sensitivity check) are resampled, with
  the same resamples for every comparison.

## The pool
The engine's product is not the single best program but the pool of island champions (`pool.py`, `report.py`): a
linear pool whose weights are fitted by EM on validation until a certified optimality gap (at most 1e-8 nats per
question below the best possible weights).

The report also lists each member's renormalized leave-one-out contribution: the pool's score minus its score with
that member removed and the remaining weights renormalized. It shows how much the fitted pool leans on each member,
but it is a diagnostic, not a payout rule: members whose errors cancel each other can both look valuable, and a copy
can keep positive credit. A contribution measure fit for rewards needs separately refitted counterfactual pools and
explicit rules for copies, and is still being designed.

## Running it
```bash
cd research
uv run laplace-research evolve init --run-id NAME      # builds the panel and seeds the islands
uv run laplace-research evolve run --run-id NAME --generations 24 --children 2
uv run laplace-research evolve report --run-id NAME    # champions, pools and baselines on the holdout window
```
Mutation calls need Workers AI credentials in the environment (see `inference.py`).
