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
Programs come from language models, so they are treated as untrusted (`sandbox.py`, `evaluate.py`):
- an AST allow-list: imports limited to `math` and `numpy`; no private attributes, dynamic code, I/O, classes, global
  state or randomness;
- restricted builtins and a restricted `__import__`;
- execution in separate worker processes with a CPU-time alarm and an address-space limit;
- any exception, timeout, write attempt or malformed output rejects the program, and the reason is shown to the next
  mutation.

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
- **Proper scoring.** Fitness is the binned log score, which rewards exactly the calibrated probabilities. Every
  forecast, baselines included, is mixed with the uniform distribution at weight 0.001 before scoring, which bounds the
  penalty for a surprise. The ranked probability score is reported alongside.
- **Three windows.** Programs evolve on a search window (February 2025 to May 2026). Each island's champion is chosen
  on a validation window (June to mid-August 2026) from its three best search programs. Everything is then reported on
  a holdout window (mid-August 2026 onward) that played no part in any choice.
- **A baseline ladder** (`baselines.py`), all fitted on the search window only: naive (a Laplace distribution scaled by
  the last week's volatility, which is also the seed program), EWMA with Student-t tails, GARCH(1,1) with Student-t
  innovations fitted by maximum likelihood, and a HAR model of realized variance. Beating naive is easy; GARCH and HAR
  are the real bar.
- **Intervals** come from a bootstrap over whole days, because questions issued a few hours apart overlap.

## The pool
The engine's product is not the single best program but the pool of island champions (`report.py`): a linear pool
whose weights are fitted by EM on validation. A member's contribution is the pool's score minus its score without
that member (weights renormalized, not refitted). Copies contribute nothing, so contribution rewards diversity that
is also right.

## Running it
```bash
cd research
uv run laplace-research evolve init --run-id NAME      # builds the panel and seeds the islands
uv run laplace-research evolve run --run-id NAME --generations 24 --children 2
uv run laplace-research evolve report --run-id NAME    # champions, pools and baselines on the holdout window
```
Mutation calls need Workers AI credentials in the environment (see `inference.py`).
