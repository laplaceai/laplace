# laplace-research

Offline experiments that decide what the live pre-season runs: how forecast cards are built, which baselines an agent
has to beat, and how agents are scored. Nothing here runs in production.

```bash
cd research
uv sync
uv run pytest
```

Experiments need Workers AI credentials in the environment: `LAPLACE_AI_ACCOUNT_ID` and `LAPLACE_AI_TOKEN`. To route
calls through an authenticated AI Gateway (logs, cost per request, rate and spend limits), also set
`LAPLACE_AI_GATEWAY` and `LAPLACE_AI_GATEWAY_TOKEN`; evolution runs then tag every mutation with its run, island,
generation and mutator. See `src/laplace_research/inference.py`. Results are written under `runs/` (ignored by git)
and summarized in `docs/research/`.
