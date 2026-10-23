# laplace-research

Offline experiments that decide what the live pre-season runs: how forecast cards are built, which baselines an agent
has to beat, and how agents are scored. Nothing here runs in production.

```bash
cd research
uv sync
uv run pytest
```

Experiments need Workers AI credentials in the environment (`LAPLACE_AI_ACCOUNT_ID`, `LAPLACE_AI_TOKEN`). Results
are written under `runs/` (ignored by git) and summarized in `docs/research/`.
