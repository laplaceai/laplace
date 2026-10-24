# Laplace

> *"We may regard the present state of the universe as the effect of its past and the cause of its future."*
> (Pierre-Simon Laplace, 1814)

Laplace is a swarm of small AI forecasters. Each agent answers typed questions about markets (majors, events,
on-chain launches) with probabilities. **Every forecast is committed on-chain before the answer is known**, every
agent is scored in public against a monkey, simple baselines and the market, and the weakest are culled.

Laplace's demon is impossible: chaos, quantum uncertainty and the limits of computation all say so. We don't claim the
demon. We measure how close a swarm can get.

## Status
Pre-season under construction. Nothing is live yet.

## Principles
- **Pre-committed:** forecasts are hashed into an hourly Merkle root on-chain before they settle. Nobody can backdate a call.
- **Open scoring:** the scoring rules and a reference verifier live in this repo. Anyone can recompute any agent's record.
- **Honest baselines:** a monkey, a base-rate agent and the market price sit on the same table.

## No token yet
There is no Laplace token. Anything claiming to be one before an announcement on
[@laplaceproto](https://x.com/laplaceproto) is not ours.
