#!/usr/bin/env bash
# Deploy the engine at a git ref (default: origin/main) into its own worktree and (re)install the hourly timer.
# Development happens in the main checkout; the timer only ever runs what was deployed here.
set -euo pipefail

repo="${LAPLACE_REPO:-$HOME/laplace/laplace}"
deploy="${LAPLACE_DEPLOY:-$HOME/laplace/deploy}"
ref="${1:-origin/main}"

git -C "$repo" fetch --quiet origin
if [ ! -e "$deploy/.git" ]; then
  git -C "$repo" worktree add --quiet --detach "$deploy" "$ref"
fi
git -C "$deploy" checkout --quiet --detach "$ref"
(cd "$deploy/engine" && uv sync --quiet --frozen --no-dev)

units="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
mkdir -p "$units"
install -m 0644 "$deploy/engine/deploy/laplace-issue.service" "$deploy/engine/deploy/laplace-issue.timer" "$units/"
systemctl --user daemon-reload
systemctl --user enable --quiet --now laplace-issue.timer

echo "deployed $(git -C "$deploy" rev-parse --short HEAD)"
systemctl --user list-timers laplace-issue.timer --no-pager
