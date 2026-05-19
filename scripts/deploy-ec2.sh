#!/usr/bin/env bash
# Runs on EC2 after CI syncs code into EC2_DEPLOY_PATH (existing project folder).
set -euo pipefail

DEPLOY_PATH="${EC2_DEPLOY_PATH:?EC2_DEPLOY_PATH must be set to your existing app folder on EC2}"
cd "$DEPLOY_PATH"

echo "==> Deploying banking-rag in $DEPLOY_PATH (sha from CI sync)"

if [[ ! -f requirements.txt ]]; then
  echo "requirements.txt missing — ensure SCP target is EC2_DEPLOY_PATH (project root)."
  exit 1
fi

if ! command -v python3 &>/dev/null; then
  echo "python3 not found — install Python 3.11+ on the instance first."
  exit 1
fi

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

if [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then
  mkdir -p src
  printf 'ANTHROPIC_API_KEY=%s\n' "$ANTHROPIC_API_KEY" > src/.env
  chmod 600 src/.env
fi

if systemctl is-active --quiet banking-rag 2>/dev/null; then
  sudo systemctl restart banking-rag
  echo "==> Restarted banking-rag service"
else
  echo "==> No banking-rag systemd unit — deps updated; start the app manually if needed"
fi

echo "==> Deploy complete"
