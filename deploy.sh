#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration — override via environment variables
# ---------------------------------------------------------------------------
REMOTE_HOST="${REMOTE_HOST:-158.160.4.48}"
REMOTE_PORT="${REMOTE_PORT:-22}"
REMOTE_USER="${REMOTE_USER:-misha-sh}"
SSH_KEY="${SSH_KEY:-$HOME/id_ed25519}"
REMOTE_DIR="${REMOTE_DIR:-/home/misha-sh/4_website}"

SSH_OPTS="-i $SSH_KEY -p $REMOTE_PORT -o StrictHostKeyChecking=accept-new -o BatchMode=yes"

# ---------------------------------------------------------------------------
# CLI flags
# ---------------------------------------------------------------------------
SKIP_BOOTSTRAP=0
SKIP_DATA=0
for arg in "$@"; do
  case "$arg" in
    --skip-bootstrap) SKIP_BOOTSTRAP=1 ;;
    --skip-data)      SKIP_DATA=1 ;;
    *) echo "Unknown flag: $arg (valid: --skip-bootstrap, --skip-data)"; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
log() { echo; echo "==> $*"; }

ssh_run() {
  ssh $SSH_OPTS "${REMOTE_USER}@${REMOTE_HOST}" "$@"
}

rsync_to() {
  local src="$1"; shift
  rsync -az --info=progress2 -e "ssh $SSH_OPTS" "$@" \
    "${SCRIPT_DIR}/${src}" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/${src}"
}

# ---------------------------------------------------------------------------
# 1. Preflight (local)
# ---------------------------------------------------------------------------
log "Preflight checks"

if [[ ! -f "$SSH_KEY" ]]; then
  echo "ERROR: SSH key not found: $SSH_KEY"; exit 1
fi
if [[ "$(stat -c %a "$SSH_KEY")" != "600" ]]; then
  echo "Fixing SSH key permissions (was $(stat -c %a "$SSH_KEY"), setting to 600)..."
  chmod 600 "$SSH_KEY"
fi

required_files=(
  "3_star_model_train/processed/meta.pt"
  "3_star_model_train/processed/best_model_tags.pt"
  "3_star_model_train/processed/llm_repo_tag_cache.json"
  "1_3_repo_tags/tags_part_0.jsonl"
  "1_2_collect_api/github_repos_100plus_stars.csv"
  ".github_token"
)
for f in "${required_files[@]}"; do
  if [[ ! -f "${SCRIPT_DIR}/${f}" ]]; then
    echo "ERROR: Required file missing locally: ${SCRIPT_DIR}/${f}"; exit 1
  fi
done
echo "OK"

# ---------------------------------------------------------------------------
# 2. Bootstrap remote (idempotent)
# ---------------------------------------------------------------------------
if [[ "$SKIP_BOOTSTRAP" -eq 0 ]]; then
  log "Bootstrap remote (apt packages + directories)"
  ssh_run bash -s -- "$REMOTE_DIR" "$REMOTE_USER" <<'REMOTE_BOOTSTRAP'
    set -euo pipefail
    REMOTE_DIR="$1"
    REMOTE_USER="$2"

    export DEBIAN_FRONTEND=noninteractive
    sudo apt-get update -qq
    sudo apt-get install -y -qq \
      python3 python3-venv python3-pip \
      nginx rsync build-essential curl

    # Install Node.js 22 from NodeSource (apt ships Node 18 which is too old)
    if ! node --version 2>/dev/null | grep -qE '^v(2[0-9]|[3-9][0-9])'; then
      curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
      sudo apt-get install -y -qq nodejs
    fi

    mkdir -p \
      "${REMOTE_DIR}/4_website" \
      "${REMOTE_DIR}/3_star_model_train/processed" \
      "${REMOTE_DIR}/1_3_repo_tags" \
      "${REMOTE_DIR}/1_2_collect_api"

    echo "Bootstrap complete"
REMOTE_BOOTSTRAP
else
  log "Skipping bootstrap (--skip-bootstrap)"
fi

# ---------------------------------------------------------------------------
# 3. Rsync
# ---------------------------------------------------------------------------
log "Rsync: 4_website/"
rsync_to "4_website/" \
  --delete \
  --exclude="frontend/node_modules" \
  --exclude="frontend/.svelte-kit" \
  --exclude="frontend/build" \
  --exclude="backend/__pycache__" \
  --exclude="backend/.pytest_cache" \
  --exclude="**/*.pyc"

if [[ "$SKIP_DATA" -eq 0 ]]; then
  log "Rsync: 3_star_model_train/ (code + selected model files)"
  rsync_to "3_star_model_train/" \
    --delete \
    --include="model_tags.py" \
    --include="train_tags.py" \
    --include="recommend_tags.py" \
    --include="processed/" \
    --include="processed/meta.pt" \
    --include="processed/best_model_tags.pt" \
    --include="processed/llm_repo_tag_cache.json" \
    --exclude="processed/*" \
    --exclude="*"

  log "Rsync: 1_3_repo_tags/ (tag jsonl files)"
  rsync_to "1_3_repo_tags/" \
    --delete \
    --include="tags_part_*.jsonl" \
    --exclude="*"

  log "Rsync: 1_2_collect_api/github_repos_100plus_stars.csv"
  rsync \
    -az --info=progress2 -e "ssh $SSH_OPTS" \
    "${SCRIPT_DIR}/1_2_collect_api/github_repos_100plus_stars.csv" \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/1_2_collect_api/github_repos_100plus_stars.csv"

  log "Rsync: .github_token"
  rsync \
    -az -e "ssh $SSH_OPTS" \
    "${SCRIPT_DIR}/.github_token" \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/.github_token"
else
  log "Skipping data rsync (--skip-data)"
fi

# ---------------------------------------------------------------------------
# 4. Frontend build (remote)
# ---------------------------------------------------------------------------
log "Frontend build (remote)"
ssh_run bash -s -- "$REMOTE_DIR" <<'REMOTE_FRONTEND'
  set -euo pipefail
  REMOTE_DIR="$1"
  FRONTEND_DIR="${REMOTE_DIR}/4_website/frontend"

  cd "$FRONTEND_DIR"

  npm ci

  # Ensure adapter-static is installed (idempotent)
  if ! node -e "require('@sveltejs/adapter-static')" 2>/dev/null; then
    npm install --save-dev @sveltejs/adapter-static
  fi

  # Overwrite svelte.config.js to use adapter-static for SPA (ssr=false throughout)
  cat > svelte.config.js <<'SVELTE_CONFIG'
import adapter from '@sveltejs/adapter-static';

/** @type {import('@sveltejs/kit').Config} */
const config = {
  kit: {
    adapter: adapter({
      fallback: 'index.html',
    }),
  },
};

export default config;
SVELTE_CONFIG

  npm run build
  echo "Frontend build complete"
REMOTE_FRONTEND

# ---------------------------------------------------------------------------
# 5. Backend venv (remote)
# ---------------------------------------------------------------------------
log "Backend Python venv (remote)"
ssh_run bash -s -- "$REMOTE_DIR" <<'REMOTE_VENV'
  set -euo pipefail
  REMOTE_DIR="$1"

  cd "$REMOTE_DIR"

  if [[ ! -d venv ]]; then
    python3 -m venv venv
  fi

  venv/bin/pip install --upgrade pip --quiet

  # Install backend requirements (app deps) + gunicorn
  venv/bin/pip install --quiet \
    -r 4_website/backend/requirements.txt \
    gunicorn \
    requests numpy pandas

  # CPU-only torch (avoids downloading the massive CUDA wheel)
  if ! venv/bin/python -c "import torch" 2>/dev/null; then
    venv/bin/pip install --quiet \
      torch \
      --index-url https://download.pytorch.org/whl/cpu
  fi

  echo "Venv ready"
REMOTE_VENV

# ---------------------------------------------------------------------------
# 6. Systemd unit for backend
# ---------------------------------------------------------------------------
log "Install systemd unit: 4website-backend.service"
ssh_run sudo tee /etc/systemd/system/4website-backend.service > /dev/null <<SYSTEMD_UNIT
[Unit]
Description=4_website Flask backend
After=network.target

[Service]
User=${REMOTE_USER}
WorkingDirectory=${REMOTE_DIR}/4_website/backend
ExecStart=${REMOTE_DIR}/venv/bin/gunicorn -w 1 -k gthread --threads 4 -b 127.0.0.1:5000 --timeout 300 app:app
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
SYSTEMD_UNIT

# ---------------------------------------------------------------------------
# 7. Nginx site
# ---------------------------------------------------------------------------
log "Install nginx site: /etc/nginx/sites-available/4website"
ssh_run sudo tee /etc/nginx/sites-available/4website > /dev/null <<NGINX_CONF
server {
    listen 80 default_server;
    server_name _;
    client_max_body_size 1m;

    root ${REMOTE_DIR}/4_website/frontend/build;
    index index.html;

    location /api/ {
        proxy_pass http://127.0.0.1:5000;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header Connection "";
        proxy_buffering off;
        proxy_read_timeout 600s;
    }

    location / {
        try_files \$uri \$uri/ /index.html;
    }
}
NGINX_CONF

ssh_run bash -s <<NGINX_ENABLE
  set -euo pipefail
  sudo ln -sf /etc/nginx/sites-available/4website /etc/nginx/sites-enabled/4website
  sudo rm -f /etc/nginx/sites-enabled/default
  sudo nginx -t
  echo "Nginx config OK"
NGINX_ENABLE

# ---------------------------------------------------------------------------
# 8. Restart services + health check
# ---------------------------------------------------------------------------
log "Restart services"
ssh_run bash -s <<RESTART
  set -euo pipefail
  sudo systemctl daemon-reload
  sudo systemctl enable --now 4website-backend
  sudo systemctl restart 4website-backend
  sudo systemctl reload nginx
  echo "Services restarted"
RESTART

log "Health check"
sleep 3
HEALTH_HOST="${HEALTH_HOST:-$REMOTE_HOST}"
if curl -fsS "http://${HEALTH_HOST}/api/health" --max-time 15 > /dev/null 2>&1; then
  echo "OK — site is live at http://${HEALTH_HOST}"
else
  echo "WARNING: /api/health did not respond yet (backend may still be preloading model data)"
  echo "Check status with: ssh $SSH_OPTS ${REMOTE_USER}@${REMOTE_HOST} journalctl -u 4website-backend -n 40"
fi

log "Deploy complete -> http://${HEALTH_HOST}"
