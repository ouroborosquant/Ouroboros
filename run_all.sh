#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# FORTRESS v5 - run_all.sh
# Path: run_all.sh
#
# Master Bootstrap Script.
# Single entry-point for training, validation, and live launch.
#
# STAGES:
#   0. Preflight   — Validate environment, connectivity, .env vars
#   1. Seed        — Download historical data into TimescaleDB (if needed)
#   2. Train       — Train all models in dependency order
#   3. Backtest    — Run full event-driven backtest + walk-forward validation
#   4. Validate    — Run 5-gate Constitutional ValidationPipeline
#   5. Live        — Launch all Docker microservices
#
# USAGE:
#   ./run_all.sh [--skip-train] [--skip-backtest] [--skip-validate] [--dry-run]
#
# REQUIREMENTS:
#   - Docker and docker compose v2+ installed
#   - .env file in project root with all required secrets
#   - Python 3.11+ with venv at .venv/ or globally installed
#   - NVIDIA CUDA drivers if GPU training is desired
#
# EXIT CODES:
#   0 — Success
#   1 — Preflight failure (missing .env, unreachable DB, etc.)
#   2 — Training failure
#   3 — Backtest failure
#   4 — Validation gate failure (system would destroy capital in live trading)
#   5 — Live launch failure
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Colour codes ──────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

# ── Flags ─────────────────────────────────────────────────────────────────────
SKIP_TRAIN=false
SKIP_BACKTEST=false
SKIP_VALIDATE=false
DRY_RUN=false

for arg in "$@"; do
  case $arg in
    --skip-train)     SKIP_TRAIN=true    ;;
    --skip-backtest)  SKIP_BACKTEST=true ;;
    --skip-validate)  SKIP_VALIDATE=true ;;
    --dry-run)        DRY_RUN=true       ;;
    *) echo "Unknown flag: $arg" && exit 1 ;;
  esac
done

# ── Helpers ───────────────────────────────────────────────────────────────────
log_header() {
  echo -e "\n${BOLD}${BLUE}══════════════════════════════════════════════════${RESET}"
  echo -e "${BOLD}${BLUE}  $1${RESET}"
  echo -e "${BOLD}${BLUE}══════════════════════════════════════════════════${RESET}"
}

log_ok()   { echo -e "  ${GREEN}✅ $1${RESET}"; }
log_warn() { echo -e "  ${YELLOW}⚠️  $1${RESET}"; }
log_err()  { echo -e "  ${RED}❌ $1${RESET}"; }
log_info() { echo -e "  ${CYAN}ℹ️  $1${RESET}"; }

run_or_dry() {
  if [ "$DRY_RUN" = true ]; then
    echo -e "  ${YELLOW}[DRY-RUN] Would run: $*${RESET}"
  else
    "$@"
  fi
}

# ── Stage 0: Preflight ────────────────────────────────────────────────────────
log_header "Stage 0: Preflight Checks"

# 0.1 Check .env
if [ ! -f ".env" ]; then
  log_err ".env file not found. Copy .env.example and fill in secrets."
  exit 1
fi
source .env
log_ok ".env loaded"

# 0.2 Required environment variables
REQUIRED_VARS=(
  "DB_PASSWORD"
  "ALPACA_API_KEY"
  "ALPACA_SECRET_KEY"
  "FRED_API_KEY"
)

MISSING=0
for var in "${REQUIRED_VARS[@]}"; do
  if [ -z "${!var:-}" ]; then
    log_err "Required env var \$$var is not set in .env"
    MISSING=1
  fi
done
[ $MISSING -eq 1 ] && exit 1
log_ok "All required environment variables set"

# 0.3 Docker availability
if ! docker compose version &>/dev/null; then
  log_err "Docker Compose v2 not found. Install via: https://docs.docker.com/compose/install/"
  exit 1
fi
log_ok "Docker Compose v2 available"

# 0.4 Python environment
PYTHON=$(which python3 2>/dev/null || which python 2>/dev/null || echo "")
if [ -z "$PYTHON" ]; then
  log_err "Python 3 not found in PATH"
  exit 1
fi
PY_VER=$($PYTHON --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
log_ok "Python $PY_VER found: $PYTHON"

# 0.5 GPU check (non-fatal)
if nvidia-smi &>/dev/null; then
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
  log_ok "GPU detected: $GPU_NAME (CUDA training enabled)"
else
  log_warn "No GPU detected. Training will run on CPU (significantly slower)."
fi

# 0.6 Disk space check (models + DB need ~50GB)
AVAILABLE_GB=$(df -BG . | awk 'NR==2 {print $4}' | tr -d 'G')
if [ "$AVAILABLE_GB" -lt 20 ]; then
  log_warn "Less than 20GB free disk space ($AVAILABLE_GB GB). DB + models may fail."
else
  log_ok "Disk space: ${AVAILABLE_GB}GB available"
fi

log_ok "Preflight passed"

# ── Start infrastructure ───────────────────────────────────────────────────────
log_header "Starting Infrastructure Services"

run_or_dry docker compose up -d timescaledb redis zookeeper kafka
log_info "Waiting for TimescaleDB to become healthy..."

if [ "$DRY_RUN" = false ]; then
  MAX_WAIT=60
  WAITED=0
  until docker compose exec timescaledb pg_isready -U postgres -d fortress &>/dev/null; do
    sleep 2
    WAITED=$((WAITED + 2))
    if [ $WAITED -ge $MAX_WAIT ]; then
      log_err "TimescaleDB failed to become healthy after ${MAX_WAIT}s"
      exit 1
    fi
  done
fi
log_ok "TimescaleDB healthy"

log_info "Running database migrations..."
run_or_dry docker compose exec -T timescaledb \
  psql -U postgres -d fortress \
  -f /docker-entrypoint-initdb.d/init.sql || true
log_ok "Migrations applied"

# ── Stage 1: Seed historical data ────────────────────────────────────────────
log_header "Stage 1: Historical Data Seed"

SEED_MARKER=".seed_complete"
if [ -f "$SEED_MARKER" ]; then
  log_ok "Seed marker found — skipping historical download (delete .seed_complete to re-seed)"
else
  log_info "Downloading historical price and macro data into TimescaleDB..."
  log_info "This takes 10-30 minutes on first run."
  run_or_dry $PYTHON scripts/download_history.py \
    --start-date 2005-01-01 \
    --end-date "$(date +%Y-%m-%d)"

  if [ "$DRY_RUN" = false ]; then
    touch "$SEED_MARKER"
  fi
  log_ok "Historical data seeded"
fi

# ── Stage 2: Training ─────────────────────────────────────────────────────────
log_header "Stage 2: Model Training"

if [ "$SKIP_TRAIN" = true ]; then
  log_info "Skipping training (--skip-train)"
else
  # Check weights already exist
  if [ -f "models/weights/mamba_kan_latest.pt" ] && \
     [ -f "models/weights/gat_alpha_latest.pt" ] && \
     [ -f "models/weights/edt_latest.pt" ] && \
     [ -f "models/weights/sde_latest.pt" ]; then
    log_warn "Weight files found. Re-training will overwrite them."
    read -p "  Continue? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
      log_info "Skipping training — using existing weights."
      SKIP_TRAIN=true
    fi
  fi

  if [ "$SKIP_TRAIN" = false ]; then
    # Training order matters: regime → world model → alpha → portfolio → execution
    TRAINING_ORDER=(
      "training/train_regime.py:Mamba-KAN VAE (Regime Encoder)"
      "training/train_world_model.py:Neural SDE World Model"
      "training/train_alpha.py:GATv2 Alpha Engine"
      "training/train_edt.py:Elastic Decision Transformer"
      "training/train_hedging.py:Deep Hedging Network"
      "training/train_execution.py:MARL Execution Agents"
    )

    for entry in "${TRAINING_ORDER[@]}"; do
      SCRIPT="${entry%%:*}"
      LABEL="${entry#*:}"
      log_info "Training: $LABEL..."
      run_or_dry $PYTHON "$SCRIPT"
      if [ $? -ne 0 ]; then
        log_err "Training failed: $SCRIPT"
        exit 2
      fi
      log_ok "$LABEL — done"
    done
  fi
fi

log_ok "All models trained"

# ── Stage 3: Backtesting ──────────────────────────────────────────────────────
log_header "Stage 3: Event-Driven Backtest"

if [ "$SKIP_BACKTEST" = true ]; then
  log_info "Skipping backtest (--skip-backtest)"
else
  run_or_dry $PYTHON research/backtest_engine.py
  if [ $? -ne 0 ]; then
    log_err "Backtest failed"
    exit 3
  fi

  if [ "$DRY_RUN" = false ] && [ -f "research/outputs/backtest_tearsheet.csv" ]; then
    # Print headline metrics
    log_ok "Backtest complete. Headline metrics:"
    $PYTHON - <<'EOF'
import pandas as pd
df = pd.read_csv("research/outputs/backtest_tearsheet.csv", index_col=0)
if not df.empty:
    last = df.iloc[-1]
    for col in ["cagr", "sharpe", "max_drawdown", "calmar", "dsr"]:
        if col in df.columns:
            print(f"    {col.upper():15s}: {last[col]:.4f}")
EOF
  fi
fi

# ── Stage 4: Constitutional Validation ───────────────────────────────────────
log_header "Stage 4: Constitutional ValidationPipeline (5 Gates)"

if [ "$SKIP_VALIDATE" = true ]; then
  log_warn "Skipping validation (--skip-validate). THIS IS DANGEROUS FOR LIVE TRADING."
else
  run_or_dry $PYTHON meta_learning/validation_pipeline.py \
    --mode standalone \
    --candidate-weights models/weights/ \
    --output-report research/outputs/validation_report.json

  if [ $? -ne 0 ]; then
    log_err "Validation pipeline FAILED — system did not pass constitutional gates."
    log_err "Refusing to launch live trading. Review research/outputs/validation_report.json."
    exit 4
  fi
  log_ok "All 5 validation gates passed"
fi

# ── Stage 5: Live Launch ──────────────────────────────────────────────────────
log_header "Stage 5: Launching Live Trading Organism"

run_or_dry docker compose up -d kafka-init
if [ "$DRY_RUN" = false ]; then
  log_info "Waiting for Kafka topics to be created..."
  sleep 10
fi

# Launch microservices in dependency order
SERVICES=(
  "regime-encoder"
  "alpha-engine"
  "portfolio-agent"
  "execution-router"
  "tda-topology"
)

for svc in "${SERVICES[@]}"; do
  log_info "Starting $svc..."
  run_or_dry docker compose up -d "$svc"
  sleep 2
done

# Health monitor as separate service
run_or_dry docker compose up -d health-monitor 2>/dev/null || \
  run_or_dry $PYTHON monitoring/strategy_health.py &

log_ok "All services launched"

# ── Final status ──────────────────────────────────────────────────────────────
log_header "FORTRESS v5 is LIVE"

if [ "$DRY_RUN" = false ]; then
  echo -e "\n${GREEN}${BOLD}  System status:${RESET}"
  docker compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null || true
  echo
  echo -e "${CYAN}  Monitor logs:  docker compose logs -f regime-encoder alpha-engine portfolio-agent${RESET}"
  echo -e "${CYAN}  Kill switch:   docker compose down${RESET}"
  echo -e "${CYAN}  Health check:  curl -s redis://localhost:6379/health:latest | jq${RESET}"
  echo
fi

exit 0