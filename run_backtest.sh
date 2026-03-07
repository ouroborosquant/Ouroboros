#!/usr/bin/env bash
# FORTRESS v5 — run_backtest.sh
# Standalone pipeline: runs without TimescaleDB or trained model weights.
# Usage:  bash run_backtest.sh [--skip-audit] [--log-dir /path]
#
# Stage 0:  Forensic data audit (DB or synthetic parquet integrity)
# Stage 1:  Precompute regime posteriors  (Markov-GBM synthetic or real Mamba-KAN)
# Stage 2:  Precompute alpha signals      (5-factor surrogate or real GATv2)
# Stage 3:  Event-driven standalone backtest + walk-forward
# Stage 4:  Institutional tearsheet visualisation

set -euo pipefail

FORTRESS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="${FORTRESS_ROOT}/fortress_env/bin/python"

# Fallback: use system python if venv not found (CI environments)
if [[ ! -x "$VENV_PYTHON" ]]; then
    VENV_PYTHON=$(command -v python3 || command -v python)
    echo "[WARN] Venv not found at fortress_env/. Using system python: ${VENV_PYTHON}"
fi

LOG_DIR="${FORTRESS_ROOT}/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/pipeline_${TIMESTAMP}.log"
SKIP_AUDIT="${1:-}"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

GREEN='\033[0;32m'; CYAN='\033[0;36m'; RED='\033[0;31m'; NC='\033[0m'

log_stage() { echo -e "\n${CYAN}══════════════════════════════════════════${NC}"; 
              echo -e "${CYAN}  $1${NC}";
              echo -e "${CYAN}══════════════════════════════════════════${NC}"; }
log_ok()    { echo -e "${GREEN}[OK]${NC} $1"; }
log_err()   { echo -e "${RED}[FATAL]${NC} $1" >&2; }

run_stage() {
    local label="$1"; local script="$2"; shift 2
    log_stage "$label"
    if [[ ! -f "${FORTRESS_ROOT}/${script}" ]]; then
        log_err "Script not found: ${FORTRESS_ROOT}/${script}"; exit 2
    fi
    PYTHONPATH="${FORTRESS_ROOT}" "$VENV_PYTHON" "${FORTRESS_ROOT}/${script}" "$@"
    log_ok "${label} complete."
}

echo "[INFO] Fortress v5 — Backtest Pipeline | $(date)"
echo "[INFO] Log: ${LOG_FILE}"

# ── Stage 0: Forensic Audit ───────────────────────────────────────────────────
if [[ "$SKIP_AUDIT" == "--skip-audit" ]]; then
    echo "[WARN] --skip-audit set. Skipping integrity gate. Results may be fiction."
else
    run_stage "Stage 0: Forensic Data Audit" "scripts/audit_database.py"
fi

# ── Stage 1: Regime Posteriors ────────────────────────────────────────────────
run_stage "Stage 1: Regime Posterior Precomputation" "scripts/precompute_regime_posteriors.py"

# ── Stage 2: Alpha Signals ────────────────────────────────────────────────────
run_stage "Stage 2: Alpha Signal Precomputation" "scripts/precompute_alpha_signals.py"

# ── Stage 3: Backtest ─────────────────────────────────────────────────────────
run_stage "Stage 3: Event-Driven Standalone Backtest" "scripts/run_standalone_backtest.py"

# ── Stage 4: Verify tearsheet exists before visualising ───────────────────────
TEARSHEET="${FORTRESS_ROOT}/research/outputs/backtest_tearsheet.csv"
if [[ ! -f "$TEARSHEET" ]]; then
    log_err "Tearsheet not produced. Check Stage 3 logs."; exit 3
fi

run_stage "Stage 4: Tearsheet Visualisation" "scripts/visualize_tearsheet.py"

echo -e "\n${GREEN}Pipeline complete at $(date).${NC}"
echo "[INFO] Full log: ${LOG_FILE}"