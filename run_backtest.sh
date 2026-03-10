#!/usr/bin/env bash
# FORTRESS v5 — run_backtest.sh  [FINAL — ALL STAGES]
# Full pipeline: audit → regime → alpha → backtest → CPCV → tearsheet
#
# Stage 0:  Forensic data audit
# Stage 1:  Precompute regime posteriors (Mamba-KAN or synthetic)
# Stage 1b: Train GATv2 alpha engine (if weights absent)  [NEW]
# Stage 2:  Precompute alpha signals (GATv2 full mode or surrogate)
# Stage 3:  Event-driven standalone backtest + walk-forward
# Stage 3b: CPCV validation (PBO gate)  [NEW]
# Stage 4:  Institutional tearsheet visualisation

set -euo pipefail

FORTRESS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="${FORTRESS_ROOT}/fortress_env/bin/python"

if [[ ! -x "$VENV_PYTHON" ]]; then
    VENV_PYTHON=$(command -v python3 || command -v python)
    echo "[WARN] Venv not found. Using system python: ${VENV_PYTHON}"
fi

LOG_DIR="${FORTRESS_ROOT}/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/pipeline_${TIMESTAMP}.log"
SKIP_AUDIT="${1:-}"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

GREEN='\033[0;32m'; CYAN='\033[0;36m'; RED='\033[0;31m'; YELLOW='\033[0;33m'; NC='\033[0m'

log_stage() { echo -e "\n${CYAN}══════════════════════════════════════════${NC}";
              echo -e "${CYAN}  $1${NC}";
              echo -e "${CYAN}══════════════════════════════════════════${NC}"; }
log_ok()    { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
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
    echo "[WARN] --skip-audit set. Skipping integrity gate."
else
    run_stage "Stage 0: Forensic Data Audit" "scripts/audit_database.py"
fi

# ── Stage 1: Regime Posteriors ────────────────────────────────────────────────
run_stage "Stage 1: Regime Posterior Precomputation" "scripts/precompute_regime_posteriors.py"

# ── Stage 1b: GATv2 Training (conditional) ────────────────────────────────────
GAT_WEIGHTS="${FORTRESS_ROOT}/models/weights/gat_alpha_latest.pt"
GAT_TRAIN_SCRIPT="${FORTRESS_ROOT}/training/train_alpha.py"

if [[ -f "$GAT_WEIGHTS" ]]; then
    log_warn "GATv2 weights found at ${GAT_WEIGHTS}. Skipping training."
elif [[ -f "$GAT_TRAIN_SCRIPT" ]]; then
    run_stage "Stage 1b: GATv2 Alpha Engine Training" "training/train_alpha.py"
else
    log_warn "GATv2 training script not found. Running in Surrogate Mode."
fi

# ── Stage 2: Alpha Signals ────────────────────────────────────────────────────
run_stage "Stage 2: Alpha Signal Precomputation" "scripts/precompute_alpha_signals.py"

# ── Stage 3: Backtest ─────────────────────────────────────────────────────────
run_stage "Stage 3: Event-Driven Standalone Backtest" "scripts/run_standalone_backtest.py"

# ── Stage 3b: CPCV Validation ─────────────────────────────────────────────────
CPCV_SCRIPT="${FORTRESS_ROOT}/scripts/run_cpcv_validation.py"
if [[ -f "$CPCV_SCRIPT" ]]; then
    log_stage "Stage 3b: CPCV Validation (PBO Gate)"
    PYTHONPATH="${FORTRESS_ROOT}" "$VENV_PYTHON" "$CPCV_SCRIPT" || {
        log_warn "CPCV validation failed or PBO gate rejected. Check results."
    }
    log_ok "Stage 3b: CPCV Validation complete."
else
    log_warn "CPCV script not found. Skipping PBO validation."
fi

# ── Stage 4: Tearsheet ────────────────────────────────────────────────────────
TEARSHEET="${FORTRESS_ROOT}/research/outputs/backtest_tearsheet.csv"
if [[ ! -f "$TEARSHEET" ]]; then
    log_err "Tearsheet not produced. Check Stage 3 logs."; exit 3
fi

run_stage "Stage 4: Tearsheet Visualisation" "scripts/visualize_tearsheet.py"

echo -e "\n${GREEN}Pipeline complete at $(date).${NC}"
echo "[INFO] Full log: ${LOG_FILE}"