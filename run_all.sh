#!/bin/bash
# FORTRESS v5 - Master Ignition Sequence
# Path: run_all.sh
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
# Load environment secrets
if [ -f .env ]; then
    set -a
    source .env
    set +a
    echo -e "${GREEN}Environment secrets loaded from .env${NC}"
else
    echo -e "${RED}ERROR: .env file not found. Create it from the template before running.${NC}"
    exit 1
fi
# Exit immediately if any command exits with a non-zero status
set -e

# Colors for terminal output
GREEN='\033[0;32m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${CYAN}======================================================${NC}"
echo -e "${CYAN}      FORTRESS v5 - APEX QUANTITATIVE ORGANISM        ${NC}"
echo -e "${CYAN}             MASTER PIPELINE IGNITION                 ${NC}"
echo -e "${CYAN}======================================================${NC}\n"

# 1. Build required local directories if they don't exist
echo -e "${GREEN}[1/8] Verifying directory structure...${NC}"
mkdir -p models/weights
mkdir -p logs
mkdir -p data

# 2. Seed the historical memory
echo -e "${GREEN}[2/8] Bootstrapping Historical Memory (TimescaleDB & Alpaca)...${NC}"
python scripts/download_history.py

# 3. Train the Mamba-KAN Regime Encoder
echo -e "${GREEN}[3/8] Compressing Market History (Mamba-KAN VAE)...${NC}"
python training/train_regime.py

# 4. Train the Neural SDE World Model
echo -e "${GREEN}[4/8] Learning Continuous-Time Physics (Neural SDE)...${NC}"
python training/train_world_model.py

# 5. Train the Graph Attention Alpha Engine
echo -e "${GREEN}[5/8] Mapping Causal Shock Propagation (GATv2)...${NC}"
python training/train_gat.py

# 6. Train the Elastic Decision Transformer (Allocator)
echo -e "${GREEN}[6/8] Optimizing Portfolio Allocation (Diffusion EDT)...${NC}"
python training/train_edt.py

# 7. Train the Multi-Agent Execution Layer
echo -e "${GREEN}[7/8] Forging Execution Fangs (MARL & LTC)...${NC}"
python training/train_execution.py
python training/train_ltc.py
python training/train_hedging.py

# 8. The Crucible (Backtesting)
echo -e "${GREEN}[8/8] Executing Event-Driven Backtest...${NC}"
python research/backtest_engine.py

# 9. Generate Institutional Reporting
echo -e "${GREEN}[9/9] Generating Visual Tear Sheet...${NC}"
python scripts/visualize_tearsheet.py

echo -e "\n${CYAN}======================================================${NC}"
echo -e "${GREEN}      ALL SYSTEMS NOMINAL. FORTRESS IS ALIVE.         ${NC}"
echo -e "${CYAN}======================================================${NC}"