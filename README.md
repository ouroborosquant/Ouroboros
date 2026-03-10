# 🐍 Ouroboros — FORTRESS v5

> *A self-evolving quantitative trading organism that fuses state-space sequence modeling, graph attention networks, and meta-learned feature engineering to survive alpha decay and front-run causal macro shocks.*

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Repository Structure](#repository-structure)
4. [Core Components](#core-components)
   - [Mamba-SSM + GATv2 Shock Detector](#1-mamba-ssm--gatv2-shock-detector)
   - [Elastic Decision Transformer (EDT)](#2-elastic-decision-transformer-edt)
   - [Meta-Learning Feature Engine](#3-meta-learning-feature-engine)
   - [Live Execution Layer](#4-live-execution-layer)
   - [Federation & Multi-Agent Layer](#5-federation--multi-agent-layer)
5. [System Requirements](#system-requirements)
6. [Installation](#installation)
7. [Configuration](#configuration)
8. [Running the System](#running-the-system)
9. [Training](#training)
10. [Monitoring & Observability](#monitoring--observability)
11. [Testing](#testing)
12. [Research & Notebooks](#research--notebooks)
13. [Infrastructure & Deployment](#infrastructure--deployment)
14. [Design Philosophy](#design-philosophy)
15. [Roadmap](#roadmap)
16. [Disclaimer](#disclaimer)

---

## Overview

**Ouroboros** (internally: **FORTRESS v5**) is a research-grade, production-capable quantitative trading system built around the idea that a trading strategy is not a static artefact — it is a living organism that must continuously re-engineer itself to survive.

The name *Ouroboros* — the ancient symbol of a serpent consuming its own tail — captures the system's core loop: a model that autonomously generates the features it trains on, evaluates its own alpha, and rewrites itself when that alpha decays.

### Key Capabilities

| Capability | Description |
|---|---|
| **Macro shock detection** | Mamba-SSM encodes long-range temporal dependencies in macro time-series; GATv2 propagates causal shock signals across an asset dependency graph |
| **Portfolio allocation** | An Elastic Decision Transformer conditions on recent return trajectories to produce dynamic, risk-aware position weights |
| **Autonomous feature engineering** | A meta-learning loop (MAML-style) synthesises and evaluates new PyTorch feature extractors on rolling windows, retiring stale ones automatically |
| **Live execution** | A low-latency live trading layer handles order routing, position management, and real-time risk limits |
| **Multi-agent federation** | Specialised sub-agents (trend, mean-reversion, macro, volatility) are coordinated by an ensemble router |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          OUROBOROS — FORTRESS v5                    │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                     DATA INGESTION LAYER                     │   │
│  │   Market feeds · Macro calendars · Alt-data · Order books   │   │
│  └──────────────────────────────┬───────────────────────────────┘   │
│                                 │                                   │
│  ┌──────────────────────────────▼───────────────────────────────┐   │
│  │               META-LEARNING FEATURE ENGINE                   │   │
│  │   Autonomous PyTorch feature synthesis · Rolling eval        │   │
│  │   MAML-gradient-based feature discovery · Alpha decay radar  │   │
│  └─────────────┬────────────────────────────────┬───────────────┘   │
│                │                                │                   │
│  ┌─────────────▼────────────┐   ┌───────────────▼───────────────┐   │
│  │   MAMBA-SSM ENCODER      │   │      GATv2 GRAPH MODULE       │   │
│  │  Long-range macro seq    │   │  Asset correlation / causal   │   │
│  │  Linear-time complexity  │   │  shock propagation graph      │   │
│  └─────────────┬────────────┘   └───────────────┬───────────────┘   │
│                │                                │                   │
│  ┌─────────────▼────────────────────────────────▼───────────────┐   │
│  │            ELASTIC DECISION TRANSFORMER (EDT)                │   │
│  │   Trajectory-conditioned allocation · Risk-adjusted sizing   │   │
│  │   Elastic context window · Multi-horizon return targets      │   │
│  └──────────────────────────────┬───────────────────────────────┘   │
│                                 │                                   │
│  ┌──────────────────────────────▼───────────────────────────────┐   │
│  │            FEDERATION / MULTI-AGENT ROUTER                   │   │
│  │   Trend · Mean-Rev · Macro · Vol — ensemble weighting        │   │
│  └──────────────────────────────┬───────────────────────────────┘   │
│                                 │                                   │
│  ┌──────────────────────────────▼───────────────────────────────┐   │
│  │                   LIVE EXECUTION LAYER                       │   │
│  │    Order routing · Position limits · Real-time P&L · Risk    │   │
│  └──────────────────────────────┬───────────────────────────────┘   │
│                                 │                                   │
│  ┌──────────────────────────────▼───────────────────────────────┐   │
│  │                 MONITORING & OBSERVABILITY                    │   │
│  │     Prometheus · Grafana dashboards · Alert pipelines        │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
Ouroboros/
│
├── .github/
│   └── workflows/            # CI/CD pipelines (lint, test, deploy)
│
├── config/                   # YAML/TOML configuration files
│                             # (model hyperparameters, broker credentials, risk limits)
│
├── data/                     # Data pipeline utilities
│                             # (fetchers, preprocessors, universe definitions)
│
├── federation/               # Multi-agent router and sub-agent definitions
│                             # (trend, mean-reversion, macro, volatility agents)
│
├── infrastructure/           # IaC, Docker configurations, cloud provisioning
│
├── live/                     # Live trading execution engine
│                             # (order management, position tracking, risk checks)
│
├── meta_learning/            # Autonomous feature engineering loop
│                             # (MAML-style gradient-based feature synthesis,
│                             #  alpha decay detection, feature retirement)
│
├── models/                   # Model definitions
│                             # (Mamba-SSM encoder, GATv2 graph module,
│                             #  Elastic Decision Transformer)
│
├── monitoring/               # Observability stack
│                             # (Prometheus exporters, Grafana dashboards, alerting)
│
├── notebooks/                # Research and exploratory Jupyter notebooks
│
├── research/                 # Hypothesis testing, strategy backtests, signal research
│
├── scripts/                  # Utility scripts (data download, checkpoint export, etc.)
│
├── services/                 # Microservice definitions (data service, inference service)
│
├── tests/                    # Unit, integration and system tests
│
├── training/                 # Training loops, experiment configs, schedulers
│
├── docker-compose.yml        # Local dev / staging stack definition
├── pyproject.toml            # Build system and tooling configuration
├── requirements.txt          # Production Python dependencies
├── requirements-dev.txt      # Development & testing dependencies
└── run_all.sh                # One-shot launch script for the full stack
```

---

## Core Components

### 1. Mamba-SSM + GATv2 Shock Detector

The signal backbone combines two complementary architectures:

**Mamba-SSM (Selective State Space Model)**

Mamba addresses the quadratic complexity bottleneck of attention-based models by replacing attention with a learnable, input-selective state space recurrence. For financial time-series this matters: macro regimes evolve over months and years, requiring the model to propagate information across sequences far longer than transformer context windows typically permit — without the $O(n^2)$ cost.

Key properties used here:
- Selective scan mechanism allows the model to *gate* which macro signals are relevant at each time step
- Linear-time inference enables intraday re-scoring of the full macro lookback window
- Hardware-aware CUDA kernels (via `mamba_ssm`) keep latency compatible with live trading

**GATv2 (Graph Attention Network v2)**

Assets do not move in isolation. Equity sectors, rates, FX, and commodities are causally linked through global macro flow. GATv2 builds a dynamic attention-weighted graph over the asset universe, propagating shock signals from causal sources (e.g. surprise CPI prints, central bank announcements) to downstream assets before the market fully reprices.

GATv2 improves on the original GAT by computing attention *jointly* from both source and target node features, avoiding the static attention rank-collapse issue of the original formulation.

---

### 2. Elastic Decision Transformer (EDT)

The Decision Transformer (DT) reframes portfolio construction as an offline sequence modelling problem: given a desired return trajectory, what sequence of allocations achieves it? The *Elastic* variant extends the standard DT with:

- **Variable context windows** — the model dynamically expands or contracts its conditioning history based on current regime volatility
- **Multi-horizon targets** — return-to-go tokens span multiple forecast horizons simultaneously (intraday, daily, weekly), encouraging consistent cross-horizon allocation
- **Risk-adjusted sizing** — position weights are modulated by a learned volatility estimator embedded in the EDT's conditioning tokens

The EDT receives fused embeddings from the Mamba encoder and the GATv2 graph readout and outputs a full portfolio weight vector at each decision step.

---

### 3. Meta-Learning Feature Engine

Alpha decays. Signals that predicted returns six months ago may be noise today. The meta-learning module addresses this through *autonomous feature lifecycle management*:

1. **Candidate generation** — small PyTorch feature modules are synthesised (via gradient-based meta-learning inspired by MAML) and registered as candidates
2. **Rolling evaluation** — each candidate is scored on an expanding out-of-sample window using information coefficient (IC), IC-IR, and downstream model improvement
3. **Promotion / retirement** — features crossing promotion thresholds are merged into the live feature set; features whose IC degrades below a decay floor are automatically retired
4. **Self-healing** — when aggregate model performance drops below a configurable threshold, the meta-learning loop increases its search budget autonomously

This loop runs asynchronously on a separate process and updates the feature set without requiring manual intervention.

---

### 4. Live Execution Layer

The `live/` module handles all aspects of real-money execution:

- **Order management** — bracket orders, limit/market routing, fill confirmation
- **Position management** — real-time P&L attribution per signal, per asset, per agent
- **Pre-trade risk checks** — gross/net exposure limits, sector concentration caps, drawdown kill-switch
- **Latency monitoring** — end-to-end signal-to-order latency tracked and alarmed via the monitoring stack

---

### 5. Federation & Multi-Agent Layer

`federation/` implements an ensemble of specialised trading agents:

| Agent | Signal Regime | Horizon |
|---|---|---|
| `TrendAgent` | Momentum & breakout | Medium (weeks–months) |
| `MeanRevAgent` | Statistical arbitrage | Short (hours–days) |
| `MacroAgent` | Macro factor rotation | Long (months) |
| `VolAgent` | Volatility risk premia | Short–medium |

The federation router dynamically re-weights agent allocations based on recent performance attribution and detected regime, then aggregates into a single portfolio instruction passed to the execution layer.

---

## System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| Python | 3.10 | 3.11 |
| CUDA | 11.8 | 12.1+ |
| GPU VRAM | 16 GB | 40 GB (A100) |
| RAM | 32 GB | 128 GB |
| Storage | 500 GB SSD | 2 TB NVMe |
| OS | Ubuntu 22.04 | Ubuntu 22.04 / 24.04 |

> **⚠️ CUDA ABI note:** `mamba_ssm` and `causal-conv1d` must be compiled against the same CUDA toolkit and PyTorch ABI as your runtime environment. Mismatches (e.g. upgrading PyTorch without recompiling these packages) will produce silent correctness failures or segfaults. Always rebuild after any PyTorch version upgrade.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/ouroborosquant/Ouroboros.git
cd Ouroboros
```

### 2. Create and activate a virtual environment

```bash
conda create -n fortress_env python=3.10 -y
conda activate fortress_env
```

### 3. Install CUDA-matched dependencies

Install PyTorch first, matching your CUDA version:

```bash
# Example: CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install `mamba_ssm` and `causal-conv1d`

These must be compiled from source against your installed PyTorch version:

```bash
pip install causal-conv1d>=1.2.0
pip install mamba-ssm
```

> If you encounter ABI errors, force a clean rebuild:
> ```bash
> pip install --no-build-isolation mamba-ssm causal-conv1d
> ```

### 5. Install project dependencies

```bash
pip install -r requirements.txt
```

For development (linting, testing, notebooks):

```bash
pip install -r requirements-dev.txt
```

### 6. Install the package in editable mode

```bash
pip install -e .
```

---

## Configuration

All runtime configuration lives under `config/`. Key files:

```
config/
├── model.yaml          # Architecture hyperparameters (Mamba, GATv2, EDT)
├── training.yaml       # Training schedule, optimiser, scheduler
├── data.yaml           # Universe definition, data sources, feature toggles
├── risk.yaml           # Position limits, drawdown thresholds, kill-switches
├── live.yaml           # Broker endpoints, order routing rules
└── meta_learning.yaml  # Feature candidate budget, IC thresholds, decay floors
```

Copy and edit the example configs before first run:

```bash
cp config/example/* config/
# Edit config/ files with your broker credentials, data API keys, etc.
```

> **Security:** Never commit API keys or broker credentials. Use environment variables or a secrets manager. The `.gitignore` is pre-configured to exclude `config/secrets.*`.

---

## Running the System

### Full stack (Docker)

```bash
docker-compose up --build
```

This spins up:
- The inference service
- The live execution engine
- The meta-learning worker
- The monitoring stack (Prometheus + Grafana)

### Full stack (native)

```bash
bash run_all.sh
```

### Individual services

```bash
# Data service only
python -m services.data_service

# Inference / signal generation only
python -m services.inference_service

# Live execution only
python -m live.engine

# Meta-learning worker only
python -m meta_learning.worker
```

---

## Training

Training experiments are managed under `training/`. The system uses a multi-stage training curriculum:

### Stage 1 — Pre-train Mamba-SSM encoder on macro time-series

```bash
python training/pretrain_mamba.py --config config/training.yaml
```

### Stage 2 — Train GATv2 graph module on asset correlation graph

```bash
python training/train_gatv2.py --config config/training.yaml
```

### Stage 3 — Fine-tune Elastic Decision Transformer end-to-end

```bash
python training/train_edt.py --config config/training.yaml
```

### Stage 4 — Bootstrap meta-learning feature engine

```bash
python training/bootstrap_meta.py --config config/meta_learning.yaml
```

### Experiment tracking

Training runs log metrics to the monitoring stack. To view training curves:

```bash
# Start Grafana locally
docker-compose up monitoring
# Then open: http://localhost:3000
```

---

## Monitoring & Observability

The `monitoring/` directory contains:

- **Prometheus exporters** — trading metrics (P&L, Sharpe, turnover, drawdown, signal IC, order fill rate, latency) scraped at configurable intervals
- **Grafana dashboards** — pre-built dashboards for live P&L, risk heatmaps, feature health, and model inference latency
- **Alert rules** — kill-switch triggers for max drawdown breach, latency spikes, and meta-learning degradation events

Access dashboards after starting the stack:
- Grafana: `http://localhost:3000` (default user: `admin` / `fortress`)
- Prometheus: `http://localhost:9090`

---

## Testing

```bash
# Run full test suite
pytest tests/ -v

# Unit tests only
pytest tests/unit/ -v

# Integration tests (requires data service running)
pytest tests/integration/ -v

# Model smoke tests (fast, CPU-only)
pytest tests/models/ -v --device cpu
```

Test coverage report:

```bash
pytest tests/ --cov=. --cov-report=html
open htmlcov/index.html
```

---

## Research & Notebooks

The `research/` and `notebooks/` directories contain:

- **Signal research** — IC analysis, factor decay curves, regime detection studies
- **Architecture ablations** — comparisons of Mamba vs. LSTM vs. Transformer encoders; GAT vs. GATv2
- **Backtest harness** — vectorised event-driven backtester with transaction cost models
- **Meta-learning analysis** — feature lifetime distributions, IC improvement curves from autonomous synthesis

To launch notebooks:

```bash
jupyter lab notebooks/
```

---

## Infrastructure & Deployment

The `infrastructure/` directory contains:

- **Docker** — base images for GPU inference, CPU worker, and monitoring services
- **docker-compose.yml** — local development and staging stack
- **Cloud provisioning** — scripts for provisioning GPU instances (AWS/GCP/Azure)
- **CI/CD** (`.github/workflows/`) — automated lint, test, and container build pipelines on push to `main`

### GitHub Actions Workflows

| Workflow | Trigger | Steps |
|---|---|---|
| `ci.yml` | Push / PR | Lint (ruff), type-check (mypy), unit tests |
| `build.yml` | Push to `main` | Build and push Docker images |
| `deploy.yml` | Manual dispatch | Deploy to staging / production |

---

## Design Philosophy

**1. Alpha is perishable — the system must not be.**
Every signal will eventually decay. Ouroboros treats feature engineering as a continuous process rather than a one-time design decision. The meta-learning loop is not an add-on; it is the immune system.

**2. Sequence matters more than cross-section.**
Most quant systems flatten temporal structure into cross-sectional features. Mamba-SSM preserves the full causal sequence of macro events, allowing the system to *front-run* repricing rather than react to it.

**3. Assets are a graph, not a list.**
Treating assets as independent time-series discards enormous amounts of causal information. GATv2 restores the relational structure, propagating shocks along economically meaningful edges.

**4. Decisions should be conditioned on intentions.**
The Decision Transformer paradigm — conditioning allocation on desired return-to-go — aligns the model's objectives with the portfolio manager's risk preferences at inference time, without retraining.

**5. The system should be its own harshest critic.**
Kill-switches, IC monitoring, and automated feature retirement exist not to protect individual positions, but to protect the system's long-run edge.

---

## Roadmap

- [ ] **v5.1** — Causal discovery integration: replace static graph edges with learned causal edges (PC-algorithm / NOTEARS)
- [ ] **v5.2** — Online RL fine-tuning loop: continuously update EDT policy weights using live P&L as reward signal
- [ ] **v5.3** — Alternative data connectors: earnings call NLP, satellite imagery, options flow
- [ ] **v5.4** — Multi-asset class support: extend GATv2 graph to include crypto, commodities, and rates
- [ ] **v6.0** — Full autonomous research loop: hypothesis generation → backtest → promotion → live without human intervention

---

## Disclaimer

> **This repository is for research and educational purposes only.** Nothing in this codebase constitutes financial advice. Trading financial instruments carries substantial risk of loss. Past performance of any strategy — simulated or live — is not indicative of future results. Use of this system in live markets is entirely at your own risk. The authors accept no liability for any financial loss arising from the use of this software.

---

<div align="center">
  <sub>
    <em>The serpent that devours itself is reborn.</em>
  </sub>
</div>