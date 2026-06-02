"""
FORTRESS v5 — scripts/train_gat_router.py  [v3.0 — Blueprint Suite]

Trains SignalRouterGAT on the Blueprint Suite 5-signal tensor:
    ["low_vol", "ramom_ts", "odpv_vwap", "clv_flow", "dtfe_trend"]

Prerequisites (must run first):
    PYTHONPATH=. python scripts/precompute_alpha_signals.py

Objective
---------
Learn per-asset, per-regime signal blending weights that maximise the
forward 21-day cross-sectional Spearman IC of the blended alpha vector.

Training protocol
-----------------
• Purged walk-forward split (80/20 by time, 21-day purge gap between IS/OOS)
• Loss: three-term (IC-MSE + IC-reward + entropy regularisation)
• Optimizer: AdamW + CosineAnnealingLR
• Gradient clipping (L2 norm ≤ 1.0)
• Early stopping: 30 epochs patience on validation IC
• Checkpoint: saves model_state_dict + signal_names for version-mismatch detection

Node feature construction (NODE_FEAT_DIM = 18)
----------------------------------------------
[0:5]   Current-day Z-scored signal values (5 signals)
[5:10]  Rolling 63-day mean |Spearman IC| per signal
[10:12] SPY beta (63d OLS), VIX beta (63d OLS)
[12:15] 63d annualised vol, 21d skew, 21d excess kurtosis
[15:18] Asset-class encoding: [is_bond, is_commodity, is_vol_product]

Global feature vector (GLOBAL_FEAT_DIM = 16)
--------------------------------------------
[0]     LTC urgency (vol regime scalar)
[1]     SPY 21d realised vol (annualised)
[2]     Breadth ratio (fraction tickers positive 5d return)
[3:8]   Cross-sectional mean of each signal (5 values)
[8:14]  Cross-sectional |IC| mean per signal (5 values), padded to 6 slots
[14:16] Normalised time index (fractional), VIX Z-score proxy
"""
from __future__ import annotations

import logging
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="An input array is constant")
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from scipy.stats import spearmanr
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Ouroboros.TrainGAT")

# ── Paths ─────────────────────────────────────────────────────────────────────
_BASE        = Path(".")
_SIGNALS_PATH = _BASE / "data/processed/signal_tensor.parquet"
_RETURNS_PATH = _BASE / "data/processed/returns.parquet"    # optional cache
_REGIME_PATH  = _BASE / "data/processed/regime_posteriors.parquet"
_WEIGHTS_DIR  = _BASE / "models/weights"
_WEIGHTS_PATH = _WEIGHTS_DIR / "gat_router.pt"

# ── Hyper-parameters ──────────────────────────────────────────────────────────
TRAIN_FRAC   = 0.80
HORIZON      = 21       # forward return horizon for IC target
BATCH_SIZE   = 32
N_EPOCHS     = 200
GRAD_CLIP    = 1.0
PURGE_DAYS   = 21       # embargo gap between train and val
PATIENCE     = 30       # early stopping
IC_WINDOW    = 63       # rolling IC estimation window for node features
LAMBDA_ENT   = 0.15     # Fuerza mayor dispersión (entropía) en los pesos asignados
LAMBDA_L2_IC = 0.05     # Penaliza con dureza las desviaciones cuadráticas del IC predicho

# Import constants from router (single source of truth)
sys.path.insert(0, str(_BASE))
from models.alpha.gat_signal_router import (
    SIGNAL_NAMES, N_SIGNALS, N_ASSETS, TICKERS,
    NODE_FEAT_DIM, GLOBAL_FEAT_DIM,
    SignalRouterGAT, signal_router_loss, build_economic_graph,
    _ASSET_CLASS_FEATS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Feature construction helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_rolling_ic(
    sig_arr: np.ndarray,   # (T, N)  signal values
    ret_arr: np.ndarray,   # (T, N)  daily log-returns
    horizon: int,
    window:  int,
) -> np.ndarray:
    """
    Rolling Spearman IC: (T, N) array where entry (t, n) = Spearman corr
    between signal[t-window:t, n] and fwd_return[t-window+horizon:t+horizon, n].

    Causal: IC at t uses only signal values up to t-1 and returns that have
    already realised (look-ahead is in the TARGET Y_ic, not the features X).
    """
    T, N = sig_arr.shape
    ic   = np.zeros((T, N), dtype=np.float32)

    # Upper bound: t + horizon must stay within the array, i.e. t <= T - horizon.
    # The original T ceiling allowed r_win to clip short and produce shape mismatches.
    for t in range(window + horizon, T - horizon + 1):
        s_win = sig_arr[t - window : t]                      # exactly (window, N)
        r_win = ret_arr[t - window + horizon : t + horizon]  # exactly (window, N)
        if s_win.shape[0] != r_win.shape[0]:
            # Defensive guard: should never trigger after the range fix
            continue
        for n in range(N):
            valid = np.isfinite(s_win[:, n]) & np.isfinite(r_win[:, n])
            if valid.sum() < 10:
                continue
            rho, _ = spearmanr(s_win[valid, n], r_win[valid, n])
            if np.isfinite(rho):
                ic[t, n] = float(rho)
    return ic


def build_node_features(
    sig_slice:    np.ndarray,   # (N, S)  current signals
    ic_slice:     np.ndarray,   # (N, S)  rolling |IC| per signal
    vol_arr:      np.ndarray,   # (N,)    63d realised vol
    skew_arr:     np.ndarray,   # (N,)    21d skew
    kurt_arr:     np.ndarray,   # (N,)    21d excess kurtosis
    spy_betas:    np.ndarray,   # (N,)    63d OLS beta vs SPY
    vix_betas:    np.ndarray,   # (N,)    63d OLS beta vs VIX changes
) -> np.ndarray:
    """
    Assemble (N, NODE_FEAT_DIM=18) node feature matrix.

    Layout:
      [0:5]   signal values   (already Z-scored + tanh-bounded at engine level)
      [5:10]  rolling |IC|
      [10:12] [spy_beta, vix_beta]
      [12:15] [vol_63d, skew_21d, kurt_21d]
      [15:18] [is_bond, is_commodity, is_vol_product]
    """
    N = len(TICKERS)
    feats = np.zeros((N, NODE_FEAT_DIM), dtype=np.float32)

    feats[:, 0:N_SIGNALS]           = sig_slice.clip(-1.0, 1.0)
    feats[:, N_SIGNALS:2*N_SIGNALS] = np.abs(ic_slice).clip(0.0, 1.0)
    feats[:, 10] = spy_betas.clip(-5.0, 5.0)
    feats[:, 11] = vix_betas.clip(-5.0, 5.0)
    feats[:, 12] = vol_arr.clip(0.0, 2.0)
    feats[:, 13] = skew_arr.clip(-5.0, 5.0)
    feats[:, 14] = kurt_arr.clip(-5.0, 5.0)

    for i, ticker in enumerate(TICKERS):
        b, c, v = _ASSET_CLASS_FEATS.get(ticker, (0, 0, 0))
        feats[i, 15] = float(b)
        feats[i, 16] = float(c)
        feats[i, 17] = float(v)

    return feats


def build_global_features(
    urgency:     float,
    spy_vol:     float,
    breadth:     float,
    sig_means:   np.ndarray,   # (S,)  cross-sectional mean of each signal
    ic_means:    np.ndarray,   # (S,)  cross-sectional mean |IC| per signal
    t_frac:      float,        # fractional time index ∈ [0, 1]
    vix_z:       float = 0.0,
) -> np.ndarray:
    """
    Assemble (GLOBAL_FEAT_DIM=16,) global context vector.
    """
    g = np.zeros(GLOBAL_FEAT_DIM, dtype=np.float32)
    g[0]  = float(urgency)
    g[1]  = float(spy_vol)
    g[2]  = float(breadth)
    g[3:3+N_SIGNALS]  = sig_means.clip(-1.0, 1.0)   # [3:8]
    g[8:8+N_SIGNALS]  = np.abs(ic_means).clip(0.0, 1.0)  # [8:13]
    g[13] = 0.0                 # reserved
    g[14] = float(t_frac)
    g[15] = float(vix_z)
    return g


# ─────────────────────────────────────────────────────────────────────────────
# Rolling distributional features
# ─────────────────────────────────────────────────────────────────────────────

def _compute_return_stats(
    ret_arr: np.ndarray,   # (T, N)
    vol_w:   int = 63,
    mom_w:   int = 21,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (vol, skew, kurt, cumret) arrays, each (T, N).
    All lookback-only — causal.
    """
    T, N = ret_arr.shape
    vol  = np.zeros((T, N), dtype=np.float32)
    skew = np.zeros((T, N), dtype=np.float32)
    kurt = np.zeros((T, N), dtype=np.float32)

    ret_df = pd.DataFrame(ret_arr)
    vol_df  = ret_df.rolling(vol_w, min_periods=20).std() * np.sqrt(252)
    skew_df = ret_df.rolling(mom_w, min_periods=10).skew()
    kurt_df = ret_df.rolling(mom_w, min_periods=10).kurt()

    vol  = vol_df.fillna(0.0).values.astype(np.float32)
    skew = skew_df.fillna(0.0).values.astype(np.float32)
    kurt = kurt_df.fillna(0.0).values.astype(np.float32)
    return vol, skew, kurt


def _compute_betas(
    ret_arr: np.ndarray,   # (T, N)
    spy_idx: int,
    window:  int = 63,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rolling 63-day OLS β vs SPY and β vs (lagged-1 SPY, as VIX proxy).
    Returns (spy_betas, vix_betas) both (T, N).
    Uses vectorised formula: β = cov(R_i, R_spy) / var(R_spy).
    """
    T, N = ret_arr.shape
    spy = ret_arr[:, spy_idx]
    spy_df = pd.Series(spy)
    ret_df = pd.DataFrame(ret_arr)

    spy_betas = np.zeros((T, N), dtype=np.float32)
    vix_proxy = pd.Series(np.abs(spy) - pd.Series(spy).rolling(21).mean().values)

    for n in range(N):
        r_n = pd.Series(ret_arr[:, n])
        cov_spy = r_n.rolling(window, min_periods=20).cov(spy_df)
        var_spy = spy_df.rolling(window, min_periods=20).var().clip(lower=1e-8)
        spy_betas[:, n] = (cov_spy / var_spy).fillna(1.0).values.astype(np.float32)

    vix_betas = np.zeros((T, N), dtype=np.float32)
    for n in range(N):
        r_n = pd.Series(ret_arr[:, n])
        cov_vix = r_n.rolling(window, min_periods=20).cov(vix_proxy)
        var_vix = vix_proxy.rolling(window, min_periods=20).var().clip(lower=1e-8)
        vix_betas[:, n] = (cov_vix / var_vix).fillna(0.0).values.astype(np.float32)

    return spy_betas, vix_betas


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("=" * 70)
    logger.info("FORTRESS v5 — train_gat_router.py  [v3.0 — Blueprint Suite]")
    logger.info(f"Signals: {SIGNAL_NAMES}")
    logger.info(f"Assets : {N_ASSETS} | NODE_FEAT_DIM={NODE_FEAT_DIM} | "
                f"GLOBAL_FEAT_DIM={GLOBAL_FEAT_DIM}")
    logger.info("=" * 70)

    # ── Guard: fail loudly if stale weights exist ─────────────────────────────
    if _WEIGHTS_PATH.exists():
        ckpt = torch.load(_WEIGHTS_PATH, map_location="cpu", weights_only=True)
        stored = ckpt.get("signal_names", [])
        if stored and stored != SIGNAL_NAMES:
            logger.error(
                f"Stale checkpoint detected!\n"
                f"  Stored signals : {stored}\n"
                f"  Current signals: {SIGNAL_NAMES}\n"
                f"  Delete {_WEIGHTS_PATH} before retraining."
            )
            sys.exit(1)
        logger.info(f"Overwriting existing checkpoint (signal_names match).")

    # ── Load data ─────────────────────────────────────────────────────────────
    if not _SIGNALS_PATH.exists():
        logger.error(
            f"Signal tensor not found at {_SIGNALS_PATH}. "
            "Run: PYTHONPATH=. python scripts/precompute_alpha_signals.py"
        )
        sys.exit(1)

    signals_df = pd.read_parquet(_SIGNALS_PATH)
    signals_df.index = pd.to_datetime(signals_df.index)
    signals_df.sort_index(inplace=True)

    # Download returns via yfinance if cache absent
    if _RETURNS_PATH.exists():
        returns_df = pd.read_parquet(_RETURNS_PATH)
        returns_df.index = pd.to_datetime(returns_df.index)
    else:
        import yfinance as yf
        logger.info("Downloading returns via yfinance...")
        raw = yf.download(TICKERS, start="2018-01-01", auto_adjust=True, progress=False)
        closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        returns_df = np.log(closes / closes.shift(1)).reindex(columns=TICKERS)
        returns_df.index = pd.to_datetime(returns_df.index)
        _RETURNS_PATH.parent.mkdir(parents=True, exist_ok=True)
        returns_df.to_parquet(_RETURNS_PATH)

    # Load regime
    regime_df = pd.DataFrame(index=returns_df.index, columns=["ltc_urgency"])
    if _REGIME_PATH.exists():
        regime_df = pd.read_parquet(_REGIME_PATH)
        regime_df.index = pd.to_datetime(regime_df.index)

    # Align all data on common trading days
    common_dates = returns_df.dropna(how="all").index
    returns_df   = returns_df.reindex(common_dates).ffill().fillna(0.0)
    signals_df   = signals_df.reindex(common_dates).ffill().fillna(0.0)
    regime_df    = regime_df.reindex(common_dates).ffill().fillna(0.3)

    T = len(common_dates)
    logger.info(f"Aligned dataset: T={T} days")

    # ── Build signal stack (T, N, S) ──────────────────────────────────────────
    ret_arr  = returns_df.reindex(columns=TICKERS).values.astype(np.float32)
    sig_stack = np.zeros((T, N_ASSETS, N_SIGNALS), dtype=np.float32)

    missing: List[str] = []
    for s_idx, sig_name in enumerate(SIGNAL_NAMES):
        cols = [f"{sig_name}_{t}" for t in TICKERS]
        present = [c for c in cols if c in signals_df.columns]
        if not present:
            missing.append(sig_name)
            continue
        sig_stack[:, :len(present), s_idx] = (
            signals_df[present].values.astype(np.float32)
        )

    if missing:
        logger.error(f"Signals absent from signal_tensor.parquet: {missing}")
        sys.exit(1)

    logger.info(f"Signal stack: {sig_stack.shape}")

    # ── Pre-compute rolling IC (expensive — O(T·N·S·window)) ─────────────────
    logger.info("Computing per-signal rolling IC (may take 3-8 min)...")
    ic_features = np.zeros((T, N_ASSETS, N_SIGNALS), dtype=np.float32)
    for s_idx in range(N_SIGNALS):
        ic_features[:, :, s_idx] = _compute_rolling_ic(
            sig_arr = sig_stack[:, :, s_idx],
            ret_arr = ret_arr,
            horizon = HORIZON,
            window  = IC_WINDOW,
        )
        mean_ic = float(np.abs(ic_features[IC_WINDOW:, :, s_idx]).mean())
        logger.info(f"  {SIGNAL_NAMES[s_idx]:14s} | mean|IC|={mean_ic:.4f}")

    # ── Pre-compute return stats for node features ────────────────────────────
    logger.info("Computing return statistics...")
    vol_arr, skew_arr, kurt_arr = _compute_return_stats(ret_arr)
    spy_idx   = TICKERS.index("SPY") if "SPY" in TICKERS else 0
    spy_betas, vix_betas = _compute_betas(ret_arr, spy_idx=spy_idx)

    # ── Build economic graph (fixed for entire training run) ──────────────────
    logger.info("Building economic correlation graph...")
    edge_index, edge_attr = build_economic_graph(
        returns_df = returns_df,
        tickers    = TICKERS,
    )

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    edge_index = edge_index.to(device)
    edge_attr  = edge_attr.to(device)

    # ── Model + optimizer ─────────────────────────────────────────────────────
    model     = SignalRouterGAT(n_signals=N_SIGNALS).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=1e-5)

    # ── Purged train/val split ─────────────────────────────────────────────────
    warmup     = IC_WINDOW + HORIZON + 252   # first valid date with full features
    train_end  = int(T * TRAIN_FRAC)
    val_start  = train_end + PURGE_DAYS
    train_dates = list(range(warmup, train_end - HORIZON))
    valid_dates = list(range(val_start, T - HORIZON))

    logger.info(
        f"Train: {len(train_dates)} steps | Val: {len(valid_dates)} steps | "
        f"Warmup: {warmup} bars"
    )

    best_val_ic    = -np.inf
    best_state: Optional[Dict] = None
    patience_count = 0

    for epoch in range(N_EPOCHS):
        model.train()
        epoch_loss = 0.0
        np.random.shuffle(train_dates)
        batches = [train_dates[i:i + BATCH_SIZE]
                   for i in range(0, len(train_dates), BATCH_SIZE)]

        for batch_t in batches:
            optimizer.zero_grad(set_to_none=True)
            batch_loss = torch.zeros(1, device=device)

            for t in batch_t:
                # ── Node features ──────────────────────────────────────────
                node_feats = build_node_features(
                    sig_slice  = sig_stack[t],         # (N, S)
                    ic_slice   = ic_features[t],        # (N, S)
                    vol_arr    = vol_arr[t],
                    skew_arr   = skew_arr[t],
                    kurt_arr   = kurt_arr[t],
                    spy_betas  = spy_betas[t],
                    vix_betas  = vix_betas[t],
                )
                x = torch.tensor(node_feats, dtype=torch.float32, device=device)

                # ── Global features ────────────────────────────────────────
                urgency    = float(regime_df["ltc_urgency"].iloc[t]
                             if "ltc_urgency" in regime_df.columns else 0.3)
                spy_vol    = float(vol_arr[t, spy_idx]) if spy_idx < N_ASSETS else 0.15
                breadth    = float((ret_arr[max(0, t-5):t].sum(0) > 0).mean())
                sig_means  = sig_stack[t].mean(0)      # (S,)
                ic_means   = ic_features[t].mean(0)    # (S,)
                g_feats    = build_global_features(
                    urgency    = urgency,
                    spy_vol    = spy_vol,
                    breadth    = breadth,
                    sig_means  = sig_means,
                    ic_means   = ic_means,
                    t_frac     = t / T,
                )
                g = torch.tensor(g_feats, dtype=torch.float32, device=device).unsqueeze(0)

                # ── Forward ────────────────────────────────────────────────
                weights, predicted_ic = model(x, g, edge_index, edge_attr)

                # Ground-truth forward IC: use NEXT-HORIZON-bar realized IC
                # This is the training TARGET — no look-ahead in inference.
                forward_ic_np = ic_features[min(t + HORIZON, T - 1)]   # (N, S)
                forward_ic    = torch.tensor(forward_ic_np, dtype=torch.float32, device=device)

                loss, _ = signal_router_loss(
                    predicted_ic     = predicted_ic,
                    forward_ic       = forward_ic,
                    blending_weights = weights,
                    lambda_ent       = LAMBDA_ENT,
                    lambda_l2_ic     = LAMBDA_L2_IC,
                )
                batch_loss = batch_loss + loss

            (batch_loss / max(len(batch_t), 1)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            epoch_loss += batch_loss.item()

        scheduler.step()

        # ── Validation IC ──────────────────────────────────────────────────
        model.eval()
        val_ics: List[float] = []
        with torch.no_grad():
            for t in valid_dates[::5]:   # stride-5: full eval in ~O(val/5) passes
                node_feats = build_node_features(
                    sig_slice = sig_stack[t],
                    ic_slice  = ic_features[t],
                    vol_arr   = vol_arr[t],
                    skew_arr  = skew_arr[t],
                    kurt_arr  = kurt_arr[t],
                    spy_betas = spy_betas[t],
                    vix_betas = vix_betas[t],
                )
                x = torch.tensor(node_feats, dtype=torch.float32, device=device)
                g = torch.tensor(
                    build_global_features(
                        urgency   = float(regime_df["ltc_urgency"].iloc[t]
                                    if "ltc_urgency" in regime_df.columns else 0.3),
                        spy_vol   = float(vol_arr[t, spy_idx]),
                        breadth   = float((ret_arr[max(0, t-5):t].sum(0) > 0).mean()),
                        sig_means = sig_stack[t].mean(0),
                        ic_means  = ic_features[t].mean(0),
                        t_frac    = t / T,
                    ),
                    dtype=torch.float32, device=device,
                ).unsqueeze(0)

                weights, _ = model(x, g, edge_index, edge_attr)
                sm = torch.tensor(sig_stack[t], dtype=torch.float32, device=device)
                alpha_np = (weights * sm).sum(-1).cpu().numpy()
                
                # Sincronizamos la validación con las etiquetas de predictibilidad temporal
                fwd_ic_np = ic_features[min(t + HORIZON, T - 1)]   # (N, S)
                
                # Medimos la alineación de los pesos dinámicos con el éxito del factor
                step_ics = []
                for n in range(N_ASSETS):
                    # Producto interno entre las ponderaciones de atención y el IC real del activo
                    score = np.dot(weights[n].cpu().numpy(), fwd_ic_np[n])
                    step_ics.append(score)
                
                val_ics.append(float(np.mean(step_ics)))

        mean_val_ic = float(np.mean(val_ics)) if val_ics else 0.0

        if mean_val_ic > best_val_ic:
            best_val_ic    = mean_val_ic
            best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"  Epoch {epoch+1:>3d}/{N_EPOCHS} | "
                f"train_loss={epoch_loss/max(len(train_dates),1):.4f} | "
                f"val_IC={mean_val_ic:+.4f} | "
                f"best_IC={best_val_ic:+.4f} | "
                f"patience={patience_count}/{PATIENCE} | "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

        if patience_count >= PATIENCE:
            logger.info(f"Early stopping at epoch {epoch+1} (patience={PATIENCE}).")
            break

    # ── Save best checkpoint ───────────────────────────────────────────────────
    _WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    if best_state is not None:
        torch.save(
            {
                "model_state_dict": best_state,
                "signal_names":     SIGNAL_NAMES,   # version-mismatch guard
                "n_signals":        N_SIGNALS,
                "n_assets":         N_ASSETS,
                "node_feat_dim":    NODE_FEAT_DIM,
                "global_feat_dim":  GLOBAL_FEAT_DIM,
                "val_ic":           best_val_ic,
            },
            _WEIGHTS_PATH,
        )
        logger.info(
            f"\n✅ GATv2 weights saved → {_WEIGHTS_PATH} | "
            f"best val IC = {best_val_ic:+.4f}"
        )
        if best_val_ic < 0.015:
            logger.warning(
                "Val IC < 0.015: signal stack may lack sufficient predictive power. "
                "Run validate_signal_ic.py before deploying."
            )
    else:
        logger.error("Training produced no valid checkpoint.")
        sys.exit(1)


if __name__ == "__main__":
    main()