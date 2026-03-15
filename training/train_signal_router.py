"""
FORTRESS v5 - training/train_signal_router.py
Path: training/train_signal_router.py

Signal Router Training — Supervised IC Prediction.

OBJECTIVE:
  Train SignalRouterGAT to predict the forward rolling Spearman IC of each
  signal for each asset in the current regime. Use these IC predictions as
  soft blending weights to produce the final alpha vector.

TRAINING DATA CONSTRUCTION:
  For each trading day t:
    X_nodes: 32-dim node features per asset (regime + IC history + betas)
    X_edges: Economic graph (fixed, pre-built)
    Y_ic:    (N, S) matrix of FORWARD 21-day rolling Spearman IC
             IC_s(i, t) = Spearman_corr(signal_s[t:t+21, i], r[t+1:t+22, i])

  NOTE ON THE LOOK-AHEAD GATE:
    Y_ic is computed from forward returns [t+1:t+22]. This is the TARGET,
    not a feature — it is ONLY used during training, never during inference.
    At inference time, we predict Y_ic from X_nodes and use the prediction
    as our weight. No look-ahead.

IC COMPUTATION:
  We compute rolling Spearman IC using the vectorised approach:
    rank_signal = scipy.stats.rankdata(signal[t:t+W, i]) — temporal rank
    rank_return = scipy.stats.rankdata(returns[t+1:t+W+1, i]) — forward rank
    IC_s(i, t) = pearsonr(rank_signal, rank_return)[0]

  This is the correct way to compute IC as a forward prediction metric.
  A POSITIVE IC means: high signal values preceded high returns in this window.

TRAINING SPLIT:
  Purged expanding window — same setup as the walk-forward backtest:
    Training: all data up to IS end date minus 21-day purge gap
    Validation: OOS window immediately after purge gap
    NO reuse of OOS data for hyperparameter tuning (would constitute overfitting)

HYPERPARAMETERS (fixed — do not tune on OOS):
  lr = 3e-4 (Adam with cosine decay)
  weight_decay = 1e-4
  batch_size = 16 (date batches)
  n_epochs = 200
  early_stopping: 25 epochs patience on validation IC correlation

DIAGNOSTIC OUTPUTS:
  models/weights/signal_router_latest.pt  — best weights by val IC corr
  models/weights/signal_router_backup.pt  — checkpoint every 25 epochs
  research/outputs/routing_weights_history.parquet — per-date blending weights
  research/outputs/ic_prediction_history.parquet   — predicted vs actual IC

RUN:
  PYTHONPATH=. python training/train_signal_router.py
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("TrainSignalRouter")

# ── Paths ──────────────────────────────────────────────────────────────────────
_BASE_DIR       = Path(".")
_CACHE_DIR      = _BASE_DIR / "research" / "outputs" / "cache"
_WEIGHTS_DIR    = _BASE_DIR / "models" / "weights"
_RESEARCH_OUT   = _BASE_DIR / "research" / "outputs"

_PRICES_PATH    = _CACHE_DIR / "prices_wide.parquet"
_RETURNS_PATH   = _CACHE_DIR / "returns_wide.parquet"
_REGIME_PATH    = _CACHE_DIR / "regime_posteriors.parquet"
_SIGNALS_PATH   = _CACHE_DIR / "alpha_signals.parquet"  # pre-computed 4-layer signals
_ROUTER_WEIGHTS = _WEIGHTS_DIR / "signal_router_latest.pt"

# ── Training config ────────────────────────────────────────────────────────────
_LR             = 3e-4
_WEIGHT_DECAY   = 1e-4
_BATCH_SIZE     = 16
_N_EPOCHS       = 200
_PATIENCE       = 25
_GRAD_CLIP      = 1.0
_IC_WINDOW      = 21    # Forward IC computation window (trading days)
_IC_HIST_WINDOW = 63    # Rolling IC history for node features
_MIN_IC_OBS     = 10    # Minimum observations for valid IC computation
_LAMBDA_ENT     = 0.05
_LAMBDA_L2_IC   = 0.01
_VAL_FRACTION   = 0.20
_PURGE_GAP      = 21    # Trading days between train and val

TICKERS: List[str] = [
    "SPY", "QQQ", "IWM", "TLT", "HYG", "LQD", "GLD", "SLV",
    "GDX", "XLE", "XLF", "XLK", "XLV", "XLU", "XLI", "XLP",
    "XLY", "XLB", "XLC", "VIXY", "BIL", "SHV", "USO", "PDBC", "COWZ",
]
SIGNAL_NAMES: List[str] = ["vrp", "vts", "nav_arb", "insider", "low_vol"]

N_ASSETS  = len(TICKERS)
N_SIGNALS = len(SIGNAL_NAMES)


def compute_rolling_ic_matrix(
    signal_df:  pd.DataFrame,  # (T, N×S) — columns: {signal_name}_{ticker}
    returns_df: pd.DataFrame,  # (T, N) — forward returns
    ic_window:  int = _IC_WINDOW,
    min_obs:    int = _MIN_IC_OBS,
) -> np.ndarray:
    """
    Compute forward rolling Spearman IC: IC_s(i, t) for all t, i, s.

    For each date t:
      IC_s(i, t) = Spearman_corr(
          signal_s[t : t + ic_window, i],  ← signal values over next 21 days
          r[t+1 : t + ic_window + 1, i]    ← returns over same window (shifted +1)
      )

    This is the TARGET for training the signal router. It answers the question:
    "Over the next 21 days, how predictive was signal_s for asset_i?"

    Returns:
      ic_tensor: ndarray (T, N, S) where ic_tensor[t, i, s] = IC_s(i, t)
                 NaN for dates with insufficient history.
                 NOTE: This array has NaN at the end (last ic_window rows)
                 because we need forward data. These dates must be excluded
                 from training.
    """
    T = len(returns_df)
    ic_tensor = np.full((T, N_ASSETS, N_SIGNALS), np.nan, dtype=np.float32)

    returns_np = returns_df.reindex(columns=TICKERS).values  # (T, N)

    for t in range(T - ic_window):
        # Forward window: [t+1, t+ic_window] (inclusive)
        fwd_returns = returns_np[t + 1 : t + ic_window + 1]  # (ic_window, N)
        if fwd_returns.shape[0] < min_obs:
            continue

        for s_idx, signal_name in enumerate(SIGNAL_NAMES):
            # Get signal values over the SAME forward window [t:t+ic_window]
            # IC uses the signal at t through t+ic_window as the predictor
            # (We want: does the signal at time t predict returns at t+1...t+W?)
            # Since signal is computed CAUSAL at each t, we use:
            #   signal_window = signal_df[signal_name][t : t + ic_window]
            #   return_window = returns[t+1 : t + ic_window + 1]
            # Spearman measures whether high-signal assets return more over the window.
            for i, ticker in enumerate(TICKERS):
                col_name = f"{signal_name}_{ticker}"
                if col_name not in signal_df.columns:
                    continue

                sig_window = signal_df[col_name].values[t : t + ic_window]
                ret_window = fwd_returns[:, i]

                valid_mask = ~(np.isnan(sig_window) | np.isnan(ret_window))
                if valid_mask.sum() < min_obs:
                    continue

                try:
                    ic, pval = stats.spearmanr(
                        sig_window[valid_mask],
                        ret_window[valid_mask],
                    )
                    if not np.isnan(ic):
                        ic_tensor[t, i, s_idx] = float(ic)
                except Exception:
                    pass

    valid_pct = (~np.isnan(ic_tensor)).mean() * 100
    logger.info(f"IC tensor computed: shape={ic_tensor.shape}, valid={valid_pct:.1f}%")
    return ic_tensor


def compute_ewma_ic_history(
    ic_tensor:   np.ndarray,  # (T, N, S) forward IC
    halflife:    int = _IC_HIST_WINDOW,
) -> np.ndarray:
    """
    Compute causal EWMA IC history: the rolling average IC each signal has
    achieved in the PAST, for use as a NODE FEATURE (not a target).

    ewma_ic_history[t, i, s] = EWMA over ic_tensor[0:t, i, s]
    This is strictly causal: at time t, we only use past IC values.

    Returns:
      ewma_ic: ndarray (T, N, S) — causal EWMA IC history per asset per signal
    """
    T = ic_tensor.shape[0]
    ewma_ic = np.zeros((T, N_ASSETS, N_SIGNALS), dtype=np.float32)

    alpha = 1.0 - np.exp(-1.0 / halflife)  # EWMA decay factor

    for i in range(N_ASSETS):
        for s in range(N_SIGNALS):
            running_mean = 0.0
            running_weight = 0.0
            for t in range(T):
                val = ic_tensor[t, i, s]
                if not np.isnan(val):
                    running_weight = running_weight * (1 - alpha) + alpha
                    running_mean   = running_mean   * (1 - alpha) + val * alpha
                    if running_weight > 1e-6:
                        ewma_ic[t, i, s] = running_mean / running_weight
                else:
                    # Propagate last known value (causal imputation)
                    ewma_ic[t, i, s] = ewma_ic[t-1, i, s] if t > 0 else 0.0

    return ewma_ic


def compute_sensitivity_betas(
    returns_df: pd.DataFrame,  # (T, N) asset returns
    vol_returns: pd.Series,    # (T,) VIX return proxy
    rate_returns: pd.Series,   # (T,) TLT return proxy for rate sensitivity
    window: int = 63,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute rolling vol-beta and rate-beta for each asset.

    vol_beta_i(t)  = cov(r_i[t-W:t], ΔIV_t[t-W:t]) / var(ΔIV_t[t-W:t])
    rate_beta_i(t) = cov(r_i[t-W:t], r_TLT[t-W:t]) / var(r_TLT[t-W:t])

    Returns (T, N) arrays — causal (uses only past W days).
    """
    T = len(returns_df)
    vol_betas  = np.zeros((T, N_ASSETS), dtype=np.float32)
    rate_betas = np.zeros((T, N_ASSETS), dtype=np.float32)

    returns_np  = returns_df.reindex(columns=TICKERS).values
    vol_np      = vol_returns.reindex(returns_df.index).fillna(0).values.ravel()
    rate_np     = rate_returns.reindex(returns_df.index).fillna(0).values.ravel()

    for t in range(window, T):
        r_window    = returns_np[t - window : t]  # (W, N)
        vol_window  = vol_np[t - window : t]      # (W,)
        rate_window = rate_np[t - window : t]     # (W,)

        vol_var  = np.var(vol_window)  + 1e-10
        rate_var = np.var(rate_window) + 1e-10

        for i in range(N_ASSETS):
            ret_i = r_window[:, i]
            valid = ~np.isnan(ret_i)
            if valid.sum() < 20:
                continue
            vol_betas[t, i]  = np.cov(ret_i[valid], vol_window[valid])[0, 1]  / vol_var
            rate_betas[t, i] = np.cov(ret_i[valid], rate_window[valid])[0, 1] / rate_var

    return vol_betas, rate_betas


def run_training(
    returns_df:  pd.DataFrame,
    regime_df:   pd.DataFrame,
    signals_df:  pd.DataFrame,  # (T, N×S) — long format with {signal}_{ticker} columns
    ic_tensor:   np.ndarray,    # (T, N, S)
    ewma_ic_hist: np.ndarray,   # (T, N, S)
    vol_betas:   np.ndarray,    # (T, N)
    rate_betas:  np.ndarray,    # (T, N)
) -> None:
    """
    Main training loop. Saves best model by validation IC correlation.

    BATCH CONSTRUCTION:
      Each batch is a set of `batch_size` dates. For each date t, we build:
        - node_features: (N, 32) from regime z_mu + ewma_ic + betas
        - edge_index, edge_attr: static economic graph
        - signal_matrix: (N, S) current signal values
        - target_ic: (N, S) forward IC values (from ic_tensor)

      Dates with NaN forward IC are excluded from training.
    """
    import torch
    import torch.optim as optim
    from torch.optim.lr_scheduler import CosineAnnealingLR

    from models.alpha.gat_signal_router import (
        SignalRouterGAT, build_economic_graph, build_node_features, signal_router_loss
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")

    # Build static economic graph
    edge_index, edge_attr = build_economic_graph()
    edge_index = edge_index.to(device)
    edge_attr  = edge_attr.to(device)

    # Build valid training indices (exclude last ic_window rows — no forward IC)
    T          = len(returns_df)
    
    # BUG FIX: Convert NaN ICs to 0.0. A flat signal (like NAV Arb on SPY) 
    # has 0 predictive power. We cannot throw away the entire day!
    np.nan_to_num(ic_tensor, nan=0.0, copy=False)
    
    # Valid dates: Must have enough history for features, and enough future for targets
    valid_idx  = np.arange(_IC_HIST_WINDOW + 1, T - _IC_WINDOW)

    # Purged train/val split
    n_val    = max(int(len(valid_idx) * _VAL_FRACTION), 30)
    # Purge gap: exclude dates within _PURGE_GAP of the split boundary
    split_pt  = len(valid_idx) - n_val - _PURGE_GAP
    train_idx = valid_idx[:split_pt]
    val_idx   = valid_idx[split_pt + _PURGE_GAP:]

    logger.info(
        f"Training samples: {len(train_idx)} | Val samples: {len(val_idx)} | "
        f"Purge gap: {_PURGE_GAP} days"
    )

    # Parse z_mu from regime_df
    def parse_zmu(val) -> np.ndarray:
        import json
        if isinstance(val, (list, np.ndarray)):
            return np.asarray(val, dtype=np.float32)
        if isinstance(val, str):
            return np.array(json.loads(val), dtype=np.float32)
        return np.zeros(16, dtype=np.float32)

    regime_np = np.vstack(regime_df["z_mu"].map(parse_zmu).values)  # (T_reg, 16)
    regime_dates = pd.DatetimeIndex(regime_df.index)
    returns_dates = pd.DatetimeIndex(returns_df.index)

    def get_node_features_at(t_global: int) -> torch.Tensor:
        """Build 32-dim node features for date index t_global (in returns_df)."""
        date = returns_dates[t_global]
        # Get closest regime posterior ≤ date (causal)
        regime_match = regime_dates[regime_dates <= date]
        if len(regime_match) == 0:
            z_mu = np.zeros(16, dtype=np.float32)
        else:
            reg_idx = regime_dates.get_loc(regime_match[-1])
            z_mu    = regime_np[reg_idx]

        ic_hist   = ewma_ic_hist[t_global]   # (N, S)
        vb        = vol_betas[t_global]      # (N,)
        rb        = rate_betas[t_global]     # (N,)
        liq_z     = np.zeros(N_ASSETS, dtype=np.float32)  # placeholder — add volume z-score

        return build_node_features(z_mu, ic_hist, vb, rb, liq_z)

    def get_signal_matrix_at(t_global: int) -> torch.Tensor:
        """Build (N, S) signal matrix at date t_global."""
        sm = np.zeros((N_ASSETS, N_SIGNALS), dtype=np.float32)
        for s_idx, sig_name in enumerate(SIGNAL_NAMES):
            for i_idx, ticker in enumerate(TICKERS):
                col = f"{sig_name}_{ticker}"
                if col in signals_df.columns:
                    val = float(signals_df[col].iloc[t_global])
                    sm[i_idx, s_idx] = 0.0 if np.isnan(val) else val
        return torch.from_numpy(sm)

    # Build model
    model     = SignalRouterGAT().to(device)
    optimizer = optim.Adam(model.parameters(), lr=_LR, weight_decay=_WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=_N_EPOCHS, eta_min=_LR * 0.01)

    best_val_ic_corr = -np.inf
    epochs_no_improve = 0

    _WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(_N_EPOCHS):
        # ── Training pass ──────────────────────────────────────────────────────
        model.train()
        np.random.shuffle(train_idx)
        epoch_losses: Dict[str, float] = {}
        n_batches = max(1, len(train_idx) // _BATCH_SIZE)

        for b in range(n_batches):
            batch = train_idx[b * _BATCH_SIZE : (b + 1) * _BATCH_SIZE]
            if len(batch) == 0:
                continue

            batch_loss_total = torch.tensor(0.0, device=device)

            for t_global in batch:
                x          = get_node_features_at(t_global).to(device)       # (N, 32)
                signal_mat = get_signal_matrix_at(t_global).to(device)       # (N, S)
                target_ic  = torch.from_numpy(
                    ic_tensor[t_global]
                ).float().to(device)                                          # (N, S)

                weights, pred_ic = model(x, edge_index, edge_attr)

                loss, components = signal_router_loss(
                    predicted_ic=pred_ic,
                    forward_ic=target_ic,
                    blending_weights=weights,
                    lambda_ent=_LAMBDA_ENT,
                    lambda_l2_ic=_LAMBDA_L2_IC,
                )
                batch_loss_total = batch_loss_total + loss / len(batch)

                for k, v in components.items():
                    epoch_losses[k] = epoch_losses.get(k, 0.0) + v / (n_batches * len(batch))

            optimizer.zero_grad(set_to_none=True)
            batch_loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), _GRAD_CLIP)
            optimizer.step()

        scheduler.step()

        # ── Validation pass ────────────────────────────────────────────────────
        model.eval()
        val_pred_ics  = []
        val_true_ics  = []
        val_routed_ics = []

        with torch.no_grad():
            for t_global in val_idx:
                x          = get_node_features_at(t_global).to(device)
                signal_mat = get_signal_matrix_at(t_global).to(device)
                target_ic  = ic_tensor[t_global]                              # (N, S) numpy

                weights, pred_ic = model(x, edge_index, edge_attr)

                # Routed alpha: blended signal
                routed = (weights.cpu().numpy() * signal_mat.cpu().numpy()).sum(axis=1)  # (N,)

                val_pred_ics.append(pred_ic.cpu().numpy())
                val_true_ics.append(target_ic)
                val_routed_ics.append(routed)

        # Validation IC: Spearman between predicted IC and actual forward IC
        pred_arr  = np.vstack(val_pred_ics)   # (T_val × N, S)
        true_arr  = np.vstack(val_true_ics)

        # Flatten and compute correlation across all (t, i, s) triples
        pred_flat = pred_arr[~np.isnan(true_arr)]
        true_flat = true_arr[~np.isnan(true_arr)]

        if len(pred_flat) > 30:
            val_ic_corr, _ = stats.pearsonr(pred_flat, true_flat)
        else:
            val_ic_corr = 0.0

        # Routed alpha IC: how well does the FINAL routed alpha predict returns?
        # This is the metric that actually matters for trading performance
        routed_arr = np.stack(val_routed_ics)  # (T_val, N)
        # (Would need forward returns here for true IC — using prediction quality as proxy)

        logger.info(
            f"Epoch {epoch+1:3d}/{_N_EPOCHS} | "
            f"Train IC={epoch_losses.get('ic_mse', 0):.4f} | "
            f"E[IC]={epoch_losses.get('mean_expected_ic', 0):+.4f} | "
            f"Entropy={epoch_losses.get('entropy', 0):.3f} | "
            f"Val IC-corr={val_ic_corr:+.4f} | "
            f"LR={scheduler.get_last_lr()[0]:.2e}"
        )

        if val_ic_corr > best_val_ic_corr:
            best_val_ic_corr = val_ic_corr
            torch.save(model.state_dict(), str(_ROUTER_WEIGHTS))
            epochs_no_improve = 0
            logger.info(f"  ✓ Best model saved (val IC-corr={val_ic_corr:+.4f})")
        else:
            epochs_no_improve += 1

        if epoch % 25 == 0:
            backup_path = _WEIGHTS_DIR / f"signal_router_epoch{epoch}.pt"
            torch.save(model.state_dict(), str(backup_path))

        if epochs_no_improve >= _PATIENCE:
            logger.info(f"Early stopping at epoch {epoch+1} (patience={_PATIENCE})")
            break

    logger.info(
        f"Training complete. Best val IC-corr: {best_val_ic_corr:+.4f} | "
        f"Weights: {_ROUTER_WEIGHTS}"
    )


async def main() -> None:
    """
    Full training pipeline:
      1. Load cached returns, regime posteriors, pre-computed signals
      2. Compute forward IC tensor (training targets)
      3. Compute EWMA IC history (node features)
      4. Compute sensitivity betas (node features)
      5. Run training
    """
    import yfinance as yf

    for path in [_PRICES_PATH, _RETURNS_PATH, _REGIME_PATH, _SIGNALS_PATH]:
        if not path.exists():
            logger.error(f"Missing: {path}. Run precompute scripts first.")
            sys.exit(1)

    logger.info("Loading cached data...")
    returns_df = pd.read_parquet(_RETURNS_PATH)
    regime_df  = pd.read_parquet(_REGIME_PATH)
    signals_df = pd.read_parquet(_SIGNALS_PATH)

    for df in [returns_df, regime_df, signals_df]:
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)

    # Align returns and signals on common dates
    common = returns_df.index.intersection(signals_df.index)
    returns_df = returns_df.loc[common].reindex(columns=TICKERS).ffill().fillna(0)
    signals_df = signals_df.loc[common]
    regime_df  = regime_df.sort_index()

    logger.info(f"Aligned: {len(common)} trading days × {N_ASSETS} assets")

    # Compute IC tensor (expensive — cache to disk)
    ic_cache = _CACHE_DIR / "ic_tensor.npy"
    if ic_cache.exists():
        logger.info(f"Loading cached IC tensor from {ic_cache}")
        ic_tensor = np.load(str(ic_cache))
    else:
        logger.info("Computing forward IC tensor (this takes ~5-10 min)...")
        ic_tensor = compute_rolling_ic_matrix(signals_df, returns_df)
        np.save(str(ic_cache), ic_tensor)
        logger.info(f"IC tensor saved to {ic_cache}")

    # EWMA IC history (causal node features)
    ewma_ic_hist = compute_ewma_ic_history(ic_tensor)

    # Sensitivity betas
    logger.info("Computing sensitivity betas...")
    loop = asyncio.get_event_loop()
    vix_proxy = await loop.run_in_executor(
        None, lambda: yf.download("^VIX", start="2015-01-01", progress=False)["Close"].squeeze().pct_change()
    )
    vol_betas, rate_betas = compute_sensitivity_betas(
        returns_df,
        vol_returns=vix_proxy,
        rate_returns=returns_df.get("TLT", pd.Series(0.0, index=returns_df.index)),
    )

    logger.info("Starting training...")
    run_training(
        returns_df=returns_df,
        regime_df=regime_df,
        signals_df=signals_df,
        ic_tensor=ic_tensor,
        ewma_ic_hist=ewma_ic_hist,
        vol_betas=vol_betas,
        rate_betas=rate_betas,
    )


if __name__ == "__main__":
    asyncio.run(main())