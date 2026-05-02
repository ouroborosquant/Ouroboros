"""
FORTRESS v5 — scripts/train_gat_router.py  [v21 — VTS + SMAX SIGNAL STACK]

v21 Changes
-----------
Updated SIGNAL_NAMES to exactly match the v21 precompute pipeline:
  ["mom", "low_vol", "conc_lead", "vts_lead", "smax_rev"]

CRITICAL: This constant must be byte-for-byte identical to the SIGNAL_NAMES
array in scripts/precompute_alpha_signals.py. Any mismatch causes the signal
stack tensor reconstruction (sig_stack[:, :, s_idx]) to silently mis-assign
signal channels, producing inverted or zero routing weights with no error.

Training targets: forward 21-day Spearman IC between each signal and asset
returns, computed at each node (asset) in the graph. The GATv2 learns to
upweight signals with recent positive IC and downweight signals in IC decay.

Run after:
  1. python scripts/bootstrap_market_data.py   (ensures ^VIX/^VIX3M present)
  2. python scripts/precompute_alpha_signals.py (produces alpha_signals.parquet)
  3. python scripts/validate_signal_ic.py      (confirm vts_lead/smax_rev IC > 0.035)
  4. rm models/weights/gat_router.pt           (purge stale weights from v19)
  5. python scripts/train_gat_router.py        (this script)
"""
import logging
import yaml
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from tqdm import tqdm

from models.alpha.gat_signal_router import (
    SignalRouterGAT, build_economic_graph, build_node_features, signal_router_loss
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("GAT_Trainer")

_CACHE_DIR   = Path("research/outputs/cache")
_WEIGHTS_DIR = Path("models/weights")
_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

with open("config/universe.yaml", "r") as f:
    _univ = yaml.safe_load(f)
TICKERS   = [a["ticker"] for a in _univ["assets"]]
N_ASSETS  = len(TICKERS)

# ── SIGNAL_NAMES must be identical to scripts/precompute_alpha_signals.py ────
# Mismatches here cause silent tensor channel mis-assignment in sig_stack.
SIGNAL_NAMES: list[str] = ["mom", "low_vol", "conc_lead", "vts_lead", "smax_rev"]
N_SIGNALS = len(SIGNAL_NAMES)


def _compute_rolling_ic(
    sig_arr: np.ndarray,    # (T, N)
    ret_arr: np.ndarray,    # (T, N)
    horizon: int = 21,
    window:  int = 63,
) -> np.ndarray:
    """
    Rolling forward-IC tensor for one signal.
    Returns shape (T, N): per-asset Spearman IC estimates.
    Uses a trailing `window`-day rolling average of per-date cross-sectional IC.

    Per-date IC computed only when at least 5 assets have non-zero signal;
    zero-IC days are filled by forward-fill then zero-fill (not excluded from
    the window to avoid look-ahead in the rolling mean).
    """
    T, N = sig_arr.shape
    ic_ts = np.zeros(T, dtype=np.float32)

    for t in range(T - horizon):
        s = sig_arr[t]
        r = ret_arr[t + horizon]
        active = np.isfinite(s) & np.isfinite(r) & (np.abs(s) > 1e-6)
        if active.sum() < 5:
            continue
        ic_val, _ = spearmanr(s[active], r[active])
        if np.isfinite(ic_val):
            ic_ts[t] = float(ic_val)

    # Rolling mean of IC → broadcast to (T, N) per-asset node feature
    ic_roll = pd.Series(ic_ts).rolling(window, min_periods=10).mean().fillna(0.0).values
    return np.broadcast_to(ic_roll[:, np.newaxis], (T, N)).copy().astype(np.float32)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training GAT Signal Router on {device} | signals: {SIGNAL_NAMES}")

    # ── Load cached data ──────────────────────────────────────────────────────
    returns_df = pd.read_parquet(_CACHE_DIR / "returns_wide.parquet").reindex(columns=TICKERS).fillna(0.0)
    regime_df  = pd.read_parquet(_CACHE_DIR / "regime_posteriors.parquet")
    signals_df = pd.read_parquet(_CACHE_DIR / "alpha_signals.parquet")

    returns_df.index = pd.to_datetime(returns_df.index)
    regime_df.index  = pd.to_datetime(regime_df.index)
    signals_df.index = pd.to_datetime(signals_df.index)

    dates      = returns_df.index.intersection(signals_df.index)
    returns_df = returns_df.loc[dates]
    regime_df  = regime_df.reindex(dates).ffill()

    T = len(dates)
    ret_arr = returns_df.values.astype(np.float32)

    # ── Reconstruct signal stack (T, N, S) ────────────────────────────────────
    # Column naming convention in alpha_signals.parquet: "{signal_name}_{ticker}"
    sig_stack = np.zeros((T, N_ASSETS, N_SIGNALS), dtype=np.float32)
    missing_signals = []
    for s_idx, s_name in enumerate(SIGNAL_NAMES):
        cols = [f"{s_name}_{t}" for t in TICKERS]
        present = [c for c in cols if c in signals_df.columns]
        if len(present) < N_ASSETS:
            missing_signals.append(s_name)
            logger.warning(
                f"  Signal '{s_name}': {len(present)}/{N_ASSETS} columns found. "
                f"Missing: {[c for c in cols if c not in signals_df.columns][:5]}..."
            )
        if present:
            sig_stack[:, :len(present), s_idx] = (
                signals_df[present].reindex(dates).fillna(0.0).values.astype(np.float32)
            )

    if missing_signals:
        logger.error(
            f"Signals absent from alpha_signals.parquet: {missing_signals}. "
            f"Re-run scripts/precompute_alpha_signals.py (v21) first."
        )
        raise RuntimeError(f"Missing signal columns for: {missing_signals}")

    logger.info(f"Signal stack assembled: {sig_stack.shape} (T={T}, N={N_ASSETS}, S={N_SIGNALS})")

    # ── Per-signal rolling IC features (T, N) per signal ─────────────────────
    logger.info("Computing per-signal rolling IC features (~2-4 min)...")
    ic_features = np.zeros((T, N_ASSETS, N_SIGNALS), dtype=np.float32)
    for s_idx in range(N_SIGNALS):
        ic_features[:, :, s_idx] = _compute_rolling_ic(
            sig_arr=sig_stack[:, :, s_idx],
            ret_arr=ret_arr,
            horizon=21,
            window=63,
        )
        mean_ic = float(np.abs(ic_features[:, 0, s_idx]).mean())
        logger.info(f"  {SIGNAL_NAMES[s_idx]:12s} rolling IC | mean|IC|={mean_ic:.4f}")

    # ── Build graph structure ─────────────────────────────────────────────────
    logger.info("Building economic correlation graph...")
    edge_index, edge_attr = build_economic_graph(
        returns_df=returns_df,
        tickers=TICKERS,
    )
    edge_index = edge_index.to(device)
    edge_attr  = edge_attr.to(device)

    # ── Model + optimiser ─────────────────────────────────────────────────────
    model = SignalRouterGAT(n_signals=N_SIGNALS).to(device)
    optimiser = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=200, eta_min=1e-5)

    # ── Training loop ─────────────────────────────────────────────────────────
    TRAIN_FRAC   = 0.80
    HORIZON      = 21
    BATCH_SIZE   = 32
    N_EPOCHS     = 200
    GRAD_CLIP    = 1.0

    train_end    = int(T * TRAIN_FRAC)
    valid_dates  = list(range(train_end, T - HORIZON))
    train_dates  = list(range(252, train_end - HORIZON))  # 252-day warm-up for SMAX

    best_val_loss = float("inf")
    best_state    = None

    logger.info(
        f"Train: {len(train_dates)} steps | Val: {len(valid_dates)} steps | "
        f"Epochs: {N_EPOCHS} | Batch: {BATCH_SIZE}"
    )

    for epoch in range(N_EPOCHS):
        model.train()
        epoch_loss = 0.0
        np.random.shuffle(train_dates)
        batches = [train_dates[i:i + BATCH_SIZE] for i in range(0, len(train_dates), BATCH_SIZE)]

        for batch_t in tqdm(batches, desc=f"Epoch {epoch+1}/{N_EPOCHS}", leave=False):
            optimiser.zero_grad()
            batch_loss = torch.tensor(0.0, device=device)

            for t in batch_t:
                # Node features: rolling IC per signal + asset-class one-hot
                node_feats = build_node_features(
                    ic_slice=ic_features[t],           # (N, S)
                    sig_slice=sig_stack[t],             # (N, S)
                    tickers=TICKERS,
                )
                node_feats = torch.tensor(node_feats, dtype=torch.float32, device=device)

                # Regime context scalar
                urgency = float(
                    regime_df["ltc_urgency"].iloc[t]
                    if "ltc_urgency" in regime_df.columns
                    else 0.3
                )
                global_feat = torch.tensor(
                    [urgency, float(t) / T], dtype=torch.float32, device=device
                ).unsqueeze(0)

                # Forward pass: routing weights (N, S)
                routing_weights = model(
                    node_feats, edge_index, edge_attr, global_feat
                )

                # Routing-weighted alpha (N,)
                alpha = (routing_weights * torch.tensor(
                    sig_stack[t], dtype=torch.float32, device=device
                )).sum(dim=-1)

                # Forward returns (N,)
                fwd_ret = torch.tensor(
                    ret_arr[t + HORIZON], dtype=torch.float32, device=device
                )

                # IC-based loss: negative Spearman proxy (rank correlation)
                loss_t = signal_router_loss(alpha, fwd_ret, routing_weights)
                batch_loss = batch_loss + loss_t

            (batch_loss / len(batch_t)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimiser.step()
            epoch_loss += batch_loss.item()

        scheduler.step()

        # ── Validation IC ─────────────────────────────────────────────────────
        model.eval()
        val_ics = []
        with torch.no_grad():
            for t in valid_dates[::5]:    # stride-5 for speed
                node_feats = build_node_features(
                    ic_slice=ic_features[t],
                    sig_slice=sig_stack[t],
                    tickers=TICKERS,
                )
                node_feats = torch.tensor(node_feats, dtype=torch.float32, device=device)
                urgency    = float(
                    regime_df["ltc_urgency"].iloc[t]
                    if "ltc_urgency" in regime_df.columns
                    else 0.3
                )
                global_feat = torch.tensor(
                    [urgency, float(t) / T], dtype=torch.float32, device=device
                ).unsqueeze(0)
                rw    = model(node_feats, edge_index, edge_attr, global_feat)
                alpha = (rw * torch.tensor(sig_stack[t], dtype=torch.float32, device=device)).sum(-1)
                fwd   = ret_arr[t + HORIZON]

                alpha_np = alpha.cpu().numpy()
                active   = np.isfinite(fwd) & (np.abs(alpha_np) > 1e-6)
                if active.sum() < 5:
                    continue
                ic, _ = spearmanr(alpha_np[active], fwd[active])
                if np.isfinite(ic):
                    val_ics.append(ic)

        mean_val_ic  = float(np.mean(val_ics)) if val_ics else 0.0
        mean_val_loss = -mean_val_ic           # loss = negative IC

        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 20 == 0 or epoch == 0:
            logger.info(
                f"  Epoch {epoch+1:>3d}/{N_EPOCHS} | "
                f"train_loss={epoch_loss/max(len(train_dates),1):.4f} | "
                f"val_IC={mean_val_ic:+.4f} | "
                f"best_val_IC={-best_val_loss:+.4f} | "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

    # ── Save best weights ─────────────────────────────────────────────────────
    weights_path = _WEIGHTS_DIR / "gat_router.pt"
    if best_state is not None:
        torch.save(
            {
                "model_state_dict": best_state,
                "signal_names":     SIGNAL_NAMES,    # stored for version-mismatch detection
                "n_signals":        N_SIGNALS,
                "n_assets":         N_ASSETS,
                "val_ic":           -best_val_loss,
            },
            weights_path,
        )
        logger.info(
            f"✅ GATv2 weights saved → {weights_path} | "
            f"best val IC = {-best_val_loss:+.4f}"
        )
    else:
        logger.error("Training produced no valid state. Weights not saved.")


if __name__ == "__main__":
    main()