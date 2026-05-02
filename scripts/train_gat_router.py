"""
FORTRESS v5 - scripts/train_gat_router.py
Trains the SignalRouterGAT to predict forward 21-day Information Coefficients.
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

_CACHE_DIR = Path("research/outputs/cache")
_WEIGHTS_DIR = Path("models/weights")
_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

with open("config/universe.yaml", "r") as f:
    _univ = yaml.safe_load(f)
TICKERS = [a["ticker"] for a in _univ["assets"]]
N_ASSETS = len(TICKERS)
SIGNAL_NAMES = ["mom", "low_vol", "conc_lead", "night_effect", "pca_statarb"]
N_SIGNALS = len(SIGNAL_NAMES)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training GAT Signal Router on {device}...")

    # 1. Load Data
    returns_df = pd.read_parquet(_CACHE_DIR / "returns_wide.parquet").reindex(columns=TICKERS).fillna(0.0)
    regime_df = pd.read_parquet(_CACHE_DIR / "regime_posteriors.parquet")
    signals_df = pd.read_parquet(_CACHE_DIR / "alpha_signals.parquet")

    dates = returns_df.index.intersection(signals_df.index)
    returns_df = returns_df.loc[dates]
    regime_df = regime_df.reindex(dates).ffill()
    
    T = len(dates)
    ret_arr = returns_df.values
    
    # Reconstruct signal stack (T, N, S)
    sig_stack = np.zeros((T, N_ASSETS, N_SIGNALS), dtype=np.float32)
    for s_idx, s_name in enumerate(SIGNAL_NAMES):
        cols = [f"{s_name}_{t}" for t in TICKERS]
        sig_stack[:, :, s_idx] = signals_df[cols].reindex(dates).fillna(0.0).values

    # 2. Build Tensors (Trailing ICs and Forward IC Targets)
    logger.info("Computing historical and forward ICs (this takes a minute)...")
    trailing_ic = np.zeros((T, N_ASSETS, N_SIGNALS), dtype=np.float32)
    forward_ic  = np.zeros((T, N_ASSETS, N_SIGNALS), dtype=np.float32)

    for s_idx in range(N_SIGNALS):
        sig_arr = sig_stack[:, :, s_idx]
        for t_idx in tqdm(range(63, T - 21), desc=f"IC Calc {SIGNAL_NAMES[s_idx]}"):
            for n_idx in range(N_ASSETS):
                # Trailing 63d IC (Input)
                s_trail = sig_arr[t_idx-63:t_idx, n_idx]
                r_trail = ret_arr[t_idx-63:t_idx, n_idx]
                if np.std(s_trail) > 1e-5 and np.std(r_trail) > 1e-5:
                    ic, _ = spearmanr(s_trail, r_trail)
                    trailing_ic[t_idx, n_idx, s_idx] = ic if np.isfinite(ic) else 0.0
                
                # Forward 21d IC (Target)
                s_fwd = sig_arr[t_idx:t_idx+21, n_idx]
                r_fwd = ret_arr[t_idx:t_idx+21, n_idx]
                if np.std(s_fwd) > 1e-5 and np.std(r_fwd) > 1e-5:
                    ic, _ = spearmanr(s_fwd, r_fwd)
                    forward_ic[t_idx, n_idx, s_idx] = ic if np.isfinite(ic) else 0.0

    # Proxies for Betas
    vixy_col = returns_df["VIXY"] if "VIXY" in TICKERS else returns_df.mean(axis=1)
    tlt_col = returns_df["TLT"] if "TLT" in TICKERS else returns_df.mean(axis=1)
    
    vol_betas = returns_df.rolling(63).corr(vixy_col).fillna(0.0).values
    rate_betas = returns_df.rolling(63).corr(tlt_col).fillna(0.0).values
    liquidity_z = np.zeros((T, N_ASSETS), dtype=np.float32) # Requires volume data, stubbed for now

    # 3. Graph Topology
    edge_index, edge_attr = build_economic_graph(returns_df)
    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)

    # 4. Model Initialization
    model = SignalRouterGAT(n_signals=N_SIGNALS).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # 5. Training Loop
    logger.info("Initiating Training Loop...")
    model.train()
    epochs = 15
    
    # Train on first 80% of data
    train_end = int(T * 0.8)

    for epoch in range(epochs):
        epoch_loss = 0.0
        valid_steps = 0
        
        for t_idx in range(63, train_end - 21):
            # Parse Regime
            z_mu_str = regime_df.iloc[t_idx]["z_mu"]
            if isinstance(z_mu_str, str):
                import ast; z_mu = np.array(ast.literal_eval(z_mu_str), dtype=np.float32)
            else:
                z_mu = np.zeros(16, dtype=np.float32)

            # Build Node Features
            x_np = build_node_features(
                signal_ic_history=trailing_ic[t_idx],
                vol_betas=vol_betas[t_idx],
                rate_betas=rate_betas[t_idx],
                liquidity_z=liquidity_z[t_idx]
            )
            
            x = x_np.to(device)
            g = torch.from_numpy(z_mu[:16]).to(device)
            y = torch.from_numpy(forward_ic[t_idx]).to(device)

            optimizer.zero_grad()
            weights, pred_ic = model(x, g, edge_index, edge_attr)
            
            loss, comps = signal_router_loss(pred_ic, y, weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += comps["total"]
            valid_steps += 1
            
        logger.info(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/valid_steps:.4f}")

    save_path = _WEIGHTS_DIR / "gat_router.pt"
    torch.save(model.state_dict(), save_path)
    logger.info(f"✅ Training Complete. Model saved to {save_path}")

if __name__ == "__main__":
    main()