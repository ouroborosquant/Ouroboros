import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from pathlib import Path

CACHE = Path("research/outputs/cache")
F8_START, F8_END = "2023-07-01", "2023-12-31"
FWD_HORIZON = 5

alpha_df   = pd.read_parquet(CACHE / "alpha_signals_blended.parquet")
returns_df = pd.read_parquet(CACHE / "returns_wide.parquet")

for df in [alpha_df, returns_df]:
    df.index = pd.to_datetime(df.index).tz_localize(None)

common_cols = alpha_df.columns.intersection(returns_df.columns)
alpha_df    = alpha_df[common_cols]
returns_df  = returns_df[common_cols]

mask        = (alpha_df.index >= F8_START) & (alpha_df.index <= F8_END)
alpha_f8    = alpha_df[mask]
fwd_returns = returns_df[mask].shift(-FWD_HORIZON)

print(f"\nF8 window: {alpha_f8.index[0].date()} → {alpha_f8.index[-1].date()}  ({len(alpha_f8)} days)")
print(f"Tickers in alpha but NOT in returns: {set(alpha_df.columns) - set(returns_df.columns)}")
print(f"\n{'Ticker':<8} {'MeanAlpha':>10} {'FwdRet':>10} {'SpearmanIC':>12} {'n_obs':>7}")
print("-" * 52)

results = {}
for ticker in alpha_f8.columns:
    a = alpha_f8[ticker].dropna()
    r = fwd_returns[ticker].reindex(a.index).dropna()
    a = a.reindex(r.index)
    if len(r) < 10:
        continue
    ic, _ = spearmanr(a.values, r.values)
    results[ticker] = ic
    print(f"{ticker:<8} {a.mean():>10.4f} {r.mean():>10.4f} {ic:>12.4f}  ({len(r)})")

print("\n--- SORTED BY IC (worst to best) ---")
for t, ic in sorted(results.items(), key=lambda x: x[1]):
    print(f"  {t:<8} {ic:+.4f}")

print("\n--- RAW ALPHA SNAPSHOT (first 3 dates in F8) ---")
print(alpha_f8.head(3).T.to_string())

breadth = (alpha_f8 > 0.02).mean(axis=1)
print(f"\nBreadth ratio over F8:  mean={breadth.mean():.3f}  min={breadth.min():.3f}  max={breadth.max():.3f}")
print(f"Days below 0.35:        {(breadth < 0.35).sum()} / {len(breadth)}")

top3 = alpha_f8.apply(lambda row: np.sort(row.values)[-3:].mean(), axis=1)
print(f"Top-3 alpha over F8:    mean={top3.mean():.4f}  min={top3.min():.4f}  max={top3.max():.4f}")
print(f"Days top3 > 0.06:       {(top3 > 0.06).sum()} / {len(top3)}")
