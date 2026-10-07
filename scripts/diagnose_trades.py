import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
import pandas as pd
from src.alpha.trend_continuation import ORBTrendContinuationStrategy
from src.backtest.engine import BacktestEngine
from src.backtest.fee_models import CMEFeeModel
from src.backtest.fill_simulator import ManagementConfig
from src.risk.position_sizer import InstrumentSpecs
from src.core.events import BarEvent

parser = argparse.ArgumentParser(description="Diagnóstico de trades, señales y salidas")
parser.add_argument("--one-lot-threshold", type=float, default=None)
parser.add_argument("--lock-trigger-rr", type=float, default=None)
parser.add_argument("--lock-giveback-rr", type=float, default=0.75)
args = parser.parse_args()

with open("configs/prop_firm/evaluation_50k.yaml") as f:
    prop_cfg = yaml.safe_load(f)
with open("configs/strategy/orb_base.yaml") as f:
    strat_cfg = yaml.safe_load(f)

specs = InstrumentSpecs(
    symbol="MNQ",
    tick_size=0.25,
    point_value=2.0,
    max_contracts=prop_cfg["limits"]["max_open_contracts"].get("MNQ", 6),
)
fee_model = CMEFeeModel(commission_per_contract=0.62, slippage_ticks=1.0)

strat = ORBTrendContinuationStrategy(
    symbol="MNQ",
    buffer_ticks=strat_cfg["opening_range"]["buffer_ticks"],
    retest_tolerance_ticks=12,
    min_rvol_threshold=strat_cfg["entry_filters"].get("min_rvol_threshold", 0.8),
    stop_atr_multiplier=strat_cfg["exit_rules"]["stop_loss"]["stop_atr_multiplier"],
    risk_reward_ratio=strat_cfg["exit_rules"]["take_profit"]["risk_reward_ratio"],
    hard_stop_max_ticks=strat_cfg["exit_rules"]["stop_loss"]["hard_stop_max_ticks"],
    stop_min_ticks=strat_cfg["exit_rules"]["stop_loss"].get("min_stop_ticks", 40),
    max_wait_bars=strat_cfg["entry_filters"].get("max_retest_bars", 25),
    allow_vwap_retest=strat_cfg["entry_filters"].get("allow_vwap_retest", False),
    rvol_min_history=strat_cfg["entry_filters"].get("rvol_min_history", 2),
)

mgmt = ManagementConfig(
    lock_trigger_rr=args.lock_trigger_rr,
    lock_giveback_rr=args.lock_giveback_rr,
)

engine = BacktestEngine(
    strategy=strat,
    specs=specs,
    fee_model=fee_model,
    initial_capital=prop_cfg["initial_balance"],
    max_trailing_drawdown=prop_cfg["limits"]["max_trailing_drawdown"],
    max_daily_loss=prop_cfg["limits"]["max_daily_loss_limit"],
    debug=False,
    max_cost_ratio=strat_cfg["exit_rules"]["stop_loss"].get("max_cost_ratio", 0.12),
    drawdown_mode="peak_unrealized",
    lock_offset=100.0,
    stop_on_breach=True,
    one_lot_threshold=args.one_lot_threshold,
    management=mgmt,
)

df_raw = pd.read_parquet("data/raw/minute/cme_mnq_1m.parquet").sort_values("timestamp").reset_index(drop=True)
bars = [
    BarEvent(
        timestamp=row["timestamp"],
        symbol=row["symbol"],
        timeframe="1m",
        open=row["open"],
        high=row["high"],
        low=row["low"],
        close=row["close"],
        volume=float(row["volume"]),
    )
    for _, row in df_raw.iterrows()
]

res = engine.run(bars)

print(f"\n=================================================================")
print(f" DIAGNÓSTICO (one_lot={args.one_lot_threshold}, lock_trig={args.lock_trigger_rr})")
print(f"=================================================================")
print("\n### SUMMARY BY EXIT REASON")
reasons = {}
for t in res.trades:
    reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
for k, v in reasons.items():
    print(f"- {k}: {v}")

print("\n### DETAILED TRADE LOG")
print("| # | Entry Time | Dir | Qty | Entry | Exit | Net PnL | R | MFE (R) | MAE (R) | Bars | Exit Reason |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for i, t in enumerate(res.trades, 1):
    r_mult = getattr(t, "r_multiple", 0.0)
    mfe_r = getattr(t, "mfe_r", getattr(t, "mfe_dollars", 0.0))
    mae_r = getattr(t, "mae_r", getattr(t, "mae_dollars", 0.0))
    bars_held = getattr(t, "bars_held", "N/A")
    entry_str = str(t.entry_time)[:16]
    direction_name = t.direction.name if hasattr(t.direction, "name") else str(t.direction)
    print(f"| {i} | {entry_str} | {direction_name} | {t.contracts} | {t.entry_price:.2f} | {t.exit_price:.2f} | ${t.net_pnl:.2f} | {r_mult:+.2f}R | {mfe_r:.2f} | {mae_r:.2f} | {bars_held} | {t.exit_reason} |")

diagnostics = getattr(res, "diagnostics", getattr(engine, "diagnostics", None))
if diagnostics and "signal_log" in diagnostics:
    print("\n### SIGNAL TELEMETRY (Timestamp, RR, Stop Ticks, ATR)")
    for s in diagnostics["signal_log"]:
        print(f"- {s}")
