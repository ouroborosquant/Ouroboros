import argparse
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
import numpy as np
import pandas as pd

NY_TZ = ZoneInfo("America/New_York")

def generate_realistic_mnq_data(days: int = 250, base_price: float = 18500.0, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    end_date = datetime.now(NY_TZ).date()
    start_date = end_date - timedelta(days=int(days * 1.5))
    trading_days = pd.bdate_range(start=start_date, end=end_date)[-days:]

    records = []
    current_price = base_price

    for day in trading_days:
        dt_open = datetime.combine(day, time(9, 30), tzinfo=NY_TZ)
        dt_close = datetime.combine(day, time(16, 0), tzinfo=NY_TZ)
        minutes = int((dt_close - dt_open).total_seconds() / 60)

        t_steps = np.linspace(0, 1, minutes)
        # Volatilidad realista del MNQ (sigma ~ 0.025% por minuto -> velas de 4-8 puntos)
        vol_envelope = 1.6 - 0.9 * np.sin(np.pi * t_steps)
        drift = np.random.normal(0.0001, 0.0004)
        returns = np.random.normal(drift / minutes, 0.00028 * vol_envelope)

        for m in range(minutes):
            bar_time = dt_open + timedelta(minutes=m)
            ret = returns[m]
            open_p = current_price
            close_p = open_p * (1.0 + ret)

            noise = abs(np.random.normal(0, 0.00015 * vol_envelope[m]))
            high_p = max(open_p, close_p) * (1.0 + noise)
            low_p = min(open_p, close_p) * (1.0 - noise)

            base_vol = 700 if m > 15 else int(2800 * np.exp(-m / 7.0) + 900)
            vol = int(np.random.gamma(shape=4.0, scale=base_vol / 4.0))

            records.append({
                "timestamp": bar_time,
                "symbol": "MNQ",
                "open": round(open_p, 2),
                "high": round(high_p, 2),
                "low": round(low_p, 2),
                "close": round(close_p, 2),
                "volume": vol,
            })
            current_price = close_p

    return pd.DataFrame(records)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=250)
    args = parser.parse_args()

    Path("data/raw/minute").mkdir(parents=True, exist_ok=True)
    df = generate_realistic_mnq_data(days=args.days)
    output = "data/raw/minute/cme_mnq_1m.parquet"
    df.to_parquet(output)
    print(f"[+] Regenerado dataset con volatilidad real (ATR 1m ~ 5-8 pts): {output} ({len(df):,} barras)")
