import argparse
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

NY_TZ = ZoneInfo("America/New_York")


def generate_synthetic_cme_data(
    symbol: str = "MNQ",
    days: int = 60,
    base_price: float = 18500.0,
    seed: int = 42,
) -> pd.DataFrame:
    np.random.seed(seed)
    end_date = datetime.now(NY_TZ).date()
    start_date = end_date - timedelta(days=int(days * 1.5))

    trading_days = pd.bdate_range(start=start_date, end=end_date)
    trading_days = trading_days[-days:]

    records = []
    current_price = base_price

    for day in trading_days:
        dt_open = datetime.combine(day, datetime.strptime("09:30", "%H:%M").time(), tzinfo=NY_TZ)
        dt_close = datetime.combine(day, datetime.strptime("16:00", "%H:%M").time(), tzinfo=NY_TZ)
        minutes = int((dt_close - dt_open).total_seconds() / 60)

        t_steps = np.linspace(0, 1, minutes)
        u_shaped_vol = 1.8 - 1.2 * np.sin(np.pi * t_steps)

        daily_drift = np.random.normal(0.0001, 0.0005)
        returns = np.random.normal(daily_drift / minutes, (0.0009 * u_shaped_vol))

        for m in range(minutes):
            bar_time = dt_open + timedelta(minutes=m)
            ret = returns[m]
            open_p = current_price
            close_p = open_p * (1.0 + ret)

            noise_high = abs(np.random.normal(0, 0.0004 * u_shaped_vol[m]))
            noise_low = abs(np.random.normal(0, 0.0004 * u_shaped_vol[m]))
            high_p = max(open_p, close_p) * (1.0 + noise_high)
            low_p = min(open_p, close_p) * (1.0 - noise_low)

            base_vol = 600 if m > 15 else int(2500 * np.exp(-m / 8.0) + 700)
            vol = int(np.random.gamma(shape=4.0, scale=base_vol / 4.0))

            records.append({
                "timestamp": bar_time,
                "symbol": symbol,
                "open": round(open_p, 2),
                "high": round(high_p, 2),
                "low": round(low_p, 2),
                "close": round(close_p, 2),
                "volume": vol,
            })
            current_price = close_p

    df = pd.DataFrame(records)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Descargador de datos CME Futures")
    parser.add_argument("--symbol", type=str, default="MNQ", help="Símbolo del contrato")
    parser.add_argument("--days", type=int, default=60, help="Días bursátiles históricos")
    parser.add_argument("--output", type=str, default="data/raw/minute/cme_mnq_1m.parquet")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[*] Generando {args.days} días de barras de 1m para {args.symbol}...")
    df = generate_synthetic_cme_data(symbol=args.symbol, days=args.days)
    df.to_parquet(out_path, engine="pyarrow", index=False)
    print(f"[+] Archivo guardado con éxito: {out_path} ({len(df):,} barras)")


if __name__ == "__main__":
    main()
