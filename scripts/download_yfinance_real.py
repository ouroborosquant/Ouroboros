from datetime import datetime, timedelta
from pathlib import Path
import yfinance as yf
import pandas as pd
from rich.console import Console

console = Console()

console.print("[bold cyan][*] Descargando histórico intradía real de CME NQ=F (últimos 30 días, 1m)...[/bold cyan]")
# yfinance permite hasta 30 días de barras de 1m en futuros CME
ticker = yf.Ticker("NQ=F")
df = ticker.history(period="1mo", interval="1m")

df = df.reset_index()
time_col = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()][0]

clean_df = pd.DataFrame({
    "timestamp": pd.to_datetime(df[time_col], utc=True),
    "symbol": "MNQ",
    "open": df["Open"].round(2),
    "high": df["High"].round(2),
    "low": df["Low"].round(2),
    "close": df["Close"].round(2),
    "volume": df["Volume"].astype(float),
})

clean_df = clean_df.dropna().sort_values("timestamp").reset_index(drop=True)
output = "data/raw/minute/cme_mnq_1m.parquet"
Path(output).parent.mkdir(parents=True, exist_ok=True)
clean_df.to_parquet(output)

console.print(f"[bold green][+] Guardado parquet con velas de 1m REALES: {output} ({len(clean_df):,} barras)[/bold green]")
