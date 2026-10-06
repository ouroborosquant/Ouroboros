import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd
from rich.console import Console
import yfinance as yf

console = Console()


def download_chunked_1m(symbol: str, days_back: int = 24, chunk_days: int = 6) -> pd.DataFrame:
    """Descarga velas de 1m en bloques contiguos de <=6 días para no vulnerar el límite de Yahoo."""
    now = datetime.now(timezone.utc)
    dfs = []
    ticker = yf.Ticker(symbol)

    console.print(f"[bold cyan][*] Descargando {symbol} en bloques de {chunk_days} días (últimos {days_back} días)...[/bold cyan]")

    for i in range(0, days_back, chunk_days):
        end_dt = now - timedelta(days=i)
        start_dt = max(now - timedelta(days=days_back), end_dt - timedelta(days=chunk_days))

        start_str = start_dt.strftime("%Y-%m-%d")
        end_str = end_dt.strftime("%Y-%m-%d")

        try:
            chunk = ticker.history(start=start_str, end=end_str, interval="1m")
            if not chunk.empty:
                dfs.append(chunk)
                console.print(f" • [green]OK[/green] Bloque {start_str} -> {end_str}: {len(chunk)} barras")
        except Exception as e:
            console.print(f" • [yellow]Aviso[/yellow] en bloque {start_str}: {e}")

    if not dfs:
        # Fallback rápido: pedir los últimos 7 días garantizados
        console.print("[yellow][*] Intentando fallback directo a period='7d'...[/yellow]")
        chunk = ticker.history(period="7d", interval="1m")
        if not chunk.empty:
            dfs.append(chunk)

    if not dfs:
        return pd.DataFrame()

    full_df = pd.concat(dfs).sort_index()
    full_df = full_df[~full_df.index.duplicated(keep="first")]
    return full_df


def main():
    parser = argparse.ArgumentParser(description="Descargador gratuito de futuros CME vía yfinance")
    parser.add_argument("--symbol", type=str, default="NQ=F", help="Ticker (NQ=F o MNQ=F)")
    parser.add_argument("--output", type=str, default="data/raw/minute/cme_mnq_1m.parquet")
    args = parser.parse_args()

    raw_df = download_chunked_1m(args.symbol, days_back=24, chunk_days=6)

    # Si NQ=F no responde por fin de semana o festivo, probar MNQ=F
    if raw_df.empty and args.symbol != "MNQ=F":
        console.print("[bold yellow][!] NQ=F vacío, intentando con MNQ=F...[/bold yellow]")
        raw_df = download_chunked_1m("MNQ=F", days_back=24, chunk_days=6)

    if raw_df.empty:
        console.print("[bold red][!] Error: Yahoo no devolvió datos. Comprueba tu conexión a internet.[/bold red]")
        return

    raw_df = raw_df.reset_index()
    time_col = [c for c in raw_df.columns if "date" in c.lower() or "time" in c.lower()][0]

    clean_df = pd.DataFrame({
        "timestamp": pd.to_datetime(raw_df[time_col], utc=True),
        "symbol": "MNQ",
        "open": raw_df["Open"].round(2),
        "high": raw_df["High"].round(2),
        "low": raw_df["Low"].round(2),
        "close": raw_df["Close"].round(2),
        "volume": raw_df["Volume"].astype(float),
    })

    # Filtrar barras inválidas o con volumen cero prolongado
    clean_df = clean_df.dropna().sort_values("timestamp").reset_index(drop=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_parquet(args.output)

    console.print(f"\n[bold green][+] Guardado parquet con éxito en {args.output}[/bold green]")
    console.print(f" • Total de barras de 1m reales: [bold cyan]{len(clean_df):,}[/bold cyan]")
    console.print(f" • Periodo cubierto: [yellow]{clean_df['timestamp'].min()}[/yellow] -> [yellow]{clean_df['timestamp'].max()}[/yellow]\n")


if __name__ == "__main__":
    main()
