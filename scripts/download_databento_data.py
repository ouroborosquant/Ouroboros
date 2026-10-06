import argparse
import os
from pathlib import Path
from datetime import datetime, date
import databento as db
import pandas as pd
from rich.console import Console

console = Console()

def main():
    parser = argparse.ArgumentParser(description="Descargador de datos CME MNQ desde Databento")
    parser.add_argument("--api-key", type=str, default=os.getenv("DATABENTO_API_KEY"), help="API Key de Databento")
    parser.add_argument("--start", type=str, default="2024-01-01", help="Fecha inicio (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="Fecha fin (YYYY-MM-DD), default hoy")
    parser.add_argument("--output", type=str, default="data/raw/minute/cme_mnq_1m.parquet")
    args = parser.parse_args()

    if not args.api_key:
        console.print("[bold red][!] Error: DATABENTO_API_KEY no encontrada.[/bold red]")
        console.print("Exporta tu clave con: [cyan]export DATABENTO_API_KEY='tu_api_key'[/cyan] o pásala con [cyan]--api-key[/cyan].")
        return

    client = db.Historical(key=args.api_key)
    end_date = args.end if args.end else datetime.now().strftime("%Y-%m-%d")

    console.print(f"[bold cyan][*] Solicitando datos históricos de CME Globex (GLBX.MDP3) para MNQ...[/bold cyan]")
    console.print(f" • Rango: [yellow]{args.start}[/yellow] -> [yellow]{end_date}[/yellow]")
    console.print(f" • Schema: [yellow]ohlcv-1m[/yellow] (Continuous front contract: MNQ.c.0)")

    # 1. Estimar coste antes de descargar
    cost = client.metadata.get_cost(
        dataset="GLBX.MDP3",
        symbols=["MNQ.c.0"],
        schema="ohlcv-1m",
        start=args.start,
        end=end_date,
    )
    console.print(f" • Coste estimado de descarga: [bold green]${cost:.2f} USD[/bold green]")

    # 2. Obtener datos
    console.print("[*] Descargando y procesando stream...")
    data = client.timeseries.get_range(
        dataset="GLBX.MDP3",
        symbols=["MNQ.c.0"],
        schema="ohlcv-1m",
        start=args.start,
        end=end_date,
    )

    df = data.to_df()

    # 3. Formatear para que encaje directamente en el motor de BarEvent
    df = df.reset_index()
    # Databento devuelve timestamp en columna 'ts_event' o index en UTC
    time_col = "ts_event" if "ts_event" in df.columns else "index"
    
    clean_df = pd.DataFrame({
        "timestamp": pd.to_datetime(df[time_col], utc=True),
        "symbol": "MNQ",
        "open": df["open"].round(2),
        "high": df["high"].round(2),
        "low": df["low"].round(2),
        "close": df["close"].round(2),
        "volume": df["volume"].astype(float),
    })

    # Filtrar barras sin volumen o fuera de trading
    clean_df = clean_df.dropna().sort_values("timestamp").reset_index(drop=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_parquet(args.output)

    console.print(f"[bold green][+] Archivo guardado con éxito: {args.output} ({len(clean_df):,} barras reales)[/bold green]")
    console.print(f" • Periodo cubierto: {clean_df['timestamp'].min()} a {clean_df['timestamp'].max()}")

if __name__ == "__main__":
    main()
