import asyncio
import asyncpg
import pandas as pd
import os

async def build():
    # 1. Conectar a la DB
    pool = await asyncpg.create_pool(
        user='postgres', password='fortress', database='fortress', host='localhost'
    )
    
    # 2. Extraer precios
    query = "SELECT metric_date as date, ticker, close FROM prices ORDER BY date, ticker"
    rows = await pool.fetch(query)
    await pool.close()
    
    # 3. Pivotar a formato ancho
    df = pd.DataFrame(rows, columns=['date', 'ticker', 'close'])
    df['date'] = pd.to_datetime(df['date'])
    
    # 🌟 EL PARCHE MÁGICO: Convertir de Decimal a Float64 para PyArrow
    df['close'] = df['close'].astype(float)
    
    prices_wide = df.pivot(index='date', columns='ticker', values='close').sort_index()
    
    # 4. Calcular retornos diarios
    returns_wide = prices_wide.pct_change(fill_method=None).fillna(0.0)
    
    # 5. Guardar en la caché donde los busca el backtester
    os.makedirs('research/outputs/cache', exist_ok=True)
    prices_wide.to_parquet('research/outputs/cache/prices_wide.parquet')
    returns_wide.to_parquet('research/outputs/cache/returns_wide.parquet')
    
    print(f"✅ Caché construida: {prices_wide.shape[1]} activos, {prices_wide.shape[0]} días.")

asyncio.run(build())
