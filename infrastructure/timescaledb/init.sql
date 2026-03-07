-- FORTRESS v5 - init.sql
-- Path: infrastructure/timescaledb/init.sql
-- Note: Requires TimescaleDB extension to be enabled in PostgreSQL.

CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ══════════════════════════════════════════════════════════
-- 1. PRICES TABLE (Primary Market Data)
-- ══════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS prices (
    metric_date DATE NOT NULL,          -- The calendar date of the trading session
    as_of_date  DATE NOT NULL,          -- When the data became actionable (usually metric_date + 1)
    ticker      VARCHAR(10) NOT NULL,
    open        NUMERIC(12,4),
    high        NUMERIC(12,4),
    low         NUMERIC(12,4),
    close       NUMERIC(12,4) NOT NULL,
    adj_close   NUMERIC(12,4) NOT NULL,
    volume      BIGINT,
    source      VARCHAR(20),            -- e.g., 'yfinance', 'tiingo'
    is_validated BOOLEAN DEFAULT FALSE, -- True only after cross-source validation passes
    UNIQUE (metric_date, ticker, source)
);

-- Convert to TimescaleDB hypertable partitioned by metric_date
SELECT create_hypertable('prices', 'metric_date', if_not_exists => TRUE);

-- Index for rapid point-in-time backtest queries
CREATE INDEX IF NOT EXISTS idx_prices_as_of ON prices(ticker, as_of_date DESC);


-- ══════════════════════════════════════════════════════════
-- 2. MACRO INDICATORS TABLE (FRED & Economic Data)
-- ══════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS macro_indicators (
    metric_date DATE NOT NULL,          -- The period the indicator describes (e.g., Q3 end)
    as_of_date  DATE NOT NULL,          -- The ACTUAL publication date of the release/revision
    series_id   VARCHAR(30) NOT NULL,   -- e.g., 'GDP', 'T10Y2Y', 'NFCI'
    value       NUMERIC(20,6) NOT NULL,
    vintage_id  INTEGER,                -- FRED vintage ID for tracking historical revisions
    units       VARCHAR(30),
    UNIQUE (metric_date, as_of_date, series_id)
);

SELECT create_hypertable('macro_indicators', 'metric_date', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_macro_as_of ON macro_indicators(series_id, as_of_date DESC);


-- ══════════════════════════════════════════════════════════
-- 3. REGIME POSTERIORS (Mamba-KAN-VAE Latent States)
-- ══════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS regime_posteriors (
    ts              TIMESTAMPTZ NOT NULL, -- Exact timestamp of inference
    z_mu            FLOAT8[] NOT NULL,    -- 16-dim latent mean
    z_sigma         FLOAT8[] NOT NULL,    -- 16-dim latent std dev
    dominant_regime VARCHAR(20),          -- Human-readable classification (e.g., 'crisis', 'low_vol_bull')
    tda_h0          INTEGER,              -- Topological Data Analysis: connected components
    tda_h1          INTEGER,              -- Topological Data Analysis: 1D holes
    tda_wasserstein FLOAT8,               -- Distance metric for structural breakdown alert
    kan_crash_score FLOAT8,               -- Direct output from KAN symbolic rule
    kan_liq_score   FLOAT8,               -- Direct output from KAN symbolic rule
    ltc_urgency     FLOAT8                -- Liquid Neural Net intraday urgency score
);

SELECT create_hypertable('regime_posteriors', 'ts', if_not_exists => TRUE);


-- ══════════════════════════════════════════════════════════
-- 4. ORDERS & FILLS TABLE (Execution Log)
-- ══════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS orders (
    order_id        UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    submitted_at    TIMESTAMPTZ NOT NULL,
    filled_at       TIMESTAMPTZ,
    ticker          VARCHAR(10) NOT NULL,
    side            VARCHAR(4) NOT NULL,  -- 'buy' or 'sell'
    quantity        NUMERIC(12,4) NOT NULL,
    decision_price  NUMERIC(12,4),        -- Mid-price at the exact moment of decision
    fill_price      NUMERIC(12,4),        -- Actual executed price
    shortfall_bps   NUMERIC(10,6),        -- Implementation shortfall in basis points
    agent_used      VARCHAR(20),          -- 'stealth_ppo', 'urgent_ddpg', 'opport_sac', 'emergency_fpga'
    regime_label    VARCHAR(20)           -- Market regime at time of execution
);

-- Standard table (no hypertable needed as order volume for 25 ETFs is low)
CREATE INDEX IF NOT EXISTS idx_orders_submitted ON orders(submitted_at DESC);


-- ══════════════════════════════════════════════════════════
-- 5. LIVE PERFORMANCE (Bayesian Health Monitor Target)
-- ══════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS live_performance (
    ts              TIMESTAMPTZ NOT NULL,
    portfolio_value NUMERIC(15,2) NOT NULL,
    cash_value      NUMERIC(15,2) NOT NULL,
    gross_leverage  NUMERIC(6,4) NOT NULL,
    drawdown_pct    NUMERIC(6,4) NOT NULL,
    daily_pnl       NUMERIC(12,2),
    sharpe_rolling  NUMERIC(8,4)          -- Rolling 6M Sharpe for live/backtest correlation
);

SELECT create_hypertable('live_performance', 'ts', if_not_exists => TRUE);


-- ══════════════════════════════════════════════════════════
-- 6. COMPRESSION POLICIES (Cost Control & Performance)
-- ══════════════════════════════════════════════════════════
-- Compress high-frequency or historical data automatically to save 90%+ disk space
ALTER TABLE prices SET (timescaledb.compress, timescaledb.compress_segmentby = 'ticker');
SELECT add_compression_policy('prices', INTERVAL '7 days');

ALTER TABLE macro_indicators SET (timescaledb.compress, timescaledb.compress_segmentby = 'series_id');
SELECT add_compression_policy('macro_indicators', INTERVAL '7 days');

ALTER TABLE regime_posteriors SET (timescaledb.compress);
SELECT add_compression_policy('regime_posteriors', INTERVAL '3 days');

ALTER TABLE live_performance SET (timescaledb.compress);
SELECT add_compression_policy('live_performance', INTERVAL '3 days');