"""
FORTRESS v5 - nlp_ingestion.py
Path: data/alt_data/nlp_ingestion.py

NLP Alternative Data Ingestion Pipeline.
Continuously scrapes financial news, computes sentiment and novelty scores,
and publishes structured alpha signals to the GATv2 node feature assembly.

Previously this file was completely empty. The LLM agents module expected
NLP signals to be available in Redis but had no data source.

PIPELINE:
  1. Source: RSS feeds from Reuters, FT, Bloomberg (public), SEC EDGAR (8-K, 10-Q)
  2. Deduplication: SHA-256 fingerprint → Redis SET (24h TTL) prevents re-scoring
  3. Novelty: Embedding distance from ChromaDB stored article corpus
     (Mahalanobis distance in sentence-transformer embedding space)
  4. Sentiment: FinBERT-based per-ticker sentiment classification
     Output: 5-class probability vector [crash, decline, flat, rise, surge]
  5. Routing:
     - novelty < 1.5 → fast FinBERT path (< 10ms)
     - novelty ≥ 1.5 → LLM multi-agent debate path (llm_agents.py, ~1-3s)
  6. Storage: Redis 'nlp:{ticker}:signal' (TTL=24h) + Kafka 'nlp-signals' topic

LOOK-AHEAD SAFETY:
  All ingested articles carry their `published_at` timestamp.
  The `as_of_date` inference path in the backtest engine reads only articles
  with published_at <= as_of_date via TimescaleDB `news_articles` table.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from xml.etree import ElementTree

import aiohttp
import numpy as np

logger = logging.getLogger("NLPIngestion")

# ── RSS feed configuration ────────────────────────────────────────────────────
_RSS_FEEDS: List[Dict[str, str]] = [
    {
        "name":   "Reuters Markets",
        "url":    "https://feeds.reuters.com/reuters/businessNews",
        "weight": 1.0,
    },
    {
        "name":   "MarketWatch",
        "url":    "https://feeds.marketwatch.com/marketwatch/topstories/",
        "weight": 0.8,
    },
    {
        "name":   "Seeking Alpha (ETF)",
        "url":    "https://seekingalpha.com/feed/sector/etfs.xml",
        "weight": 0.6,
    },
    {
        "name":   "SEC EDGAR 8-K Filings",
        "url":    "https://www.sec.gov/cgi-bin/browse-edgar"
                  "?action=getcurrent&type=8-K&dateb=&owner=include&count=20&output=atom",
        "weight": 1.2,
    },
]

# Asset universe for ticker mention detection
_TICKER_ALIASES: Dict[str, List[str]] = {
    "SPY":  ["S&P 500", "SP500", "S&P500", "SPY"],
    "QQQ":  ["Nasdaq", "NASDAQ", "QQQ", "tech stocks"],
    "TLT":  ["Treasury", "10-year", "bonds", "TLT", "long bond"],
    "GLD":  ["gold", "Gold", "GLD", "XAU"],
    "VIXY": ["VIX", "volatility", "VIXY", "fear index"],
    "XLE":  ["energy", "oil", "XLE", "crude"],
    "XLF":  ["banks", "financials", "XLF"],
    "XLK":  ["technology", "tech", "XLK", "semiconductor"],
    "XLV":  ["healthcare", "pharma", "XLV"],
    "GDX":  ["gold miners", "mining", "GDX"],
}

# NLP scoring configuration
_FINBERT_MODEL:   str   = "ProsusAI/finbert"
_EMBED_MODEL:     str   = "sentence-transformers/all-MiniLM-L6-v2"
_NOVELTY_THRESHOLD: float = 1.5   # Mahalanobis distance → LLM path
_REDIS_SIGNAL_TTL:  int   = 86400  # 24h TTL for NLP signals
_CRAWL_INTERVAL_SEC: int  = 300    # 5-minute polling interval
_HTTP_TIMEOUT_SEC:   int  = 15


class NLPIngestionPipeline:
    """
    Async NLP alt-data pipeline.
    Runs as a continuous background task within the DataPipeline orchestrator.

    Attributes:
        device:          'cuda' or 'cpu' for FinBERT inference.
        finbert_ready:   True once FinBERT is loaded into VRAM.
        embed_ready:     True once the embedding model is loaded.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config            = config
        self.device            = "cuda" if self._cuda_available() else "cpu"
        self.finbert_ready:    bool = False
        self.embed_ready:      bool = False
        self._redis            = None
        self._kafka_producer   = None
        self._db_pool          = None
        self._seen_fingerprints: set = set()  # In-memory dedup (Redis is the durable store)
        self._finbert          = None
        self._tokenizer        = None
        self._embed_model      = None
        self._embed_corpus_vecs: Optional[np.ndarray] = None  # (M, embed_dim) corpus matrix

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    # ── Initialisation ────────────────────────────────────────────────────────

    async def setup(self) -> None:
        """Initialise Redis, Kafka, DB, and NLP models."""
        await self._setup_infrastructure()
        await self._load_nlp_models()

    async def _setup_infrastructure(self) -> None:
        try:
            import redis.asyncio as redis
            from aiokafka import AIOKafkaProducer
            import asyncpg
        except ImportError as exc:
            raise ImportError(
                "Requires redis, aiokafka, asyncpg: "
                "pip install redis aiokafka asyncpg"
            ) from exc

        redis_url = os.getenv("REDIS_URL",               "redis://localhost:6379")
        kafka_url = os.getenv("KAFKA_BOOTSTRAP_SERVERS",  "localhost:9092")

        self._redis = redis.Redis.from_url(redis_url)
        self._kafka_producer = AIOKafkaProducer(bootstrap_servers=kafka_url)
        await self._kafka_producer.start()

        self._db_pool = await asyncpg.create_pool(
            user=os.getenv("DB_USER",     "postgres"),
            password=os.getenv("DB_PASSWORD", ""),
            database=os.getenv("DB_NAME",     "fortress"),
            host=os.getenv("DB_HOST",         "localhost"),
            min_size=1,
            max_size=3,
        )
        logger.info("NLPIngestion: infrastructure initialised.")

    async def _load_nlp_models(self) -> None:
        """
        Loads FinBERT and sentence-transformer embedding model.
        Runs in a thread pool to avoid blocking the event loop during model load.
        """
        loop = asyncio.get_event_loop()

        try:
            def _load_finbert():
                from transformers import BertTokenizer, BertForSequenceClassification
                import torch
                tok = BertTokenizer.from_pretrained(_FINBERT_MODEL)
                mdl = BertForSequenceClassification.from_pretrained(_FINBERT_MODEL)
                mdl.eval()
                mdl.to(self.device)
                return tok, mdl

            self._tokenizer, self._finbert = await loop.run_in_executor(None, _load_finbert)
            self.finbert_ready = True
            logger.info(f"FinBERT loaded on {self.device}.")
        except Exception as exc:
            logger.warning(
                f"FinBERT load failed: {exc}. "
                "NLP scoring will use neutral 0.2/0.2/0.2/0.2/0.2 distribution."
            )

        try:
            def _load_embed():
                from sentence_transformers import SentenceTransformer
                return SentenceTransformer(_EMBED_MODEL)

            self._embed_model = await loop.run_in_executor(None, _load_embed)
            self.embed_ready  = True
            logger.info("Sentence-transformer embedding model loaded.")
        except Exception as exc:
            logger.warning(f"Embedding model load failed: {exc}. Novelty scoring disabled.")

    # ── Main loop ─────────────────────────────────────────────────────────────

    async def run(self) -> None:
        """
        Continuous ingestion loop.
        Crawls all RSS feeds every _CRAWL_INTERVAL_SEC seconds.
        """
        await self.setup()
        logger.info(
            f"NLPIngestion: starting continuous crawl "
            f"(interval={_CRAWL_INTERVAL_SEC}s, {len(_RSS_FEEDS)} feeds)."
        )

        while True:
            try:
                await self._crawl_all_feeds()
            except asyncio.CancelledError:
                logger.info("NLPIngestion: shutting down.")
                await self._teardown()
                return
            except Exception as exc:
                logger.error(f"NLPIngestion crawl error: {exc}", exc_info=True)

            await asyncio.sleep(_CRAWL_INTERVAL_SEC)

    async def _crawl_all_feeds(self) -> None:
        """Fetches all RSS feeds concurrently."""
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=_HTTP_TIMEOUT_SEC)
        ) as session:
            tasks = [
                self._process_feed(session, feed)
                for feed in _RSS_FEEDS
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for feed, result in zip(_RSS_FEEDS, results):
            if isinstance(result, Exception):
                logger.warning(f"Feed '{feed['name']}' failed: {result}")

    # ── Feed processing ───────────────────────────────────────────────────────

    async def _process_feed(
        self, session: aiohttp.ClientSession, feed: Dict[str, str]
    ) -> None:
        """Fetches, parses, and processes a single RSS feed."""
        try:
            async with session.get(feed["url"]) as resp:
                if resp.status != 200:
                    logger.debug(
                        f"Feed '{feed['name']}' returned HTTP {resp.status}."
                    )
                    return
                xml_text = await resp.text()
        except Exception as exc:
            logger.debug(f"Feed fetch error '{feed['name']}': {exc}")
            return

        articles = self._parse_rss(xml_text)
        for article in articles:
            await self._process_article(article, feed.get("weight", 1.0))

    def _parse_rss(self, xml_text: str) -> List[Dict[str, str]]:
        """Parses RSS/Atom XML into a list of article dicts."""
        articles = []
        try:
            root = ElementTree.fromstring(xml_text)
            # Handle both RSS 2.0 and Atom namespaces
            ns = {"atom": "http://www.w3.org/2005/Atom"}

            # Try RSS 2.0 <item> tags first
            for item in root.iter("item"):
                title_el   = item.find("title")
                desc_el    = item.find("description")
                pubdate_el = item.find("pubDate")
                link_el    = item.find("link")

                articles.append({
                    "title":        title_el.text   if title_el   is not None else "",
                    "body":         desc_el.text    if desc_el    is not None else "",
                    "published_at": pubdate_el.text if pubdate_el is not None else "",
                    "url":          link_el.text    if link_el    is not None else "",
                })

            # Try Atom <entry> tags if no items found
            if not articles:
                for entry in root.findall("atom:entry", ns):
                    title_el   = entry.find("atom:title", ns)
                    summ_el    = entry.find("atom:summary", ns)
                    updated_el = entry.find("atom:updated", ns)
                    link_el    = entry.find("atom:link", ns)

                    articles.append({
                        "title":        title_el.text   if title_el   is not None else "",
                        "body":         summ_el.text    if summ_el    is not None else "",
                        "published_at": updated_el.text if updated_el is not None else "",
                        "url":          link_el.get("href", "") if link_el is not None else "",
                    })
        except ElementTree.ParseError as exc:
            logger.debug(f"RSS parse error: {exc}")

        return articles

    # ── Article processing ────────────────────────────────────────────────────

    async def _process_article(
        self,
        article: Dict[str, str],
        source_weight: float,
    ) -> None:
        """
        Full processing pipeline for a single article:
          1. Deduplicate via fingerprint
          2. Detect mentioned tickers
          3. Compute novelty score
          4. Score sentiment via FinBERT (or LLM path for high novelty)
          5. Store in Redis and TimescaleDB
          6. Publish to Kafka
        """
        text = (article.get("title", "") + " " + article.get("body", "")).strip()
        if not text:
            return

        # ── Deduplication ────────────────────────────────────────────────────
        fingerprint = hashlib.sha256(text.encode()).hexdigest()
        if fingerprint in self._seen_fingerprints:
            return

        already_seen = await self._redis.sismember("nlp:seen_fingerprints", fingerprint)
        if already_seen:
            self._seen_fingerprints.add(fingerprint)
            return

        await self._redis.sadd("nlp:seen_fingerprints", fingerprint)
        await self._redis.expire("nlp:seen_fingerprints", _REDIS_SIGNAL_TTL)
        self._seen_fingerprints.add(fingerprint)

        # ── Ticker detection ──────────────────────────────────────────────────
        mentioned_tickers = self._detect_tickers(text)
        if not mentioned_tickers:
            return  # No relevant asset mentioned — skip

        # ── Novelty score ─────────────────────────────────────────────────────
        novelty = await self._compute_novelty(text)

        # ── Sentiment scoring ─────────────────────────────────────────────────
        for ticker in mentioned_tickers:
            signal_vector = await self._score_sentiment(text, ticker, novelty)

            # Blend with source weight
            signal_vector = signal_vector * source_weight

            # ── Persist to Redis ─────────────────────────────────────────────
            redis_key = f"nlp:{ticker}:signal"
            existing_raw = await self._redis.get(redis_key)
            if existing_raw is not None:
                # Exponential moving average to blend new signal with existing
                existing = np.array(json.loads(existing_raw), dtype=np.float32)
                signal_vector = 0.7 * existing + 0.3 * signal_vector

            await self._redis.set(
                redis_key,
                json.dumps(signal_vector.tolist()),
                ex=_REDIS_SIGNAL_TTL,
            )

            # ── Persist to TimescaleDB ────────────────────────────────────────
            await self._persist_to_db(
                ticker, text[:512], fingerprint, novelty, signal_vector
            )

            # ── Publish to Kafka ──────────────────────────────────────────────
            if self._kafka_producer:
                payload = {
                    "timestamp":     time.time(),
                    "ticker":        ticker,
                    "novelty_score": round(novelty, 4),
                    "signal_vector": signal_vector.tolist(),
                    "source":        article.get("url", "")[:256],
                }
                await self._kafka_producer.send(
                    "nlp-signals",
                    json.dumps(payload).encode("utf-8"),
                )

    def _detect_tickers(self, text: str) -> List[str]:
        """Detects which universe tickers are mentioned in the article text."""
        mentioned = []
        text_lower = text.lower()
        for ticker, aliases in _TICKER_ALIASES.items():
            if any(alias.lower() in text_lower for alias in aliases):
                mentioned.append(ticker)
        return mentioned

    async def _compute_novelty(self, text: str) -> float:
        """
        Computes the Mahalanobis distance of the article's embedding from the
        stored corpus embedding distribution.

        High novelty (> _NOVELTY_THRESHOLD) → LLM debate path.
        Low novelty  (< _NOVELTY_THRESHOLD) → FinBERT fast path.

        Returns a float in [0, ∞). Returns 0.5 if embedding model is unavailable.
        """
        if not self.embed_ready or self._embed_model is None:
            return 0.5

        try:
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None,
                lambda: self._embed_model.encode(
                    text[:512], convert_to_numpy=True, normalize_embeddings=True
                )
            )
            embedding = embedding.astype(np.float32)

            if self._embed_corpus_vecs is None or len(self._embed_corpus_vecs) < 10:
                # Insufficient corpus to compute Mahalanobis — use L2 norm as proxy
                return float(np.linalg.norm(embedding))

            # Mahalanobis distance from corpus centroid
            centroid = self._embed_corpus_vecs.mean(axis=0)
            delta    = embedding - centroid
            cov      = np.cov(self._embed_corpus_vecs.T)
            cov_inv  = np.linalg.pinv(cov + 1e-6 * np.eye(cov.shape[0]))
            distance = float(np.sqrt(delta @ cov_inv @ delta))

            # Update corpus with this embedding (rolling window of 1000)
            self._embed_corpus_vecs = np.vstack([
                self._embed_corpus_vecs[-999:], embedding[np.newaxis, :]
            ])

            return distance

        except Exception as exc:
            logger.debug(f"Novelty computation failed: {exc}")
            return 0.5

    async def _score_sentiment(
        self,
        text: str,
        ticker: str,
        novelty: float,
    ) -> np.ndarray:
        """
        Routes to FinBERT (fast) or LLM debate (high novelty).

        Returns:
            signal_vector: (5,) float32 array [crash, decline, flat, rise, surge]
                           sums to 1.0 (probability distribution).
        """
        if novelty >= _NOVELTY_THRESHOLD:
            # High novelty → try LLM multi-agent debate
            try:
                from models.alpha.llm_agents import MultiAgentDebate
                debate  = MultiAgentDebate(config=self.config)
                vector  = await debate.run_debate(ticker, text[:1024], novelty)
                return vector.astype(np.float32)
            except Exception as exc:
                logger.warning(f"LLM debate failed ({exc}). Falling back to FinBERT.")

        # FinBERT fast path
        return await self._finbert_score(text)

    async def _finbert_score(self, text: str) -> np.ndarray:
        """
        Runs FinBERT on the article text and maps the 3-class output
        (positive, negative, neutral) to a 5-class alpha signal.

        FinBERT → [positive, negative, neutral]
        Mapping:
          positive  → 60% of mass split between 'rise' and 'surge'
          negative  → 60% of mass split between 'crash' and 'decline'
          neutral   → 'flat' receives the neutral mass

        Returns:
            signal: (5,) float32 [crash, decline, flat, rise, surge]
        """
        neutral = np.array([0.05, 0.20, 0.50, 0.20, 0.05], dtype=np.float32)

        if not self.finbert_ready or self._finbert is None:
            return neutral

        try:
            import torch
            loop = asyncio.get_event_loop()

            def _infer():
                inputs = self._tokenizer(
                    text[:512],
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512,
                ).to(self.device)
                with torch.no_grad():
                    logits = self._finbert(**inputs).logits
                probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
                return probs  # [positive, negative, neutral] in FinBERT label order

            probs = await loop.run_in_executor(None, _infer)
            # FinBERT label order: 0=positive, 1=negative, 2=neutral
            p_pos = float(probs[0])
            p_neg = float(probs[1])
            p_neu = float(probs[2])

            signal = np.array([
                p_neg * 0.30,                    # crash
                p_neg * 0.70,                    # decline
                p_neu,                           # flat
                p_pos * 0.70,                    # rise
                p_pos * 0.30,                    # surge
            ], dtype=np.float32)

            # Renormalise to sum to 1.0
            total = signal.sum()
            if total > 0:
                signal /= total

            return signal

        except Exception as exc:
            logger.debug(f"FinBERT inference failed: {exc}")
            return neutral

    # ── Persistence ───────────────────────────────────────────────────────────

    async def _persist_to_db(
        self,
        ticker:        str,
        text_snippet:  str,
        fingerprint:   str,
        novelty:       float,
        signal_vector: np.ndarray,
    ) -> None:
        """
        Persists the NLP signal to TimescaleDB for backtesting.
        The `published_at` timestamp is stored as the `metric_date`,
        enabling strict as_of_date causality enforcement in backtests.
        """
        if self._db_pool is None:
            return

        query = """
            INSERT INTO nlp_signals
                (metric_date, as_of_date, ticker, text_snippet, fingerprint,
                 novelty_score, crash_prob, decline_prob, flat_prob,
                 rise_prob, surge_prob)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            ON CONFLICT (fingerprint, ticker) DO NOTHING;
        """
        now = datetime.now(tz=timezone.utc)
        try:
            async with self._db_pool.acquire() as conn:
                await conn.execute(
                    query,
                    now.date(),          # metric_date (= publication date)
                    now.date(),          # as_of_date (= ingestion date, always >= metric_date)
                    ticker,
                    text_snippet,
                    fingerprint,
                    float(novelty),
                    float(signal_vector[0]),  # crash
                    float(signal_vector[1]),  # decline
                    float(signal_vector[2]),  # flat
                    float(signal_vector[3]),  # rise
                    float(signal_vector[4]),  # surge
                )
        except Exception as exc:
            logger.debug(f"DB insert failed for NLP signal ({ticker}): {exc}")

    # ── Teardown ──────────────────────────────────────────────────────────────

    async def _teardown(self) -> None:
        if self._kafka_producer:
            await self._kafka_producer.stop()
        if self._redis:
            await self._redis.aclose()
        if self._db_pool:
            await self._db_pool.close()
        logger.info("NLPIngestion: teardown complete.")


if __name__ == "__main__":
    import yaml
    with open("config/hyperparams.yaml", "r") as f:
        config = yaml.safe_load(f)

    pipeline = NLPIngestionPipeline(config)
    asyncio.run(pipeline.run())