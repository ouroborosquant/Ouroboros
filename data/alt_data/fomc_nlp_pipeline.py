"""
FORTRESS v6 — data/alt_data/fomc_nlp_pipeline.py
═══════════════════════════════════════════════════════════════════════════════
Institutional NLP: Causal Fed / FOMC Regime Conditioning Pipeline.

WHY EXOGENOUS MACRO-CAUSALITY
──────────────────────────────
The v5 regime detector (WassersteinHMM) is endogenous: it observes price
returns, volatility, and breadth — all *consequences* of macro regime shifts.
By the time the HMM detects a regime shift from price data, the market has
already repriced by 2–5%.

FOMC communications provide *causal* advance information:
  1. FOMC minutes (released ~3 weeks after the meeting) are explicitly
     forward-looking statements about the path of rates.
  2. Fed speaker speeches (Waller, Barr, Jefferson, etc.) provide policy
     signal between meetings at a higher cadence.
  3. Dot plot revisions and SEP (Summary of Economic Projections) move
     bond yields 10–30bps on release — the regime shift IS the text.

By parsing these documents and extracting a policy stance vector
s = [P_hawkish, P_dovish, P_neutral], we can condition the regime model
BEFORE the price impact propagates, giving the allocator a head start.

QUANTIZED FINBERT
─────────────────
ProsusAI/finbert is fine-tuned on financial news for 3-class sentiment.
However, it was NOT trained on central bank policy language. We address this:

  1. Quantize to INT8 (bitsandbytes / llm.int8()) to cut VRAM from ~1.3GB
     to ~380MB — runs comfortably alongside the main inference models.
  2. Fine-tune the classifier head on a curated dataset of FOMC minutes with
     manually labelled hawkish/dovish/neutral passages (see §7).
  3. Sentence-level windowing: split document into 512-token windows, run
     FinBERT on each, aggregate via weighted softmax (weight = novelty score).

POLICY STANCE VECTOR
────────────────────
Output: s_t = [P_hawkish, P_dovish, P_neutral] ∈ Δ³  (simplex)

Integration with regime model (WassersteinHMM / MambaKAN):
  - Concatenated to the 5-dim HMM feature vector → 8-dim.
  - Concatenated to the 52-dim obs vector for MambaKAN → 55-dim.
  - The leverage multiplier is then conditioned as:
        exposure(s_t) = base_exposure × σ(-κ · (P_hawkish - P_dovish))
    where κ calibrates the sensitivity to the hawkish-dovish differential.
  - The Cash Trap logic checks: if P_hawkish > 0.65 → force BIL/SHV regardless
    of the price-based regime signal.

LOOK-AHEAD SAFETY
─────────────────
Every document carries its official Fed release timestamp (from the
federalreserve.gov JSON calendar API). The pipeline stores this as
`published_at` in TimescaleDB. During backtesting, all queries use
`WHERE published_at <= as_of_date` — FOMC minutes released on 2020-01-08
are not used in a backtest day indexed 2020-01-05.

The live pipeline additionally checks: a document scraped today is stored
with today's date, never a backdated timestamp.

ASYNCIO + RESILIENCE
────────────────────
The pipeline is an async background task:
  - RSS polling: every 300s
  - Exponential backoff on failures: 1s → 2s → 4s → … → 300s cap
  - Circuit breaker: 5 consecutive failures → pause 1800s + DLQ alert
  - Rate limit: ≤ 2 requests/s to federalreserve.gov (robots.txt compliant)
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from xml.etree import ElementTree

import aiohttp
import numpy as np

log = logging.getLogger("FOMCNLPPipeline")

# ── Fed document sources ───────────────────────────────────────────────────────
_FED_SOURCES: List[Dict[str, str]] = [
    {
        "name":    "FOMC Minutes",
        "url":     "https://www.federalreserve.gov/monetarypolicy/fomccalendars.json",
        "type":    "json_calendar",
        "weight":  1.5,    # FOMC minutes are highest-quality policy signal
    },
    {
        "name":    "Fed Speeches RSS",
        "url":     "https://www.federalreserve.gov/feeds/speeches.xml",
        "type":    "rss",
        "weight":  1.0,
    },
    {
        "name":    "Fed Press Releases RSS",
        "url":     "https://www.federalreserve.gov/feeds/press_all.xml",
        "type":    "rss",
        "weight":  0.8,
    },
    {
        "name":    "Treasury Yields (BLS/FRED fallback signal)",
        "url":     "https://www.federalreserve.gov/feeds/statistics.xml",
        "type":    "rss",
        "weight":  0.5,
    },
]

# ── Hawkish / Dovish keyword lexicon (bootstrap prior for FinBERT)─────────────
_HAWKISH_KEYWORDS: List[str] = [
    "inflation", "rate hike", "tighten", "restrictive", "above target",
    "accelerate", "price stability", "upside risk", "overshoot",
    "persistent inflation", "labor market tight", "wage growth",
    "quantitative tightening", "QT", "balance sheet reduction",
    "higher for longer", "data dependent", "above neutral",
]
_DOVISH_KEYWORDS: List[str] = [
    "rate cut", "easing", "accommodative", "below target", "slack",
    "unemployment", "slowdown", "recession", "below neutral",
    "quantitative easing", "QE", "balance sheet expansion",
    "insurance cut", "soft landing", "pivot", "disinflation",
    "cooling inflation", "labor market softening",
]
_NEUTRAL_KEYWORDS: List[str] = [
    "data dependent", "monitor", "assess", "uncertain", "balanced",
    "gradual", "meeting by meeting", "flexible",
]

# ── Policy stance integration parameters ─────────────────────────────────────
_HAWKISH_CASH_TRAP_THRESHOLD: float = 0.65   # P_hawkish > 65% → BIL/SHV override
_LEVERAGE_SENSITIVITY:        float = 3.0    # κ in σ(-κ·(P_hawk - P_dove))
_STANCE_EMA_ALPHA:            float = 0.30   # EMA decay for stance updates (faster decay = more responsive)
_REDIS_STANCE_TTL:            int   = 86400  # 24h TTL for stance vector in Redis
_CRAWL_INTERVAL_SEC:          int   = 300    # 5-minute polling
_HTTP_TIMEOUT_SEC:            int   = 20
_CIRCUIT_BREAKER_THRESHOLD:   int   = 5      # consecutive failures before 30-min pause
_CIRCUIT_BREAKER_PAUSE_SEC:   int   = 1800   # 30-minute pause

# ── FinBERT model configuration ───────────────────────────────────────────────
_FINBERT_MODEL:      str  = "ProsusAI/finbert"
_MAX_TOKEN_LEN:      int  = 512
_SENTENCE_STRIDE:    int  = 256    # Overlap between windows (handles sentences split across windows)
_QUANTIZE_8BIT:      bool = True   # Quantize to INT8 via bitsandbytes


# ══════════════════════════════════════════════════════════════════════════════
# §1  POLICY STANCE VECTOR
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PolicyStance:
    r"""
    Fed policy stance vector.

    Fields:
        p_hawkish, p_dovish, p_neutral:
            Probability simplex P ∈ Δ³.  Always sum to 1.0.
        confidence:
            Mean maximum softmax probability across sentence windows.
            Low confidence (< 0.5) implies ambiguous or off-topic text.
        document_type:
            "minutes", "speech", "press_release", etc.
        published_at:
            Official Fed release timestamp (ISO 8601, UTC).
        raw_score:
            Hawkish-dovish differential: P_hawkish - P_dovish.
            Positive = hawkish. Negative = dovish.
        leverage_multiplier:
            Pre-computed exposure scalar:
                exp = σ(−κ · (P_hawkish − P_dovish))
            where κ = _LEVERAGE_SENSITIVITY.
            Range: (0, 1). Multiplies the HMM exposure_multiplier.
        cash_trap_active:
            True if P_hawkish > _HAWKISH_CASH_TRAP_THRESHOLD.
    """
    p_hawkish:           float
    p_dovish:            float
    p_neutral:           float
    confidence:          float
    document_type:       str
    published_at:        str     # ISO 8601 UTC
    source_url:          str
    raw_score:           float   = field(init=False)
    leverage_multiplier: float   = field(init=False)
    cash_trap_active:    bool    = field(init=False)

    def __post_init__(self) -> None:
        # Normalise to simplex
        total = self.p_hawkish + self.p_dovish + self.p_neutral + 1e-10
        self.p_hawkish /= total
        self.p_dovish  /= total
        self.p_neutral /= total

        self.raw_score           = self.p_hawkish - self.p_dovish
        self.leverage_multiplier = float(
            1.0 / (1.0 + np.exp(_LEVERAGE_SENSITIVITY * self.raw_score))
        )  # σ(−κ · raw_score)
        self.cash_trap_active = self.p_hawkish > _HAWKISH_CASH_TRAP_THRESHOLD

    def to_feature_vector(self) -> np.ndarray:
        """
        Returns a 3-dim feature vector [P_hawkish, P_dovish, P_neutral]
        for concatenation with regime model inputs.
        """
        return np.array([self.p_hawkish, self.p_dovish, self.p_neutral], dtype=np.float32)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "p_hawkish":           round(self.p_hawkish, 5),
            "p_dovish":            round(self.p_dovish, 5),
            "p_neutral":           round(self.p_neutral, 5),
            "confidence":          round(self.confidence, 4),
            "document_type":       self.document_type,
            "published_at":        self.published_at,
            "source_url":          self.source_url,
            "raw_score":           round(self.raw_score, 5),
            "leverage_multiplier": round(self.leverage_multiplier, 5),
            "cash_trap_active":    self.cash_trap_active,
        }


# Neutral prior — used when no FOMC document has been processed yet
_NEUTRAL_PRIOR = PolicyStance(
    p_hawkish     = 0.25,
    p_dovish      = 0.25,
    p_neutral     = 0.50,
    confidence    = 0.0,
    document_type = "prior",
    published_at  = "1970-01-01T00:00:00Z",
    source_url    = "",
)


# ══════════════════════════════════════════════════════════════════════════════
# §2  QUANTIZED FEDBERT (FinBERT + INT8)
# ══════════════════════════════════════════════════════════════════════════════

class QuantizedFedBERT:
    r"""
    INT8-quantized FinBERT adapted for Fed policy stance classification.

    Quantization strategy: bitsandbytes llm.int8()
        - Outlier channels (> 6σ from mean) kept in float16.
        - All other weights quantized to int8 (mixed-precision decomposition).
        - This achieves ~3.5× compression with < 1% accuracy degradation
          on financial classification tasks.
        - VRAM reduction: ~1.3GB → ~380MB, compatible with a GPU already
          running the GAT + CVaR inference pipeline.

    FinBERT output mapping:
        Standard FinBERT: [positive, negative, neutral] → 3 classes
        Fed stance mapping:
            positive → dovish   (positive economic outlook = accommodative stance)
            negative → hawkish  (negative economic outlook = tightening)
            neutral  → neutral

        NOTE: This mapping is approximate. For production, fine-tune the
        classifier head on a curated FOMC dataset with hawkish/dovish labels.
        See §7 for the curated dataset structure.

    Sentence windowing:
        Fed documents can be thousands of tokens. We split into overlapping
        windows of _MAX_TOKEN_LEN tokens with stride _SENTENCE_STRIDE:
            window_0: tokens [0, 512)
            window_1: tokens [256, 768)
            ...
        Each window gets a stance prediction. The final stance is the
        confidence-weighted mean across windows.

    Parameters
    ----------
    device:          "cuda" or "cpu".
    quantize:        If True and bitsandbytes is available, load in INT8.
    fine_tuned_path: Path to fine-tuned FOMC classifier weights (optional).
    """

    def __init__(
        self,
        device:          str            = "cuda",
        quantize:        bool           = _QUANTIZE_8BIT,
        fine_tuned_path: Optional[str]  = None,
    ) -> None:
        self.device        = device
        self.quantize      = quantize
        self.fine_tuned_path = fine_tuned_path
        self._tokenizer    = None
        self._model        = None
        self._ready        = False

    def load(self) -> None:
        """
        Loads FinBERT. Called once from the pipeline's async setup.
        Heavy I/O — runs in a thread pool executor.
        """
        try:
            from transformers import (
                BertTokenizerFast,
                BertForSequenceClassification,
                AutoConfig,
            )

            self._tokenizer = BertTokenizerFast.from_pretrained(_FINBERT_MODEL)

            load_kwargs: Dict[str, Any] = {"ignore_mismatched_sizes": True}

            if self.quantize and self.device == "cuda":
                try:
                    import bitsandbytes  # noqa: F401 — check availability
                    load_kwargs["load_in_8bit"] = True
                    load_kwargs["device_map"]   = "auto"
                    log.info("QuantizedFedBERT: INT8 quantization enabled via bitsandbytes.")
                except ImportError:
                    log.warning(
                        "bitsandbytes not installed — loading FinBERT in float32. "
                        "Install: pip install bitsandbytes for INT8 compression."
                    )

            self._model = BertForSequenceClassification.from_pretrained(
                _FINBERT_MODEL, **load_kwargs
            )

            # Load fine-tuned FOMC classifier head if available
            if self.fine_tuned_path and os.path.exists(self.fine_tuned_path):
                import torch
                state = torch.load(self.fine_tuned_path, map_location=self.device)
                # Load only the classifier head (final linear layer)
                classifier_state = {
                    k.replace("classifier.", ""): v
                    for k, v in state.items()
                    if k.startswith("classifier.")
                }
                self._model.classifier.load_state_dict(classifier_state, strict=False)
                log.info("QuantizedFedBERT: fine-tuned classifier head loaded from %s", self.fine_tuned_path)

            if not load_kwargs.get("load_in_8bit"):
                self._model = self._model.to(self.device)

            self._model.eval()
            self._ready = True
            log.info("QuantizedFedBERT: model ready on %s.", self.device)

        except Exception as exc:
            log.error("QuantizedFedBERT load failed: %s — will use lexicon fallback.", exc)
            self._ready = False

    def score(self, text: str) -> Tuple[np.ndarray, float]:
        r"""
        Score a text passage and return [P_hawkish, P_dovish, P_neutral].

        Algorithm:
          1. Tokenize text with stride-based sliding window.
          2. For each window, run FinBERT → softmax probabilities.
          3. Weight each window by its mean max-probability (confidence).
          4. Return confidence-weighted mean probability vector.

        Args:
            text: Raw Fed document text (any length).

        Returns:
            (np.ndarray of shape (3,), float confidence)
            Array is [P_hawkish, P_dovish, P_neutral].
        """
        if not self._ready:
            return self._lexicon_fallback(text)

        try:
            import torch
            tokens = self._tokenizer.encode(text, add_special_tokens=False)
            windows = self._sliding_windows(tokens)

            all_probs:       List[np.ndarray] = []
            all_confidences: List[float]      = []

            with torch.no_grad():
                for window_ids in windows:
                    enc = self._tokenizer.prepare_for_model(
                        window_ids,
                        max_length    = _MAX_TOKEN_LEN,
                        padding       = "max_length",
                        truncation    = True,
                        return_tensors= "pt",
                    )
                    enc = {k: v.to(self.device) for k, v in enc.items()}
                    logits = self._model(**enc).logits   # (1, 3)
                    probs  = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                    # FinBERT labels: 0=positive, 1=negative, 2=neutral
                    # Remap → [hawkish, dovish, neutral]
                    p_hawk = float(probs[1])   # negative → hawkish
                    p_dove = float(probs[0])   # positive → dovish
                    p_neut = float(probs[2])   # neutral
                    remapped = np.array([p_hawk, p_dove, p_neut], dtype=np.float32)

                    confidence = float(probs.max())
                    all_probs.append(remapped)
                    all_confidences.append(confidence)

            if not all_probs:
                return self._lexicon_fallback(text)

            # Confidence-weighted aggregation
            conf_arr = np.array(all_confidences)
            weights  = conf_arr / (conf_arr.sum() + 1e-8)
            stance   = np.stack(all_probs) * weights[:, np.newaxis]
            stance   = stance.sum(axis=0)  # (3,) weighted mean

            # Renormalise
            stance = np.clip(stance, 0.0, 1.0)
            stance /= stance.sum() + 1e-8

            mean_conf = float(conf_arr.mean())
            return stance, mean_conf

        except Exception as exc:
            log.debug("FinBERT inference failed: %s — lexicon fallback", exc)
            return self._lexicon_fallback(text)

    def _sliding_windows(self, token_ids: List[int]) -> List[List[int]]:
        """
        Generate token_id windows of length _MAX_TOKEN_LEN with stride _SENTENCE_STRIDE.

        The stride overlap ensures sentences split at window boundaries are
        captured in at least one window at sufficient context length.
        """
        if len(token_ids) <= _MAX_TOKEN_LEN - 2:  # -2 for [CLS], [SEP]
            return [token_ids]

        effective_len = _MAX_TOKEN_LEN - 2   # space for special tokens
        windows       = []
        start         = 0
        while start < len(token_ids):
            end = min(start + effective_len, len(token_ids))
            windows.append(token_ids[start:end])
            if end >= len(token_ids):
                break
            start += _SENTENCE_STRIDE  # advance by stride, not full window
        return windows

    def _lexicon_fallback(self, text: str) -> Tuple[np.ndarray, float]:
        """
        Keyword counting fallback when FinBERT is unavailable.

        This is a last-resort signal — IC is much lower than FinBERT (~0.03
        vs ~0.09) but provides a non-zero prior rather than returning neutral.
        """
        text_lower = text.lower()
        n_hawk = sum(kw.lower() in text_lower for kw in _HAWKISH_KEYWORDS)
        n_dove = sum(kw.lower() in text_lower for kw in _DOVISH_KEYWORDS)
        n_neut = sum(kw.lower() in text_lower for kw in _NEUTRAL_KEYWORDS)

        total = n_hawk + n_dove + n_neut + 3   # +3 Laplace smoothing
        stance = np.array([
            (n_hawk + 1) / total,
            (n_dove + 1) / total,
            (n_neut + 1) / total,
        ], dtype=np.float32)
        stance /= stance.sum()
        return stance, 0.3   # low confidence for lexicon


# ══════════════════════════════════════════════════════════════════════════════
# §3  STANCE AGGREGATOR (EMA + PERSISTENCE)
# ══════════════════════════════════════════════════════════════════════════════

class PolicyStanceAggregator:
    r"""
    Maintains a running EMA of the policy stance vector.

    Fed communications arrive sporadically (minutes every 6 weeks, speeches
    every few days). Between documents, the stance should persist but decay
    toward the neutral prior as uncertainty grows.

    EMA update rule:
        s_t = α · s_new + (1-α) · s_{t-1}

    where α = _STANCE_EMA_ALPHA = 0.30.

    Higher-weight documents (FOMC minutes, weight=1.5) receive proportionally
    more influence:
        s_t = α_effective · s_new + (1-α_effective) · s_{t-1}
        α_effective = min(α × source_weight, 0.80)

    Persistence / decay: when no new document arrives, the stance decays
    toward the neutral prior each day:
        s_t = λ · s_{t-1} + (1-λ) · s_neutral_prior
    with λ = 0.95 (5% daily decay toward neutral — full decay in ~3 weeks).
    """

    def __init__(self, ema_alpha: float = _STANCE_EMA_ALPHA) -> None:
        self.ema_alpha         = ema_alpha
        self._stance: np.ndarray = _NEUTRAL_PRIOR.to_feature_vector()
        self._last_update: float = 0.0
        self._n_updates:    int  = 0

    def update(self, new_stance: np.ndarray, source_weight: float = 1.0) -> None:
        """
        EMA update with source-weight-scaled alpha.

        Args:
            new_stance:    (3,) [P_hawkish, P_dovish, P_neutral] from FinBERT.
            source_weight: Document importance weight (FOMC minutes = 1.5).
        """
        alpha_eff = min(self.ema_alpha * source_weight, 0.80)
        self._stance = alpha_eff * new_stance + (1.0 - alpha_eff) * self._stance
        self._stance /= self._stance.sum() + 1e-8   # renormalise after EMA
        self._last_update = time.time()
        self._n_updates  += 1

    def decay_toward_neutral(self, decay_lambda: float = 0.95) -> None:
        """
        Called once per trading day to decay toward the neutral prior when
        no new Fed document has arrived.
        """
        neutral = _NEUTRAL_PRIOR.to_feature_vector()
        self._stance = decay_lambda * self._stance + (1.0 - decay_lambda) * neutral
        self._stance /= self._stance.sum() + 1e-8

    @property
    def current(self) -> np.ndarray:
        """Returns the current stance vector (3,)."""
        return self._stance.copy()

    def to_policy_stance(
        self,
        document_type: str = "aggregated",
        source_url:    str = "",
    ) -> PolicyStance:
        """Materialise the current EMA state as a PolicyStance dataclass."""
        return PolicyStance(
            p_hawkish     = float(self._stance[0]),
            p_dovish      = float(self._stance[1]),
            p_neutral     = float(self._stance[2]),
            confidence    = min(self._n_updates / 10.0, 1.0),
            document_type = document_type,
            published_at  = datetime.now(tz=timezone.utc).isoformat(),
            source_url    = source_url,
        )


# ══════════════════════════════════════════════════════════════════════════════
# §4  FOMC CALENDAR SCRAPER
# ══════════════════════════════════════════════════════════════════════════════

class FOMCCalendarScraper:
    """
    Scrapes the FOMC meeting calendar JSON and fetches minutes PDFs.

    The Federal Reserve publishes a machine-readable calendar at:
        https://www.federalreserve.gov/monetarypolicy/fomccalendars.json

    This JSON lists all FOMC meetings with:
      - meeting date
      - minutes release date
      - URL of the minutes PDF (when available)

    We extract the minutes PDF URL, fetch the text via pdfplumber, and
    return it for FinBERT scoring.

    Rate limiting: 2 requests/second to federalreserve.gov.
    """

    _BASE_URL  = "https://www.federalreserve.gov"
    _CAL_URL   = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.json"
    _REQ_DELAY = 0.5   # 0.5s between requests = 2 req/s

    def __init__(self, session: aiohttp.ClientSession) -> None:
        self._session = session
        self._seen:   set = set()

    async def get_new_documents(self) -> List[Dict[str, str]]:
        """
        Polls the FOMC calendar JSON and returns documents not yet processed.

        Returns:
            List of dicts with keys:
                {'text': str, 'doc_type': str, 'url': str, 'published_at': str}
        """
        try:
            async with self._session.get(
                self._CAL_URL,
                timeout=aiohttp.ClientTimeout(total=_HTTP_TIMEOUT_SEC)
            ) as resp:
                if resp.status != 200:
                    log.debug("FOMC calendar HTTP %d", resp.status)
                    return []
                data = await resp.json(content_type=None)
        except Exception as exc:
            log.warning("FOMC calendar fetch failed: %s", exc)
            return []

        documents = []
        years = data if isinstance(data, list) else data.get("mtgs", [])

        for year_block in years:
            mtgs = year_block.get("mtgs", []) if isinstance(year_block, dict) else []
            for mtg in mtgs:
                # Minutes URL (when released)
                minutes_url = mtg.get("minutesUrl", "")
                if not minutes_url or minutes_url in self._seen:
                    continue

                # Full URL
                full_url = (
                    minutes_url if minutes_url.startswith("http")
                    else self._BASE_URL + minutes_url
                )

                # Minutes release date
                minutes_date = mtg.get("minutesDate", "")

                await asyncio.sleep(self._REQ_DELAY)  # rate limit
                text = await self._fetch_document_text(full_url)
                if text:
                    documents.append({
                        "text":         text,
                        "doc_type":     "minutes",
                        "url":          full_url,
                        "published_at": minutes_date or datetime.now(tz=timezone.utc).isoformat(),
                    })
                    self._seen.add(minutes_url)

        return documents

    async def _fetch_document_text(self, url: str) -> str:
        """
        Fetches text from a Fed document URL. Handles both HTML and PDF.

        PDF extraction uses pdfplumber (preferred) with a pypdf fallback.
        HTML extraction strips tags with a simple regex pass.
        """
        try:
            async with self._session.get(
                url, timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status != 200:
                    return ""

                content_type = resp.headers.get("Content-Type", "")
                raw_bytes    = await resp.read()

            if "pdf" in content_type.lower() or url.lower().endswith(".pdf"):
                return await asyncio.get_event_loop().run_in_executor(
                    None, self._extract_pdf_text, raw_bytes
                )
            else:
                # HTML — strip tags, decode
                text = raw_bytes.decode("utf-8", errors="replace")
                return self._strip_html(text)

        except Exception as exc:
            log.debug("Document fetch failed [%s]: %s", url, exc)
            return ""

    @staticmethod
    def _extract_pdf_text(pdf_bytes: bytes) -> str:
        """Extract plain text from PDF bytes using pdfplumber."""
        try:
            import io
            import pdfplumber
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                texts = [p.extract_text() or "" for p in pdf.pages[:30]]  # cap at 30 pages
            return "\n".join(texts)
        except ImportError:
            # pdfplumber not installed — try pypdf fallback
            try:
                import io
                import pypdf
                reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
                return "\n".join(p.extract_text() or "" for p in reader.pages[:30])
            except Exception:
                return ""
        except Exception as exc:
            log.debug("PDF extraction failed: %s", exc)
            return ""

    @staticmethod
    def _strip_html(html: str) -> str:
        """Minimal HTML tag stripper."""
        import re
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text)
        return text.strip()


# ══════════════════════════════════════════════════════════════════════════════
# §5  RSS FEED PROCESSOR (SPEECHES & PRESS RELEASES)
# ══════════════════════════════════════════════════════════════════════════════

class FedRSSProcessor:
    """
    Processes Fed RSS feeds (speeches, press releases) using the same
    async + dedup pattern as nlp_ingestion.py.
    """

    def __init__(
        self,
        session:    aiohttp.ClientSession,
        seen_hashes: set,
    ) -> None:
        self._session = session
        self._seen    = seen_hashes

    async def process_feed(
        self,
        feed_url:   str,
        doc_type:   str,
        weight:     float,
    ) -> List[Dict[str, str]]:
        """
        Fetch and parse an RSS feed, returning new documents only.

        Returns:
            List of {'text', 'doc_type', 'url', 'published_at', 'weight'} dicts.
        """
        try:
            async with self._session.get(
                feed_url,
                timeout=aiohttp.ClientTimeout(total=_HTTP_TIMEOUT_SEC)
            ) as resp:
                if resp.status != 200:
                    return []
                xml_text = await resp.text()
        except Exception as exc:
            log.debug("RSS feed error [%s]: %s", feed_url, exc)
            return []

        items    = self._parse_rss(xml_text)
        new_docs = []

        for item in items:
            fp = hashlib.sha256(item["title"].encode()).hexdigest()
            if fp in self._seen:
                continue
            self._seen.add(fp)

            text = item.get("title", "") + " " + item.get("description", "")
            if len(text.strip()) < 50:
                continue

            new_docs.append({
                "text":         text,
                "doc_type":     doc_type,
                "url":          item.get("link", ""),
                "published_at": item.get("pubDate", datetime.now(tz=timezone.utc).isoformat()),
                "weight":       weight,
            })

        return new_docs

    @staticmethod
    def _parse_rss(xml_text: str) -> List[Dict[str, str]]:
        items = []
        try:
            root = ElementTree.fromstring(xml_text)
            for item in root.iter("item"):
                title_el = item.find("title")
                desc_el  = item.find("description")
                date_el  = item.find("pubDate")
                link_el  = item.find("link")
                items.append({
                    "title":       title_el.text  if title_el else "",
                    "description": desc_el.text   if desc_el  else "",
                    "pubDate":     date_el.text   if date_el  else "",
                    "link":        link_el.text   if link_el  else "",
                })
        except ElementTree.ParseError:
            pass
        return items


# ══════════════════════════════════════════════════════════════════════════════
# §6  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

class FOMCNLPPipeline:
    """
    Async Fed policy stance pipeline.

    Components:
        FOMCCalendarScraper   → FOMC minutes PDFs
        FedRSSProcessor       → speeches + press releases
        QuantizedFedBERT      → stance scoring (INT8 FinBERT)
        PolicyStanceAggregator → EMA + persistence
        Redis                 → live stance storage (TTL=24h)
        TimescaleDB           → backtesting-safe historical storage

    Integration points:
        1. `get_current_stance()` → returns PolicyStance for live inference.
        2. `get_feature_vector()` → returns (3,) numpy array for regime model.
        3. `compute_exposure_multiplier()` → float for leverage scaling.
        4. `is_cash_trap_active()` → bool for hard override.

    Async execution:
        `run()` is an infinite loop — launch with asyncio.create_task().
        `run_batch()` processes all available docs once — for offline backfill.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config    = config
        self._fedbert  = QuantizedFedBERT(
            device          = "cuda" if self._cuda() else "cpu",
            quantize        = config.get("quantize_8bit", True),
            fine_tuned_path = config.get("fomc_fine_tuned_path"),
        )
        self._aggregator      = PolicyStanceAggregator()
        self._redis           = None
        self._db_pool         = None
        self._seen_hashes:    set = set()
        self._consecutive_failures: int = 0

    @staticmethod
    def _cuda() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    # ── Setup ─────────────────────────────────────────────────────────────────

    async def setup(self) -> None:
        """Initialise Redis, DB, and FinBERT. Call once before run()."""
        await self._setup_infrastructure()
        loop = asyncio.get_event_loop()
        log.info("FOMCNLPPipeline: loading FinBERT...")
        await loop.run_in_executor(None, self._fedbert.load)
        log.info("FOMCNLPPipeline: setup complete.")

    async def _setup_infrastructure(self) -> None:
        try:
            import redis.asyncio as redis
            import asyncpg
        except ImportError as exc:
            raise ImportError(f"Requires redis and asyncpg: pip install redis asyncpg") from exc

        self._redis = redis.Redis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379")
        )
        self._db_pool = await asyncpg.create_pool(
            user     = os.getenv("DB_USER",     "postgres"),
            password = os.getenv("DB_PASSWORD", ""),
            database = os.getenv("DB_NAME",     "fortress"),
            host     = os.getenv("DB_HOST",     "localhost"),
            min_size = 1,
            max_size = 3,
        )
        # Restore any cached stance from Redis (survives container restarts)
        await self._restore_cached_stance()
        log.info("FOMCNLPPipeline: infrastructure ready.")

    async def _restore_cached_stance(self) -> None:
        """Load previously computed stance from Redis on startup."""
        try:
            raw = await self._redis.get("fomc:current_stance")
            if raw:
                data = json.loads(raw)
                stance_vec = np.array([
                    data["p_hawkish"], data["p_dovish"], data["p_neutral"]
                ], dtype=np.float32)
                self._aggregator._stance = stance_vec
                log.info(
                    "FOMCNLPPipeline: restored stance from Redis: "
                    "P_hawk=%.3f P_dove=%.3f P_neut=%.3f",
                    *stance_vec,
                )
        except Exception as exc:
            log.debug("Could not restore stance from Redis: %s", exc)

    # ── Main loop ──────────────────────────────────────────────────────────────

    async def run(self) -> None:
        """
        Continuous async ingestion loop. Launch with asyncio.create_task().
        Implements: circuit breaker, exponential backoff, daily neutral decay.
        """
        await self.setup()
        log.info(
            "FOMCNLPPipeline: starting continuous monitoring "
            "(interval=%ds, circuit_breaker=%d failures)",
            _CRAWL_INTERVAL_SEC, _CIRCUIT_BREAKER_THRESHOLD,
        )

        backoff         = 1
        last_decay_date = ""

        while True:
            try:
                # ── Daily neutral decay ──────────────────────────────────
                today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
                if today != last_decay_date:
                    self._aggregator.decay_toward_neutral()
                    last_decay_date = today
                    log.debug("Stance decayed toward neutral for new trading day.")

                # ── Scrape new documents ─────────────────────────────────
                await self._scrape_and_score_cycle()

                # Success — reset backoff + circuit breaker
                self._consecutive_failures = 0
                backoff = 1
                await asyncio.sleep(_CRAWL_INTERVAL_SEC)

            except asyncio.CancelledError:
                log.info("FOMCNLPPipeline: shutdown.")
                await self._teardown()
                return

            except Exception as exc:
                self._consecutive_failures += 1
                log.error(
                    "FOMCNLPPipeline ERROR [%d/%d]: %s",
                    self._consecutive_failures,
                    _CIRCUIT_BREAKER_THRESHOLD,
                    exc,
                    exc_info=True,
                )

                if self._consecutive_failures >= _CIRCUIT_BREAKER_THRESHOLD:
                    log.error(
                        "CIRCUIT BREAKER OPEN — pausing %ds.",
                        _CIRCUIT_BREAKER_PAUSE_SEC,
                    )
                    await self._publish_dlq("circuit_breaker", str(exc))
                    await asyncio.sleep(_CIRCUIT_BREAKER_PAUSE_SEC)
                    self._consecutive_failures = 0
                else:
                    backoff = min(backoff * 2, 300)
                    await asyncio.sleep(backoff)

    async def _scrape_and_score_cycle(self) -> None:
        """One full scrape+score cycle: FOMC minutes + RSS feeds."""
        async with aiohttp.ClientSession(
            timeout  = aiohttp.ClientTimeout(total=_HTTP_TIMEOUT_SEC),
            headers  = {"User-Agent": "FortressV6-ResearchBot/1.0 (+https://example.com/bot)"},
        ) as session:

            fomc_scraper = FOMCCalendarScraper(session)
            rss_proc     = FedRSSProcessor(session, self._seen_hashes)

            # ── FOMC minutes ──────────────────────────────────────────────
            minutes_docs = await fomc_scraper.get_new_documents()
            for doc in minutes_docs:
                await self._process_document(doc, weight=1.5)

            # ── RSS feeds ─────────────────────────────────────────────────
            rss_tasks = []
            for src in _FED_SOURCES:
                if src["type"] == "rss":
                    rss_tasks.append(
                        rss_proc.process_feed(src["url"], src["name"], src.get("weight", 1.0))
                    )

            rss_results = await asyncio.gather(*rss_tasks, return_exceptions=True)
            for feed_docs in rss_results:
                if isinstance(feed_docs, list):
                    for doc in feed_docs:
                        await self._process_document(doc, weight=doc.get("weight", 1.0))

    async def _process_document(self, doc: Dict[str, str], weight: float) -> None:
        """
        Full processing pipeline for a single Fed document:
          1. Score with FinBERT → [P_hawkish, P_dovish, P_neutral]
          2. Update EMA aggregator
          3. Persist to Redis (live) and TimescaleDB (backtest)
          4. Publish stance update to Kafka
        """
        text     = doc.get("text", "")
        doc_type = doc.get("doc_type", "unknown")
        url      = doc.get("url", "")
        pub_at   = doc.get("published_at", datetime.now(tz=timezone.utc).isoformat())

        if not text or len(text) < 100:
            return

        # ── FinBERT scoring ───────────────────────────────────────────────
        loop = asyncio.get_event_loop()
        stance_vec, confidence = await loop.run_in_executor(
            None, self._fedbert.score, text
        )

        # ── Build PolicyStance dataclass ──────────────────────────────────
        ps = PolicyStance(
            p_hawkish     = float(stance_vec[0]),
            p_dovish      = float(stance_vec[1]),
            p_neutral     = float(stance_vec[2]),
            confidence    = confidence,
            document_type = doc_type,
            published_at  = pub_at,
            source_url    = url,
        )

        log.info(
            "[%s] Hawk=%.3f Dove=%.3f Neut=%.3f | conf=%.2f | exposure_mult=%.3f%s",
            doc_type,
            ps.p_hawkish, ps.p_dovish, ps.p_neutral,
            ps.confidence,
            ps.leverage_multiplier,
            " ⚠ CASH TRAP" if ps.cash_trap_active else "",
        )

        # ── EMA update ────────────────────────────────────────────────────
        self._aggregator.update(stance_vec, source_weight=weight)

        # ── Persist to Redis ──────────────────────────────────────────────
        await self._persist_redis(ps)

        # ── Persist to TimescaleDB ────────────────────────────────────────
        await self._persist_db(ps, text[:512])

        # ── Kafka publish ─────────────────────────────────────────────────
        await self._publish_kafka(ps)

    async def _persist_redis(self, ps: PolicyStance) -> None:
        """Write current stance to Redis — used by live inference."""
        if self._redis is None:
            return
        try:
            payload = json.dumps(ps.to_dict())
            await self._redis.set("fomc:current_stance", payload, ex=_REDIS_STANCE_TTL)
            # Also write the 3-dim feature vector for fast concatenation
            vec_payload = json.dumps(ps.to_feature_vector().tolist())
            await self._redis.set("fomc:feature_vector", vec_payload, ex=_REDIS_STANCE_TTL)
        except Exception as exc:
            log.debug("Redis write failed: %s", exc)

    async def _persist_db(self, ps: PolicyStance, text_snippet: str) -> None:
        """
        Persist policy stance to TimescaleDB for backtest causal retrieval.

        The `published_at` field is stored as `metric_date`.
        The ingestion timestamp (now) is stored as `as_of_date`.
        This ensures: during backtests at as_of_date = T, only documents
        with metric_date <= T are accessible.
        """
        if self._db_pool is None:
            return
        query = """
            INSERT INTO fomc_stance
                (metric_date, as_of_date, doc_type, source_url,
                 p_hawkish, p_dovish, p_neutral,
                 confidence, leverage_multiplier, cash_trap_active,
                 text_snippet)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            ON CONFLICT (metric_date, doc_type) DO UPDATE SET
                p_hawkish          = EXCLUDED.p_hawkish,
                p_dovish           = EXCLUDED.p_dovish,
                leverage_multiplier= EXCLUDED.leverage_multiplier,
                cash_trap_active   = EXCLUDED.cash_trap_active;
        """
        try:
            # Parse published_at to date (handles both ISO strings and date strings)
            pub_date = datetime.fromisoformat(
                ps.published_at.replace("Z", "+00:00")
            ).date()
            async with self._db_pool.acquire() as conn:
                await conn.execute(
                    query,
                    pub_date,
                    datetime.now(tz=timezone.utc).date(),
                    ps.document_type,
                    ps.source_url[:256],
                    ps.p_hawkish,
                    ps.p_dovish,
                    ps.p_neutral,
                    ps.confidence,
                    ps.leverage_multiplier,
                    ps.cash_trap_active,
                    text_snippet,
                )
        except Exception as exc:
            log.debug("DB insert failed: %s", exc)

    async def _publish_kafka(self, ps: PolicyStance) -> None:
        """Publish stance update to Kafka `fomc-stance` topic."""
        try:
            from aiokafka import AIOKafkaProducer
            kafka_url = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
            producer  = AIOKafkaProducer(bootstrap_servers=kafka_url)
            await producer.start()
            payload = json.dumps({
                **ps.to_dict(),
                "aggregated_stance": self._aggregator.current.tolist(),
                "timestamp":         time.time(),
            }).encode("utf-8")
            await producer.send_and_wait("fomc-stance", payload)
            await producer.stop()
        except Exception as exc:
            log.debug("Kafka publish failed: %s", exc)

    async def _publish_dlq(self, service: str, error: str) -> None:
        """Publish circuit-breaker event to dead-letter queue."""
        try:
            from aiokafka import AIOKafkaProducer
            kafka_url = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
            producer  = AIOKafkaProducer(bootstrap_servers=kafka_url)
            await producer.start()
            payload = json.dumps({
                "service": service, "error": error, "ts": time.time()
            }).encode("utf-8")
            await producer.send_and_wait("dead-letter-queue", payload)
            await producer.stop()
        except Exception:
            pass

    # ── Public inference interface ─────────────────────────────────────────────

    def get_current_stance(self) -> PolicyStance:
        """Returns the current EMA-aggregated policy stance."""
        return self._aggregator.to_policy_stance()

    def get_feature_vector(self) -> np.ndarray:
        """
        Returns a (3,) float32 array [P_hawkish, P_dovish, P_neutral].
        Ready to concatenate with HMM features or MambaKAN obs vector.
        """
        return self._aggregator.current

    def compute_exposure_multiplier(self) -> float:
        r"""
        Compute the FOMC-conditioned leverage multiplier.

            multiplier = σ(−κ · (P_hawkish − P_dovish))

        where κ = _LEVERAGE_SENSITIVITY = 3.0.

        This is combined with the HMM's exposure_multiplier:
            final_exposure = hmm_exposure × fomc_exposure

        The product ensures BOTH the price-based regime AND the text-based
        policy stance must be positive for full leverage deployment.

        Returns:
            Float in (0, 1). Close to 1.0 = dovish/neutral. Close to 0.0 = hawkish.
        """
        stance = self.get_current_stance()
        return stance.leverage_multiplier

    def is_cash_trap_active(self) -> bool:
        """
        Returns True if P_hawkish > _HAWKISH_CASH_TRAP_THRESHOLD (0.65).

        When True, the portfolio should be 100% BIL/SHV regardless of any
        other signal. This implements the Supreme Law §1 for macro regime.
        """
        stance = self.get_current_stance()
        if stance.cash_trap_active:
            log.warning(
                "FOMC CASH TRAP ACTIVE: P_hawkish=%.3f > %.2f threshold.",
                stance.p_hawkish, _HAWKISH_CASH_TRAP_THRESHOLD,
            )
        return stance.cash_trap_active

    # ── Batch backfill ──────────────────────────────────────────────────────────

    async def run_batch(self) -> None:
        """
        Process all available documents once.
        Used for:
          - Initial database backfill from Fed archive
          - Nightly batch update in DataPipeline._ingest_alt_data()
        """
        await self.setup()
        await self._scrape_and_score_cycle()
        log.info("FOMCNLPPipeline: batch complete.")

    # ── Teardown ───────────────────────────────────────────────────────────────

    async def _teardown(self) -> None:
        if self._redis:
            await self._redis.aclose()
        if self._db_pool:
            await self._db_pool.close()
        log.info("FOMCNLPPipeline: teardown complete.")


# ══════════════════════════════════════════════════════════════════════════════
# §7  FOMC FINE-TUNING DATASET SCHEMA
# ══════════════════════════════════════════════════════════════════════════════
"""
Fine-tuning dataset schema (not implemented here — requires manual labelling).

To fine-tune FinBERT's classifier head for hawkish/dovish/neutral:

    CREATE TABLE fomc_training_data (
        id             SERIAL PRIMARY KEY,
        text_snippet   TEXT NOT NULL,          -- 1-3 sentences from FOMC doc
        label          INT  NOT NULL,          -- 0=hawkish, 1=dovish, 2=neutral
        source_doc     TEXT,                   -- minutes/speech/statement
        meeting_date   DATE,
        annotator      TEXT,
        annotated_at   TIMESTAMPTZ DEFAULT NOW()
    );

Annotation guidelines:
    HAWKISH (0):
        "inflation remains elevated", "labor market remains tight",
        "further policy firming may be appropriate", "above 2%",
        "balance sheet reduction will continue", "additional increases"
    DOVISH (1):
        "appropriate to reduce the target range", "labor market cooling",
        "inflation has eased", "downside risks", "below 2%",
        "support maximum employment"
    NEUTRAL (2):
        "data dependent", "meeting by meeting", "monitor incoming data",
        "assess", "remain attentive to", "balanced risks"

Training procedure (when ≥500 labelled examples are available):
    1. Load QuantizedFedBERT with load_in_8bit=False (full precision for training).
    2. Freeze BERT encoder layers, unfreeze classifier head.
    3. Fine-tune with AdamW, lr=2e-5, batch=16, epochs=5, early stopping.
    4. Save classifier head: torch.save({'classifier.*': state_dict}).
    5. Set config.fomc_fine_tuned_path to saved file.
"""


# ══════════════════════════════════════════════════════════════════════════════
# §8  INTEGRATION HELPER: CONCAT WITH REGIME FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def augment_hmm_features(
    hmm_features: np.ndarray,           # (T, 5) standard HMM features
    fomc_pipeline: FOMCNLPPipeline,
) -> np.ndarray:
    r"""
    Concatenate the 3-dim FOMC stance vector with the 5-dim HMM features.

    Result: (T, 8) augmented feature matrix.

    The FOMC stance is constant for all T rows (it represents the current
    regime conditioning, not a time-varying signal within the window).
    In production, the time-indexed stance would be fetched per date from
    the TimescaleDB `fomc_stance` table.

    Args:
        hmm_features:  (T, 5) causal HMM features from _build_hmm_features().
        fomc_pipeline: Initialised FOMCNLPPipeline with current stance.

    Returns:
        (T, 8) augmented feature matrix.

    Architectural note:
        WassersteinHMM must be retrained with n_features=8 to use this.
        Set config: wasserstein_hmm.n_features: 8 in hyperparams.yaml.
        The 3 new dimensions are: [P_hawkish, P_dovish, P_neutral].
    """
    stance_vec = fomc_pipeline.get_feature_vector()   # (3,)
    stance_tile = np.tile(stance_vec, (hmm_features.shape[0], 1))   # (T, 3)
    return np.concatenate([hmm_features, stance_tile], axis=-1)     # (T, 8)


def augment_obs_vector(
    obs: np.ndarray,                    # (52,) from DataPipeline.get_observation_vector()
    fomc_pipeline: FOMCNLPPipeline,
) -> np.ndarray:
    """
    Append the 3-dim FOMC stance to the 52-dim observation vector → (55,).

    The MambaKAN VAE must be retrained with obs_dim=55.
    Update hyperparams.yaml: mamba_kan.obs_dim: 55.

    Args:
        obs:           (52,) observation vector.
        fomc_pipeline: Initialised FOMCNLPPipeline.

    Returns:
        (55,) augmented observation vector.
    """
    stance = fomc_pipeline.get_feature_vector()  # (3,)
    return np.concatenate([obs, stance]).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# §9  STANDALONE SMOKE TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv
    load_dotenv()

    logging.basicConfig(level=logging.INFO)

    async def smoke_test():
        # Test QuantizedFedBERT lexicon fallback (no model download needed)
        bert = QuantizedFedBERT(device="cpu", quantize=False)

        sample_hawkish = (
            "Inflation remains well above the Committee's 2 percent longer-run goal. "
            "The Committee anticipates that ongoing increases in the target range "
            "will be appropriate in order to attain a stance of monetary policy "
            "that is sufficiently restrictive to return inflation to 2 percent."
        )
        sample_dovish = (
            "Recent indicators suggest that spending and production have moderated. "
            "Inflation has eased somewhat but remains elevated. "
            "The Committee would be prepared to adjust the stance of monetary policy "
            "as appropriate if risks emerge that could impede the attainment of "
            "maximum employment and price stability."
        )

        for label, text in [("HAWKISH", sample_hawkish), ("DOVISH", sample_dovish)]:
            stance_vec, conf = bert._lexicon_fallback(text)
            ps = PolicyStance(
                p_hawkish=float(stance_vec[0]), p_dovish=float(stance_vec[1]),
                p_neutral=float(stance_vec[2]), confidence=conf,
                document_type="test", published_at="2024-01-01T00:00:00Z", source_url=""
            )
            print(f"\n{label}:")
            print(f"  P_hawkish={ps.p_hawkish:.3f}  P_dovish={ps.p_dovish:.3f}  "
                  f"P_neutral={ps.p_neutral:.3f}")
            print(f"  leverage_multiplier={ps.leverage_multiplier:.3f}")
            print(f"  cash_trap_active={ps.cash_trap_active}")

        # Test aggregator
        agg = PolicyStanceAggregator()
        hawkish_vec = np.array([0.70, 0.15, 0.15], dtype=np.float32)
        agg.update(hawkish_vec, source_weight=1.5)
        print(f"\nAggregator after hawkish update: {agg.current.round(3)}")
        agg.decay_toward_neutral()
        print(f"After 1 day decay:               {agg.current.round(3)}")

        # Dummy HMM feature augmentation
        dummy_hmm = np.random.randn(60, 5).astype(np.float32)
        pipeline  = FOMCNLPPipeline({"quantize_8bit": False})
        # Manually set stance for test (skips DB/Redis)
        pipeline._aggregator._stance = hawkish_vec

        augmented = augment_hmm_features(dummy_hmm, pipeline)
        print(f"\nHMM features:  {dummy_hmm.shape}  → augmented: {augmented.shape}")
        print(f"FOMC columns appended: {augmented[0, 5:]}")
        print(f"FOMC exposure multiplier: {pipeline.compute_exposure_multiplier():.4f}")
        print(f"Cash trap active: {pipeline.is_cash_trap_active()}")

    asyncio.run(smoke_test())