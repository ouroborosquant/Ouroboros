"""
FORTRESS v5 - llm_agents.py
Path: models/alpha/llm_agents.py

Agentic LLM Debate System with Graph-Structured RAG.
Produces probabilistic market outlooks by debating causal supply-chain shocks.
"""

import os
import json
import asyncio
import logging
import numpy as np
from typing import Dict, List, Any, Tuple

# External AI Dependencies
try:
    from langchain_anthropic import ChatAnthropic
    from openai import AsyncOpenAI
    import chromadb
except ImportError:
    raise ImportError("Requires langchain_anthropic, openai, and chromadb.")

logger = logging.getLogger("LLMAgentEngine")

# The strict reasoning structure that all agents must follow
CHAIN_OF_THOUGHT_TEMPLATE = """
You are a top-tier quantitative analyst. Follow these steps exactly:
STEP 1 — IDENTIFY THE SHOCK: What specific new information has arrived?
STEP 2 — FIRST-ORDER EFFECTS: Which assets are directly impacted and via which mechanism?
STEP 3 — SECOND-ORDER EFFECTS: What indirect transmission channels exist?
STEP 4 — COUNTERVAILING FORCES: What buffers or offsets reduce the impact?
STEP 5 — PROBABILITY DISTRIBUTION: Given the above, output a JSON probability distribution 
         for {ticker} over the next 21 trading days. 
         Format: {{"crash": 0.0, "decline": 0.0, "flat": 0.0, "rise": 0.0, "surge": 0.0}}
         The sum must equal 1.0.
"""

class MultiAgentDebate:
    def __init__(self, kg_manager: Any, chroma_client: Any):
        """
        Initializes the 3-agent debate ecosystem and the Graph-RAG clients.
        """
        self.kg = kg_manager
        self.chroma = chroma_client
        
        # Premium Model (Synthesis / High Novelty)
        self.premium_llm = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # In a full deployment, the local model would be Llama-3 via vLLM or llama.cpp
        # self.local_llm = LlamaCpp(model_path='models/weights/llama-3.1-70b.gguf')

    async def run_debate(self, ticker: str, news_context: str, novelty_score: float) -> np.ndarray:
        """
        Executes the debate and returns a 5-dimensional probability vector.
        Cost Routing: 
          - Low novelty (< 1.5): Skips premium debate, uses fast/local estimation.
          - High novelty (> 1.5): Triggers full 3-agent premium debate.
        """
        if novelty_score < 1.5:
            # Fast-path: Routine news does not require a $0.05 LLM call.
            return await self._fast_path_estimation(ticker, news_context)

        logger.info(f"High novelty ({novelty_score:.2f}) detected for {ticker}. Initiating Debate.")
        
        # 1. Graph-Structured Retrieval
        extended_context = await self._retrieve_causal_context(ticker, news_context)
        
        # 2. Parallel Bull and Bear generation
        bull_task = asyncio.create_task(self._agent_generation(ticker, extended_context, "Bull"))
        bear_task = asyncio.create_task(self._agent_generation(ticker, extended_context, "Bear"))
        
        bull_thesis, bear_thesis = await asyncio.gather(bull_task, bear_task)
        
        # 3. Synthesis Arbiter determines final probabilities
        final_distribution = await self._synthesis_arbitration(ticker, bull_thesis, bear_thesis)
        
        # 4. Map the dictionary to a 5-dim numpy array (alpha vector)
        return self._dict_to_alpha_vector(final_distribution)

    async def _retrieve_causal_context(self, ticker: str, query: str) -> str:
        """
        Graph-Structured RAG. 
        Instead of just searching for the ticker, it queries the Knowledge Graph
        to find all upstream causal drivers, then searches the vector DB for them too.
        """
        # Step 1: Query Neo4j/DYNOTEARS graph for causal parents
        # e.g., If ticker == 'XLE', upstream might be 'USO' and 'Interest_Rates'
        upstream_nodes = self.kg.get_causal_upstream(ticker, max_depth=2)
        
        search_terms = [ticker] + upstream_nodes
        
        # Step 2: Query ChromaDB for all related nodes
        # This is abstracted; in reality, we'd query a collection of recent financial news
        # context_docs = self.chroma.query(query_texts=search_terms, n_results=5)
        
        # Dummy context for scaffolding
        context = f"Retrieved documents regarding {ticker} and its causal drivers: {upstream_nodes}."
        return f"NEWS SHOCK: {query}\n\nCAUSAL CONTEXT:\n{context}"

    async def _agent_generation(self, ticker: str, context: str, stance: str) -> str:
        """
        Forces an LLM to take a specific adversarial stance.
        """
        system_prompt = (
            f"You are the {stance} Agent. You must interpret the following context "
            f"to construct the most compelling {stance.lower()} case for {ticker}. "
            f"Ignore opposing evidence. Build a rigorous quantitative thesis."
        )
        
        try:
            response = await self.premium_llm.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context}
                ],
                temperature=0.4
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"{stance} Agent failed: {e}")
            return f"{stance} thesis unavailable due to API error."

    async def _synthesis_arbitration(self, ticker: str, bull_thesis: str, bear_thesis: str) -> Dict[str, float]:
        """
        The Synthesis Agent reviews both arguments, detects logical flaws,
        and outputs the final probability distribution.
        """
        system_prompt = (
            "You are the Synthesis Arbiter. Review the Bull and Bear theses. "
            "Identify logical leaps or ignored variables. "
            "You must output ONLY valid JSON matching the requested probability format."
        )
        
        prompt = (
            f"{CHAIN_OF_THOUGHT_TEMPLATE.format(ticker=ticker)}\n\n"
            f"--- BULL THESIS ---\n{bull_thesis}\n\n"
            f"--- BEAR THESIS ---\n{bear_thesis}\n\n"
            f"Provide the final JSON probability distribution."
        )
        
        try:
            response = await self.premium_llm.chat.completions.create(
                model="gpt-4o",
                response_format={ "type": "json_object" }, # Enforce JSON output
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1 # Very low temperature for highly deterministic parsing
            )
            
            # Parse the JSON string
            content = response.choices[0].message.content
            distribution = json.loads(content)
            return self._normalize_distribution(distribution)
            
        except Exception as e:
            logger.error(f"Synthesis Agent failed: {e}")
            return self._get_neutral_distribution()

    async def _fast_path_estimation(self, ticker: str, context: str) -> np.ndarray:
        """Fallback for low novelty news. Returns a flat/neutral distribution to save compute."""
        return self._dict_to_alpha_vector(self._get_neutral_distribution())

    def _normalize_distribution(self, dist: Dict[str, float]) -> Dict[str, float]:
        """Ensures the probabilities sum exactly to 1.0 (graceful degradation)."""
        keys = ["crash", "decline", "flat", "rise", "surge"]
        total = sum(dist.get(k, 0.0) for k in keys)
        if total == 0:
            return self._get_neutral_distribution()
        return {k: dist.get(k, 0.0) / total for k in keys}

    def _get_neutral_distribution(self) -> Dict[str, float]:
        """Fallback distribution if parsing fails."""
        return {"crash": 0.05, "decline": 0.20, "flat": 0.50, "rise": 0.20, "surge": 0.05}

    def _dict_to_alpha_vector(self, dist: Dict[str, float]) -> np.ndarray:
        """Converts the JSON dict into a 5-dim numpy array for the GAT."""
        return np.array([
            dist["crash"], dist["decline"], dist["flat"], dist["rise"], dist["surge"]
        ], dtype=np.float32)