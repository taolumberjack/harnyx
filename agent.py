#!/usr/bin/env python3
"""
Harnyx SN67 Miner Agent - Speed Demon v3

SCORING PHILOSOPHY CHANGED:
From: "Perfect scores on 5 tasks" = 2.244 total ❌
To: "Good scores on ALL 10 tasks" = 7.0+ total ✅

To surpass 7.58, we MUST complete all tasks.
Tradeoff: Faster execution, simpler answers, no luxuries.

v3 Changes:
- Remove AI search (slow)
- Remove page fetch (slow)
- Reduce searches: 4 → 2
- Simplified synthesis or structured summary
- Aggressive per-operation timeouts
- Fail fast logic: partial answer > timeout

If scoring: faster + complete > slower + timeout
"""

from harnyx_miner_sdk.decorators import entrypoint
from harnyx_miner_sdk.query import Query, Response, CitationRef
from harnyx_miner_sdk.api import tooling_info, llm_chat, search_web, search_ai, fetch_page

import asyncio
import time
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class SearchResult:
    """Normalized search result with relevance tracking."""
    receipt_id: str
    result_id: str
    url: str
    title: str
    note: str
    source_type: str = "web"
    relevance_score: float = 0.0


@dataclass
class BudgetState:
    """Track budget consumption across tool calls."""
    remaining: float = 0.0
    hard_limit: float = 0.0
    used: float = 0.0
    pricing: Dict[str, Any] = field(default_factory=dict)
    
    def update(self, tool_response: Any) -> None:
        """Extract budget info from any tool response."""
        if hasattr(tool_response, "session_remaining_budget_usd"):
            self.remaining = float(getattr(tool_response, "session_remaining_budget_usd", 0.0))
        if hasattr(tool_response, "session_hard_limit_usd"):
            self.hard_limit = float(getattr(tool_response, "session_hard_limit_usd", 0.0))
        if hasattr(tool_response, "session_used_budget_usd"):
            self.used = float(getattr(tool_response, "session_used_budget_usd", 0.0))
        if hasattr(tool_response, "response") and isinstance(tool_response.response, dict):
            self.pricing.update(tool_response.response.get("pricing", {}))
    
    def can_afford(self, cost: float, buffer: float = 0.002) -> bool:
        """Check if we can afford a tool call with safety buffer."""
        return self.remaining > (cost + buffer)


# ============================================================================
# TIMEOUT TRACKING
# ============================================================================

class OpTimer:
    """Aggressive timeout tracking for per-operation limits."""
    
    def __init__(self, op_timeout: float = 12.0):
        self.start_time = time.time()
        self.timeout = op_timeout
    
    def check(self) -> bool:
        """Return True if still within time budget."""
        return (time.time() - self.start_time) < self.timeout
    
    def elapsed(self) -> float:
        return time.time() - self.start_time
    
    def remaining(self) -> float:
        return max(0, self.timeout - self.elapsed())


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def extract_llm_text(llm_result: Any) -> str:
    """Robust LLM response extraction."""
    if not llm_result:
        return ""
    
    if hasattr(llm_result, "results") and llm_result.results:
        for item in llm_result.results:
            if hasattr(item, "text") and item.text:
                return str(item.text).strip()
            if hasattr(item, "message") and item.message:
                content = getattr(item.message, "content", "")
                if isinstance(content, list):
                    texts = [getattr(p, "text", "") for p in content if hasattr(p, "text")]
                    combined = "".join(texts)
                    if combined.strip():
                        return combined.strip()
                if content:
                    return str(content).strip()
    
    if hasattr(llm_result, "response") and llm_result.response:
        resp = llm_result.response
        if isinstance(resp, dict):
            if "choices" in resp and resp["choices"]:
                choice = resp["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    return str(choice["message"]["content"]).strip()
        elif hasattr(resp, "text"):
            return str(resp.text).strip()
    
    try:
        text = str(llm_result)
        json_match = re.search(r'\{.*"text".*\}', text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            if "text" in data:
                return str(data["text"]).strip()
        text_match = re.search(r'"text":\s*"([^"]+)"', text, re.DOTALL)
        if text_match:
            return text_match.group(1).strip()
    except Exception:
        pass
    
    return str(llm_result)[:2000]


def compute_relevance(query: str, result: SearchResult) -> float:
    """Score result relevance to query."""
    query_words = set(query.lower().split())
    stopwords = {"the", "a", "an", "is", "are", "what", "how", "why", "when", "where", "which", "that", "this", "for", "and", "or", "but", "in", "on", "at", "to", "of"}
    query_words = query_words - stopwords
    
    if not query_words:
        return 0.5
    
    text = f"{result.title} {result.note}".lower()
    text_words = set(text.split())
    
    overlap = len(query_words & text_words)
    score = overlap / max(len(query_words), 1)
    
    title_words = set(result.title.lower().split())
    if query_words & title_words:
        score += 0.2
    
    return min(score, 1.0)


def parse_json_safely(text: str, default: Any) -> Any:
    """Extract and parse JSON from LLM output."""
    try:
        json_match = re.search(r'\{.*\}', text, re.DOTALL | re.IGNORECASE)
        if json_match:
            return json.loads(json_match.group(0))
    except Exception:
        pass
    return default


# ============================================================================
# MAIN AGENT
# ============================================================================

@entrypoint("query")
async def query(q: Query) -> Response:
    """
    Harnyx SN67 miner - Speed Demon v3.
    
    PHILOSOPHY: Complete ALL tasks with good scores, not perfect scores on half.
    
    Strategy:
    1. Fast web searches (2 max)
    2. Quick relevance ranking
    3. Minimal LLM synthesis or structured summary
    4. Aggressive timeouts (fail fast)
    5. Always return something (partial > timeout)
    """
    
    query_text = q.text.strip()
    budget = BudgetState()
    global_timer = OpTimer(op_timeout=70.0)  # 70s total budget
    
    logger.info(f"⚡ Speed Demon v3: {query_text[:60]}...")
    
    try:
        # ================================================================
        # STEP 1: Get tooling info (must be fast)
        # ================================================================
        info = await tooling_info()
        budget.update(info)
        
        # Select model
        allowed_models = info.response.get("allowed_tool_models", [])
        model_priority = [
            "Qwen/Qwen3-Next-80B-A3B-Instruct",
            "openai/gpt-oss-20b-TEE",            # Prefer 20b for speed!
            "openai/gpt-oss-120b-TEE",           # Only if 20b not available
        ]
        model = next((m for m in model_priority if m in allowed_models), 
                     allowed_models[0] if allowed_models else "openai/gpt-oss-20b-TEE")
        
        logger.info(f"⚡ Model: {model}")
        
        # ================================================================
        # STEP 2: Execute FAST web searches (2 max, aggressive timeout)
        # ================================================================
        all_results: List[SearchResult] = []
        
        # Generate 2 search prompts efficiently
        prompts = [query_text]
        if len(query_text.split()) > 5:
            # Create focused variant if query is long
            words = query_text.split()
            focused = " ".join(words[:5] + words[-5:])
            if focused != query_text:
                prompts.append(focused)
        
        prompts = prompts[:2]  # Max 2 searches
        
        async def fast_search(prompt: str) -> List[SearchResult]:
            """Execute search with aggressive 8s timeout."""
            if not global_timer.check():
                logger.warning("⏰ Global timeout - skipping search")
                return []
            
            try:
                result = await asyncio.wait_for(
                    search_web(prompt, num=6),
                    timeout=8.0  # 8s max per search
                )
                budget.update(result)
                
                results = []
                for res in getattr(result, "results", []):
                    results.append(SearchResult(
                        receipt_id=getattr(result, "receipt_id", ""),
                        result_id=getattr(res, "result_id", ""),
                        url=getattr(res, "url", ""),
                        title=getattr(res, "title", "") or "Untitled",
                        note=getattr(res, "note", ""),
                        source_type="web",
                    ))
                return results
                
            except asyncio.TimeoutError:
                logger.warning(f"⏰ Search timeout: '{prompt[:30]}...'")
                return []
            except Exception as e:
                logger.warning(f"❌ Search failed: {e}")
                return []
        
        # Execute searches in parallel
        search_tasks = [fast_search(p) for p in prompts]
        search_batches = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        for batch in search_batches:
            if isinstance(batch, list):
                all_results.extend(batch)
        
        logger.info(f"⚡ Got {len(all_results)} results in {global_timer.elapsed():.1f}s")
        
        # ================================================================
        # STEP 3: Check if time for LLM synthesis
        # ================================================================
        if not all_results:
            # No results - ultra-fast fallback
            logger.warning("⚠️ No results")
            return Response(
                text=f"Unable to find relevant information for: {query_text}",
                citations=[]
            )
        
        # Rank by relevance
        for r in all_results:
            r.relevance_score = compute_relevance(query_text, r)
        
        ranked_results = sorted(all_results, key=lambda x: x.relevance_score, reverse=True)[:8]
        
        # ================================================================
        # STEP 4: Decide synthesis strategy based on time
        # ================================================================
        time_left = global_timer.remaining()
        
        # If <15s left, use structured summary (NO LLM)
        if time_left < 15:
            logger.info(f"⚡ Time low ({time_left:.1f}s) - using structured summary")
            
            summary = f"**Answer for: {query_text}**\n\n"
            for i, r in enumerate(ranked_results[:5]):
                summary += f"{i+1}. **{r.title}**\n   {r.note[:200]}...\n\n"
            
            citations = [
                CitationRef(receipt_id=r.receipt_id, result_id=r.result_id)
                for r in ranked_results[:3]
                if r.receipt_id and r.result_id
            ]
            
            return Response(text=summary.strip(), citations=citations)
        
        # ================================================================
        # STEP 5: Quick LLM synthesis (15s max)
        # ================================================================
        # Build minimal context
        context = ""
        for i, r in enumerate(ranked_results[:5]):
            context += f"[{i+1}] {r.title}\n{r.note}\n\n"
        
        synthesis_prompt = f"""Answer this question using the evidence. Be concise and direct.

Question: {query_text}

Evidence:
{context}

Answer:"""
        
        try:
            final_llm = await asyncio.wait_for(
                llm_chat(
                    messages=[{"role": "user", "content": synthesis_prompt}],
                    model=model,
                    temperature=0.0,
                    max_tokens=600,  # Reduced for speed
                ),
                timeout=15.0  # 15s max for LLM
            )
            budget.update(final_llm)
            
            answer_text = extract_llm_text(final_llm)
            
            if not answer_text.strip():
                # Fallback to structured summary
                answer_text = f"**Answer for: {query_text}**\n\n"
                for i, r in enumerate(ranked_results[:5]):
                    answer_text += f"{i+1}. {r.title}\n   {r.note[:180]}...\n\n"
            
            citations = [
                CitationRef(receipt_id=r.receipt_id, result_id=r.result_id)
                for r in ranked_results[:3]
                if r.receipt_id and r.result_id
            ]
            
            logger.info(f"⚡ Complete in {global_timer.elapsed():.1f}s | {len(citations)} citations")
            return Response(text=answer_text.strip(), citations=citations)
            
        except asyncio.TimeoutError:
            logger.warning("⏰ LLM synthesis timeout - using structured summary")
            
            summary = f"**Answer for: {query_text}**\n\n"
            for i, r in enumerate(ranked_results[:5]):
                summary += f"{i+1}. {r.title}\n   {r.note[:200]}...\n\n"
            
            citations = [
                CitationRef(receipt_id=r.receipt_id, result_id=r.result_id)
                for r in ranked_results[:3]
                if r.receipt_id and r.result_id
            ]
            
            return Response(text=summary.strip(), citations=citations)
        
    except Exception as e:
        logger.exception(f"💥 Error: {e}")
        return Response(
            text=f"Query processed: {query_text}. (Error: {type(e).__name__})",
            citations=[]
        )
