#!/usr/bin/env python3
"""
Harnyx SN67 Miner Agent - Optimized for Champion Performance v2

Scoring: total_score = 0.5 * comparison_score + 0.5 * similarity_score
Ties broken by: lower total tool cost wins

v2 Improvements:
- Time budget tracking to avoid 120s timeout
- Reduced LLM calls (skip strategy generation)
- Faster parallel search execution  
- Cached model selection
- Better error recovery
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
    source_type: str  # "web" or "ai"
    full_content: str = ""
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
# TIME BUDGET TRACKING
# ============================================================================

class TimeBudget:
    """Track time to avoid 120s timeout."""
    
    def __init__(self, timeout_seconds: float = 115.0):
        self.start_time = time.time()
        self.timeout = timeout_seconds
        self.warnings_issued = 0
    
    def elapsed(self) -> float:
        return time.time() - self.start_time
    
    def remaining(self) -> float:
        return max(0, self.timeout - self.elapsed())
    
    def should_abort(self) -> bool:
        """Check if we should abort to avoid timeout."""
        return self.elapsed() >= self.timeout
    
    def check_time(self, operation: str) -> bool:
        """Check time and log warning if running low."""
        elapsed = self.elapsed()
        remaining = self.remaining()
        
        if remaining < 10 and self.warnings_issued == 0:
            logger.warning(f"⏰ Time warning ({operation}): {remaining:.1f}s remaining")
            self.warnings_issued = 1
        
        if self.should_abort():
            logger.error(f"⏰ TIMEOUT approaching ({operation}): aborting after {elapsed:.1f}s")
            return False
        
        return True
    
    def has_time_for(self, estimated_seconds: float, buffer: float = 5.0) -> bool:
        """Check if we have enough time for an operation."""
        return self.remaining() > (estimated_seconds + buffer)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def extract_llm_text(llm_result: Any) -> str:
    """
    Robust LLM response extraction.
    Handles various SDK response formats (ToolResultDTO, nested structures, etc.)
    """
    if not llm_result:
        return ""
    
    # Try .results list first (SDK standard)
    if hasattr(llm_result, "results") and llm_result.results:
        for item in llm_result.results:
            # Direct .text attribute
            if hasattr(item, "text") and item.text:
                return str(item.text).strip()
            # OpenAI-style .message.content
            if hasattr(item, "message") and item.message:
                content = getattr(item.message, "content", "")
                if isinstance(content, list):
                    # Multi-part content (text blocks)
                    texts = [getattr(p, "text", "") for p in content if hasattr(p, "text")]
                    combined = "".join(texts)
                    if combined.strip():
                        return combined.strip()
                if content:
                    return str(content).strip()
    
    # Try .response dict
    if hasattr(llm_result, "response") and llm_result.response:
        resp = llm_result.response
        if isinstance(resp, dict):
            if "choices" in resp and resp["choices"]:
                choice = resp["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    return str(choice["message"]["content"]).strip()
        elif hasattr(resp, "text"):
            return str(resp.text).strip()
    
    # Last resort: stringify and extract
    try:
        text = str(llm_result)
        # Try JSON extraction
        json_match = re.search(r'\{.*"text".*\}', text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            if "text" in data:
                return str(data["text"]).strip()
        # Try direct "text": "..." pattern
        text_match = re.search(r'"text":\s*"([^"]+)"', text, re.DOTALL)
        if text_match:
            return text_match.group(1).strip()
    except Exception:
        pass
    
    return str(llm_result)[:2000]  # Safety truncate


def compute_relevance(query: str, result: SearchResult) -> float:
    """
    Score result relevance to query.
    Higher score = better match for synthesis.
    """
    query_words = set(query.lower().split())
    # Remove common stopwords for better matching
    stopwords = {"the", "a", "an", "is", "are", "what", "how", "why", "when", "where", "which", "that", "this", "for", "and", "or", "but", "in", "on", "at", "to", "of"}
    query_words = query_words - stopwords
    
    if not query_words:
        return 0.5  # Neutral if no meaningful query words
    
    # Search in title, note, and content
    text = f"{result.title} {result.note} {result.full_content}".lower()
    text_words = set(text.split())
    
    # Jaccard-like overlap score
    overlap = len(query_words & text_words)
    score = overlap / max(len(query_words), 1)
    
    # Boost for title matches (usually more relevant)
    title_words = set(result.title.lower().split())
    if query_words & title_words:
        score += 0.2
    
    return min(score, 1.0)


def generate_search_variants(query: str) -> List[str]:
    """
    Generate search variants without LLM call.
    Fast, deterministic approach.
    """
    variants = [query]
    
    # Extract key terms and create variations
    words = query.split()
    
    # If query is longer than 5 words, create focused variant
    if len(words) > 5:
        # Take first 5 and last 5 words
        focused = " ".join(words[:5] + words[-5:])
        if focused != query:
            variants.append(focused)
    
    # If query has question words, create statement variant
    question_words = {"what", "how", "why", "when", "where", "who", "which"}
    if any(q in query.lower() for q in question_words):
        # Remove question words and convert to keyword search
        keywords = [w for w in words if w.lower() not in question_words and len(w) > 2]
        if keywords:
            keyword_search = " ".join(keywords[:8])
            if keyword_search not in variants:
                variants.append(keyword_search)
    
    # Limit to 4 variants
    return variants[:4]


# ============================================================================
# MAIN AGENT
# ============================================================================

@entrypoint("query")
async def query(q: Query) -> Response:
    """
    Harnyx SN67 miner entrypoint v2.
    
    Strategy:
    1. Get tool info and initialize time budget
    2. Generate search variants (no LLM call)
    3. Execute parallel searches (fast, concurrency-limited)
    4. Rank results by relevance
    5. Synthesize answer with targeted citations
    6. Return early if time is running out
    
    Scoring optimization:
    - Stay under 120s timeout (critical)
    - Maximize answer quality (comparison_score + similarity_score)
    - Use fewer LLM calls (reduce latency and cost)
    - Precise citations (only load-bearing facts)
    """
    
    query_text = q.text.strip()
    budget = BudgetState()
    time_budget = TimeBudget(timeout_seconds=115.0)
    
    logger.info(f"🎯 Query: {query_text[:100]}...")
    
    try:
        # ================================================================
        # STEP 1: Get tooling info and budget (check time)
        # ================================================================
        if not time_budget.check_time("init"):
            return Response(text=f"Time limit exceeded", citations=[])
        
        info = await tooling_info()
        budget.update(info)
        
        # Select best available model (prefer cost-effective + capable)
        allowed_models = info.response.get("allowed_tool_models", [])
        model_priority = [
            "Qwen/Qwen3-Next-80B-A3B-Instruct",  # Best cost/quality ratio
            "openai/gpt-oss-120b-TEE",           # High quality, higher cost
            "openai/gpt-oss-20b-TEE",            # Fallback
        ]
        model = next((m for m in model_priority if m in allowed_models), 
                     allowed_models[0] if allowed_models else "openai/gpt-oss-20b-TEE")
        
        # ================================================================
        # STEP 2: Generate search variants (fast, no LLM)
        # ================================================================
        if not time_budget.check_time("search_variants"):
            return Response(text=f"Time limit exceeded", citations=[])
        
        search_prompts = generate_search_variants(query_text)
        max_per_search = 6  # Reduced from 8 for speed
        
        logger.info(f"📋 Strategy: {len(search_prompts)} searches, {max_per_search} results each | Model: {model}")
        
        # ================================================================
        # STEP 3: Execute parallel web searches (concurrency-limited)
        # ================================================================
        if not time_budget.check_time("searches"):
            return Response(text=f"Time limit exceeded", citations=[])
        
        all_results: List[SearchResult] = []
        search_semaphore = asyncio.Semaphore(3)  # Increased from 2 for speed
        
        async def execute_search(prompt: str) -> List[SearchResult]:
            """Execute a single search with budget check."""
            async with search_semaphore:
                if not budget.can_afford(0.009, buffer=0.001):
                    logger.warning(f"⏸️ Skipping search - budget low")
                    return []
                
                try:
                    result = await search_web(prompt, num=max_per_search)
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
                    logger.info(f"✅ Search '{prompt[:30]}...' → {len(results)} results")
                    return results
                    
                except Exception as e:
                    logger.warning(f"❌ Search failed: {e}")
                    return []
        
        # Execute all searches in parallel
        search_tasks = [execute_search(prompt) for prompt in search_prompts]
        search_batches = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        for batch in search_batches:
            if isinstance(batch, list):
                all_results.extend(batch)
        
        # ================================================================
        # STEP 4: Quick AI search if time allows (boost quality)
        # ================================================================
        if all_results and time_budget.has_time_for(8) and budget.can_afford(0.015):
            try:
                ai_result = await search_ai(search_prompts[0])
                budget.update(ai_result)
                for res in getattr(ai_result, "results", []):
                    all_results.append(SearchResult(
                        receipt_id=getattr(ai_result, "receipt_id", ""),
                        result_id=getattr(res, "result_id", ""),
                        url=getattr(res, "url", ""),
                        title=getattr(res, "title", "") or "AI Result",
                        note=getattr(res, "note", ""),
                        source_type="ai"
                    ))
                logger.info(f"✅ AI search → {len(ai_result.results)} results")
            except Exception as e:
                logger.warning(f"❌ AI search failed: {e}")
        
        # ================================================================
        # STEP 5: Fallback if no results (urgent)
        # ================================================================
        if not all_results:
            logger.warning("⚠️ No results - trying fallback")
            fallback_results = await execute_search(query_text)
            all_results.extend(fallback_results)
        
        if not all_results:
            # Ultimate fallback: direct answer without citations
            logger.warning("⚠️ No search results - using direct LLM")
            if time_budget.has_time_for(10):
                try:
                    fallback_llm = await llm_chat(
                        messages=[{"role": "user", "content": f"Answer concisely: {query_text}"}],
                        model=model,
                        temperature=0.1,
                        max_tokens=600,
                    )
                    answer_text = extract_llm_text(fallback_llm)
                    if answer_text.strip():
                        return Response(text=answer_text, citations=[])
                except Exception:
                    pass
            return Response(text=f"Unable to find relevant information for: {query_text}", citations=[])
        
        # ================================================================
        # STEP 6: Rank results by relevance
        # ================================================================
        if not time_budget.check_time("ranking"):
            # If time is low, just use unranked results
            ranked_results = all_results[:12]
        else:
            for r in all_results:
                r.relevance_score = compute_relevance(query_text, r)
            ranked_results = sorted(all_results, key=lambda x: x.relevance_score, reverse=True)[:12]
        
        logger.info(f"📊 Ranked {len(ranked_results)} results | Time: {time_budget.elapsed():.1f}s")
        
        # Build context for synthesis
        context_parts = []
        for i, r in enumerate(ranked_results):
            part = f"[{i+1}] {r.title} ({r.source_type})\n"
            part += f"Snippet: {r.note}\n"
            context_parts.append(part)
        
        context = "\n".join(context_parts)
        
        # ================================================================
        # STEP 7: Synthesize final answer (check time)
        # ================================================================
        if not time_budget.has_time_for(15):
            # Not enough time for synthesis - return structured summary
            summary = f"**Answer Summary for: {query_text}**\n\nBased on the search results:\n\n"
            for i, r in enumerate(ranked_results[:5]):
                summary += f"{i+1}. {r.title}\n   {r.note[:200]}...\n\n"
            
            citations = [
                CitationRef(receipt_id=r.receipt_id, result_id=r.result_id)
                for r in ranked_results[:3]
                if r.receipt_id and r.result_id
            ]
            
            logger.info(f"✅ Early return | Time: {time_budget.elapsed():.1f}s")
            return Response(text=summary.strip(), citations=citations)
        
        try:
            synthesis_prompt = f"""Synthesize a precise answer using the evidence.

Query: {query_text}

Evidence:
{context}

Guidelines:
- Answer directly and completely
- Be factual and concise
- Match reference answer style
- Use [N] notation for sources

Answer:"""

            final_llm = await llm_chat(
                messages=[{"role": "user", "content": synthesis_prompt}],
                model=model,
                temperature=0.05,
                max_tokens=1500,
            )
            budget.update(final_llm)
            
            answer_text = extract_llm_text(final_llm)
            
            # Fallback if synthesis failed
            if not answer_text.strip():
                answer_text = f"**Research: {query_text}**\n\n"
                for i, r in enumerate(ranked_results[:6]):
                    answer_text += f"{i+1}. {r.title}\n   {r.note[:250]}...\n\n"
            
        except Exception as e:
            logger.warning(f"⚠️ Synthesis failed: {e}, using structured summary")
            answer_text = f"**Answer for: {query_text}**\n\n"
            for i, r in enumerate(ranked_results[:5]):
                answer_text += f"{i+1}. {r.title}\n   {r.note[:200]}...\n\n"
        
        # ================================================================
        # STEP 8: Build citations (only load-bearing facts)
        # ================================================================
        citation_results = ranked_results[:min(4, len(ranked_results))]
        citations = [
            CitationRef(receipt_id=r.receipt_id, result_id=r.result_id)
            for r in citation_results
            if r.receipt_id and r.result_id
        ]
        
        logger.info(f"✅ Success | Time: {time_budget.elapsed():.1f}s | Budget: ${budget.remaining:.4f} | Citations: {len(citations)}")
        
        return Response(
            text=answer_text.strip(),
            citations=citations
        )
        
    except Exception as e:
        logger.exception(f"💥 Pipeline error: {e}")
        # Graceful degradation - always return something
        return Response(
            text=f"Query processed: {query_text}. (Error: {type(e).__name__})",
            citations=[]
        )
