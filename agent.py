#!/usr/bin/env python3
"""
Harnyx SN67 Miner Agent - Optimized for Champion Performance

Scoring: total_score = 0.5 * comparison_score + 0.5 * similarity_score
Ties broken by: lower total tool cost wins

Strategy:
1. Budget-aware tool usage (maximize quality per $)
2. Precise search → high-relevance results
3. Synthesis that matches reference answer style (factual, concise, complete)
4. Targeted citations (only load-bearing facts)
5. Graceful degradation on errors
"""

from harnyx_miner_sdk.decorators import entrypoint
from harnyx_miner_sdk.query import Query, Response, CitationRef
from harnyx_miner_sdk.api import tooling_info, llm_chat, search_web, search_ai, fetch_page

import asyncio
import json
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
    estimated_cost: float = 0.008  # default search cost


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
    
    def get_tool_cost(self, tool_name: str) -> float:
        """Get estimated cost for a tool from pricing info."""
        if tool_name in self.pricing:
            pricing = self.pricing[tool_name]
            if isinstance(pricing, dict):
                return float(pricing.get("per_call", pricing.get("default", 0.01)))
        # Defaults based on typical Harnyx pricing
        defaults = {"search_web": 0.008, "search_ai": 0.012, "fetch_page": 0.005, "llm_chat": 0.003}
        return defaults.get(tool_name, 0.01)


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


def parse_json_safely(text: str, default: Any) -> Any:
    """Extract and parse JSON from LLM output, with fallback."""
    try:
        # Try to find JSON object in text
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
    Harnyx SN67 miner entrypoint.
    
    Strategy:
    1. Analyze query and plan search strategy
    2. Execute parallel searches (budget-aware)
    3. Fetch pages for top results (if budget allows)
    4. Rank results by relevance
    5. Synthesize answer with targeted citations
    
    Scoring optimization:
    - Maximize answer quality (comparison_score + similarity_score)
    - Minimize tool cost (tie-breaker)
    - Use precise citations (only load-bearing facts)
    """
    
    query_text = q.text.strip()
    budget = BudgetState()
    
    logger.info(f"🎯 Query: {query_text[:100]}...")
    
    try:
        # ================================================================
        # STEP 1: Get tooling info and budget
        # ================================================================
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
        
        logger.info(f"💰 Budget: ${budget.remaining:.4f} | Model: {model}")
        
        # ================================================================
        # STEP 2: Plan search strategy (lightweight LLM call)
        # ================================================================
        strategy_prompt = f"""You are a search strategist for a research AI. Generate diverse search queries.

Return ONLY valid JSON:
{{
  "search_prompts": ["original query", "variant 1", "variant 2"],
  "needs_page_fetch": true,
  "max_results_per_search": 5
}}

Query: {query_text}
Budget: ${budget.remaining:.4f} (be efficient)"""

        strategy_result = await llm_chat(
            messages=[{"role": "user", "content": strategy_prompt}],
            model=model,
            temperature=0.0,      # Deterministic for consistency
            max_tokens=400,       # Keep it short
        )
        budget.update(strategy_result)
        
        strategy = parse_json_safely(extract_llm_text(strategy_result), {
            "search_prompts": [query_text],
            "needs_page_fetch": True,
            "max_results_per_search": 5
        })
        
        search_prompts = strategy.get("search_prompts", [query_text])[:5]
        max_per_search = min(strategy.get("max_results_per_search", 5), 8)
        needs_fetch = strategy.get("needs_page_fetch", True)
        
        logger.info(f"📋 Strategy: {len(search_prompts)} searches, {max_per_search} results each")
        
        # ================================================================
        # STEP 3: Execute parallel web searches (concurrency-limited)
        # ================================================================
        all_results: List[SearchResult] = []
        search_semaphore = asyncio.Semaphore(2)  # Respect concurrency limit
        
        async def execute_search(prompt: str) -> List[SearchResult]:
            """Execute a single search with budget check."""
            async with search_semaphore:
                search_cost = budget.get_tool_cost("search_web")
                if not budget.can_afford(search_cost):
                    logger.warning(f"⏸️ Skipping search '{prompt[:40]}...' - budget")
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
                            estimated_cost=search_cost / max(1, len(getattr(result, "results", [1])))
                        ))
                    logger.info(f"✅ Search '{prompt[:30]}...' → {len(results)} results")
                    return results
                    
                except Exception as e:
                    logger.warning(f"❌ Search failed '{prompt[:40]}...': {e}")
                    return []
        
        # Execute all searches in parallel
        search_tasks = [execute_search(prompt) for prompt in search_prompts]
        search_batches = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        for batch in search_batches:
            if isinstance(batch, list):
                all_results.extend(batch)
        
        # ================================================================
        # STEP 4: Boost with AI search (if budget allows)
        # ================================================================
        if search_prompts and budget.can_afford(budget.get_tool_cost("search_ai")):
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
        # STEP 5: Fetch full pages for top results (if budget allows)
        # ================================================================
        if all_results and needs_fetch and budget.can_afford(budget.get_tool_cost("fetch_page") * 3):
            # Score and sort for fetching
            for r in all_results:
                r.relevance_score = compute_relevance(query_text, r)
            
            top_for_fetch = sorted(all_results, key=lambda x: x.relevance_score, reverse=True)[:3]
            fetch_semaphore = asyncio.Semaphore(2)
            
            async def fetch_page_content(result: SearchResult) -> SearchResult:
                async with fetch_semaphore:
                    if not result.url:
                        return result
                    try:
                        page_result = await fetch_page(result.url)
                        budget.update(page_result)
                        if hasattr(page_result, "data") and page_result.data:
                            content = getattr(page_result.data[0], "content", "")
                            result.full_content = content[:1500]  # Truncate for context
                    except Exception:
                        pass  # Keep result even if fetch fails
                    return result
            
            fetched_results = await asyncio.gather(
                *(fetch_page_content(r) for r in top_for_fetch),
                return_exceptions=True
            )
            
            # Update original results with fetched content
            for fetched in fetched_results:
                if not isinstance(fetched, Exception):
                    for i, r in enumerate(all_results):
                        if r.url == fetched.url:
                            all_results[i] = fetched
                            break
            
            logger.info(f"✅ Fetched {len(fetched_results)} pages")
        
        # ================================================================
        # STEP 6: Fallback if no results
        # ================================================================
        if not all_results:
            logger.warning("⚠️ No results - trying fallback search")
            fallback_results = await execute_search(query_text)
            all_results.extend(fallback_results)
        
        if not all_results:
            # Ultimate fallback: direct answer without citations
            logger.warning("⚠️ No search results available - using direct LLM")
            fallback_llm = await llm_chat(
                messages=[{"role": "user", "content": f"Answer concisely: {query_text}"}],
                model=model,
                temperature=0.1,
                max_tokens=800,
            )
            budget.update(fallback_llm)
            return Response(
                text=extract_llm_text(fallback_llm) or f"Query processed: {query_text}",
                citations=[]
            )
        
        # ================================================================
        # STEP 7: Rank results by relevance
        # ================================================================
        for r in all_results:
            r.relevance_score = compute_relevance(query_text, r)
        
        ranked_results = sorted(all_results, key=lambda x: x.relevance_score, reverse=True)[:12]
        logger.info(f"📊 Ranked {len(ranked_results)} results (top score: {ranked_results[0].relevance_score:.2f})")
        
        # Build context for synthesis
        context_parts = []
        for i, r in enumerate(ranked_results):
            part = f"[{i+1}] {r.title} ({r.source_type})\n"
            part += f"URL: {r.url}\n"
            part += f"Relevance: {r.relevance_score:.2f}\n"
            part += f"Snippet: {r.note}\n"
            if r.full_content:
                part += f"Content: {r.full_content[:600]}...\n"
            context_parts.append(part)
        
        context = "\n".join(context_parts)
        
        # ================================================================
        # STEP 8: Synthesize final answer
        # ================================================================
        synthesis_prompt = f"""You are a precise research assistant. Synthesize a high-quality answer.

Requirements:
- Be factual, concise, and complete
- Answer the query directly in first paragraph
- Use evidence from provided context
- Reference sources with [N] notation (e.g., "according to source [1]")
- Match the style and completeness of expert reference answers

Query: {query_text}

Evidence Context:
{context}

Answer:"""

        final_llm = await llm_chat(
            messages=[{"role": "user", "content": synthesis_prompt}],
            model=model,
            temperature=0.05,     # Low temp for consistency
            max_tokens=1800,      # Allow detailed answers
        )
        budget.update(final_llm)
        
        answer_text = extract_llm_text(final_llm)
        
        # Fallback if synthesis failed
        if not answer_text.strip():
            answer_text = f"**Research Summary: {query_text}**\n\n"
            for i, r in enumerate(ranked_results[:8]):
                answer_text += f"{i+1}. **{r.title}** ({r.source_type})\n   {r.note[:300]}...\n\n"
        
        # ================================================================
        # STEP 9: Build citations (only load-bearing facts)
        # ================================================================
        # Cite top 3-5 most relevant results that support key claims
        citation_results = ranked_results[:min(5, len(ranked_results))]
        citations = [
            CitationRef(receipt_id=r.receipt_id, result_id=r.result_id)
            for r in citation_results
            if r.receipt_id and r.result_id
        ]
        
        logger.info(f"✅ Success | Budget: ${budget.remaining:.4f} | Citations: {len(citations)}")
        
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
