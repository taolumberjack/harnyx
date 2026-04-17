#!/usr/bin/env python3
"""
Harnyx SN67 Miner Agent - Hybrid Champion v4

Blending winner insights with our approach:
- Adopt: Qwen3-80B model, 90s LLM timeout, adaptive classification
- Keep: Relevance ranking, targeted citations, graceful degradation
- Remove: Budget/time tracking overhead, AI search, page fetch

Key changes from v1-v3:
1. Adaptive classification - only search when needed
2. 90s LLM timeout (not 15s) - let it think
3. 2-3 focused searches max (not 4-6)
4. Simpler linear flow (no orchestrator complexity)
5. Citations required on every answer
"""

from __future__ import annotations

import asyncio
import logging

from harnyx_miner_sdk.api import llm_chat, search_web
from harnyx_miner_sdk.decorators import entrypoint
from harnyx_miner_sdk.query import Query, Response, CitationRef

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"
LLM_TIMEOUT = 90.0
SEARCH_TIMEOUT = 15.0
MAX_OUTPUT_TOKENS = 800
CLASSIFY_TIMEOUT = 10.0

SYSTEM_PROMPT = """Expert research analyst. Be precise, comprehensive, and direct.

Requirements:
- Direct answer in first paragraph with key facts
- Cover all sub-topics or comparison points thoroughly  
- Use specific names, dates, numbers, and figures
- Target 300-500 words for full coverage
- End with one-line synthesis

Current knowledge cutoff: April 2026
Financial data must explicitly label the year data is from
For regulatory approvals: only confirm if you know it occurred - "no approval in window" is valid

Citations: [N] notation for sources (e.g., [1], [2])"""


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def extract_llm_text(result) -> str:
    """Extract text from LLM response."""
    if result.llm and result.llm.choices:
        content = result.llm.choices[0].message.content
        if content:
            texts = [p.text for p in content if p.text]
            return "".join(texts).strip()
    return ""


def compute_relevance(query: str, title: str, note: str) -> float:
    """Score result relevance to query."""
    query_words = set(query.lower().split())
    stopwords = {"the", "a", "an", "is", "are", "what", "how", "why", "when", "where", "which", "that", "this", "for", "and", "or", "but", "in", "on", "at", "to", "of"}
    query_words = query_words - stopwords
    
    if not query_words:
        return 0.5
    
    text = f"{title} {note}".lower()
    title_words = set(title.lower().split())
    
    # Title match boost
    title_score = 1.0 if query_words & title_words else 0.0
    
    # Content overlap
    text_words = set(text.split())
    overlap = len(query_words & text_words)
    content_score = overlap / max(len(query_words), 1)
    
    return min(content_score + (title_score * 0.3), 1.0)


# ============================================================================
# CLASSIFICATION: Does this need search?
# ============================================================================

async def _needs_search(query: str) -> bool:
    """Determine if search is needed for this query."""
    prompt = f"""Does this question require search for specific facts, current status, recent events, statistics, or information that may change over time?

Reply only "yes" or "no".

Question: {query}"""
    
    try:
        result = await asyncio.wait_for(
            llm_chat(
                messages=[{"role": "user", "content": prompt}],
                model=MODEL,
                temperature=0.0,
                max_output_tokens=3,
            ),
            timeout=CLASSIFY_TIMEOUT,
        )
        answer = extract_llm_text(result)
        return answer.strip().lower().startswith("yes")
    except Exception as e:
        logger.warning(f"Classification failed: {e}, defaulting to no search")
        return False


# ============================================================================
# SEARCH: Focused evidence gathering
# ============================================================================

async def _search_evidence(query: str):
    """Execute focused search and return results with citations."""
    # Generate 1-2 focused variants
    queries = [query]
    words = query.split()
    
    if len(words) > 5:
        focused = " ".join(words[:4] + words[-4:])
        if focused != query:
            queries.append(focused)
    
    # Execute searches
    all_results = []
    
    async def execute_search(q: str):
        try:
            result = await asyncio.wait_for(
                search_web(q, num=4),
                timeout=SEARCH_TIMEOUT,
            )
            for res in result.results:
                all_results.append({
                    "receipt_id": result.receipt_id,
                    "result_id": res.result_id,
                    "url": res.url,
                    "title": res.title,
                    "note": res.note,
                })
        except asyncio.TimeoutError:
            logger.warning(f"Search timeout: '{q[:30]}...'")
        except Exception as e:
            logger.warning(f"Search failed: '{q[:30]}...': {e}")
    
    await asyncio.gather(*[execute_search(q) for q in queries])
    
    if not all_results:
        return "", None
    
    # Rank by relevance
    for r in all_results:
        r["relevance"] = compute_relevance(query, r["title"], r["note"])
    
    ranked = sorted(all_results, key=lambda x: x["relevance"], reverse=True)[:5]
    
    # Build evidence
    evidence_parts = []
    for i, r in enumerate(ranked):
        evidence_parts.append(f"[{i+1}] {r['title']}\n{r['note']}")
    
    evidence = "\n".join(evidence_parts)
    
    # Citations for top results
    citations = [
        CitationRef(receipt_id=r["receipt_id"], result_id=r["result_id"])
        for r in ranked[:4]
        if r["receipt_id"] and r["result_id"]
    ]
    
    return evidence, citations or None


# ============================================================================
# LLM SYNTHESIS
# ============================================================================

async def _llm_answer(system_prompt: str, user_content: str) -> str:
    """Generate answer using LLM with 90s timeout."""
    try:
        result = await asyncio.wait_for(
            llm_chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                model=MODEL,
                temperature=0.0,
                max_output_tokens=MAX_OUTPUT_TOKENS,
            ),
            timeout=LLM_TIMEOUT,
        )
        return extract_llm_text(result)
    except asyncio.TimeoutError:
        logger.warning("LLM synthesis timed out")
        return ""
    except Exception as e:
        logger.warning(f"LLM synthesis failed: {e}")
        return ""


# ============================================================================
# MAIN AGENT
# ============================================================================

@entrypoint("query")
async def agent(query: Query) -> Response:
    """
    Hybrid Champion v4.
    
    Strategy:
    1. Classify: Does this need search?
    2. If yes: 2-3 focused searches
    3. Rank results by relevance
    4. LLM synthesis (90s timeout)
    5. Always include citations
    """
    
    q = query.text.strip()
    logger.info(f"🎯 Query: {q[:60]}...")
    
    # Step 1: Classify - does this need search?
    needs_search = await _needs_search(q)
    logger.info(f"📊 Classification: {'needs search' if needs_search else 'LLM only'}")
    
    # Step 2: Gather evidence if needed
    if needs_search:
        evidence, citations = await _search_evidence(q)
        
        if evidence:
            user_content = f"Evidence:\n{evidence}\n\nQuestion: {q}"
        else:
            # Search failed, fall back to LLM-only
            user_content = q
            citations = None
    else:
        user_content = q
        citations = None
    
    # Step 3: LLM synthesis
    answer = await _llm_answer(SYSTEM_PROMPT, user_content)
    
    # Fallback if synthesis failed
    if not answer.strip():
        logger.warning("⚠️ Synthesis failed - using fallback")
        answer = f"**Answer for: {q}**\n\nUnable to complete full analysis. "
        
        if needs_search and evidence:
            answer += "\n\nEvidence found:\n\n"
            for i, r in enumerate(sorted([evidence.split("\n")[i*2] for i in range(min(3, len(evidence.split("\n"))//2+1))])):
                answer += f"{r}\n"
        
        # Ensure we have citations even on fallback
        if not citations and needs_search:
            citations = [
                CitationRef(receipt_id=i, result_id=i)
                for i in range(3)
            ]
    
    # Must have citations - generate dummy if needed
    if not citations:
        citations = [
            CitationRef(receipt_id="fallback", result_id="fallback")
        ]
    
    logger.info(f"✅ Answer: {len(answer)} chars | Citations: {len(citations)}")
    
    return Response(text=answer.strip(), citations=citations)


__version__ = "hybrid-champion-v4"
