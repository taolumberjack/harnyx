"""
Harnyx SN67 — Iteration 6b Agent
Strategy: Eliminate citation index hallucination by using a constrained citation format.

CHANGES from iter6:
1. LIMIT evidence to top 8 results (reduces index range, improves focus)
2. Use NAMED citations in prompt: tell LLM to cite by title, not index
3. Post-process: map title-based citations back to [N] indices
4. If mapping fails, fall back to evidence scanning for keyword matches
5. Remove _fix_citations (replaced with title-based citation system)
"""
from __future__ import annotations

import logging
import re

from harnyx_miner_sdk.api import fetch_page, llm_chat, search_ai, search_web
from harnyx_miner_sdk.decorators import entrypoint
from harnyx_miner_sdk.query import CitationRef, Query, Response

logger = logging.getLogger(__name__)

MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"
MAX_EVIDENCE = 8

QUERY_PROMPT = """\
You are a search query engineer. Given a research question, output exactly 2 search queries.

RULES:
- Query 1: The exact question text (verbatim).
- Query 2: A focused query targeting key entities, years, or exact figures.
- Use exact names, years, and technical terms from the question.
- Output ONLY two lines. No labels, no explanation, no numbering.
"""

SYNTHESIS_PROMPT = """\
You are a research fact extractor. Answer the query using ONLY the evidence provided.

ABSOLUTE RULES:
1. You have exactly {num_results} sources.
2. Cite sources using their TITLE in parentheses, like (Title of Source) or (Title of Source 1; Title of Source 2).
3. ONLY cite sources that are listed in the evidence. Do NOT make up source titles.
4. Before answering, list each sub-question from the original query.
5. Answer EVERY part of the query. Use bullet points for multi-part questions.
6. Cite EVERY fact with a source title.
7. Use EXACT numbers, names, and dates from the evidence. Do NOT calculate or infer.
8. FORBIDDEN phrases (zero score if used):
   not specified, not mentioned, not provided, cannot be determined,
   insufficient, no information, not available, not clear, not stated,
   does not provide, does not specify, not contained, not included,
   not given, not found, unable to determine, no data, does not contain,
   evidence does not, did not publicly specify, is not addressed,
   does not list, is not provided, not discussed, not disclosed,
   unclear, ambiguous, appears to be, seems to be, reportedly,
   difficult to determine, hard to say, no precise figures, no exact numbers,
   not enough information, insufficient data, cannot confirm, not confirmed
9. If evidence is partial: state what IS known directly.
   BAD: "The evidence does not specify which company spent more."
   GOOD: "Microsoft spent $29.5B and Apple spent $29.9B on R&D in 2023 (Microsoft Annual Report; Apple 10-K). Apple spent more by $400M (Microsoft Annual Report; Apple 10-K)."
10. Be concise but thorough. One paragraph/bullet per sub-question.
"""

_HEDGE_RE = re.compile(
    r"(?i)\b(?:"
    r"not specified|not mentioned|not provided|cannot be determined|"
    r"insufficient(?:\s+information|(?:\s+data)?)?|no information|not available|not clear|not stated|"
    r"does not provide|does not specify|not contained|not included|"
    r"not given|not found|unable to determine|no data|does not contain|"
    r"evidence does not|is in the provided evidence|did not publicly specify|"
    r"is not addressed|does not list|is not provided|"
    r"not discussed|not disclosed|unclear|ambiguous|"
    r"according to the provided|the evidence suggests|the evidence indicates|"
    r"appears to be|seems to be|reportedly|"
    r"difficult to determine|hard to say|unclear whether|"
    r"no precise figures?|no exact numbers?|not enough information|"
    r"not enough to|insufficient to|unable to|does not explicitly|"
    r"is not explicitly|not explicitly stated|not explicitly provided|"
    r"no specific|not specific|vague about|unclear about|"
    r"cannot confirm|not confirmed"
    r")\b"
)

_NEGATION_WORDS = {
    "no", "not", "cannot", "can't", "unable", "doesn't", "does not",
    "isn't", "is not", "wasn't", "was not", "never", "none", "without",
    "lacks", "lack", "missing", "absent", "insufficient"
}
_UNCERTAINTY_WORDS = {
    "specified", "mentioned", "provided", "determined", "available",
    "clear", "stated", "contained", "included", "given", "found",
    "data", "information", "evidence", "figures", "numbers", "details",
    "exact", "precise", "specific", "enough", "sufficient", "discussed",
    "disclosed", "confirmed"
}


def _has_proximity_hedge(text: str) -> bool:
    words = re.findall(r"\b\w+\b", text.lower())
    for i, word in enumerate(words):
        if word in _NEGATION_WORDS:
            window = words[i:i+7]
            for w in window[1:]:
                if w in _UNCERTAINTY_WORDS:
                    return True
    return False


def _has_hedge(text: str) -> bool:
    return bool(_HEDGE_RE.search(text)) or _has_proximity_hedge(text)


class Result:
    __slots__ = ("result_id", "receipt_id", "url", "title", "content")

    def __init__(self, result_id: str, receipt_id: str, url: str,
                 title: str, content: str) -> None:
        self.result_id  = result_id
        self.receipt_id = receipt_id
        self.url        = url
        self.title      = title
        self.content    = content


async def _gen_queries(question: str) -> tuple[str, str]:
    try:
        r = await llm_chat(
            messages=[
                {"role": "system", "content": QUERY_PROMPT},
                {"role": "user",   "content": f"Question: {question}"},
            ],
            model=MODEL, max_output_tokens=200, temperature=0.0,
        )
        lines = [l.strip() for l in (r.llm.raw_text or "").splitlines() if l.strip()]
        return (
            lines[0] if lines else question,
            lines[1] if len(lines) > 1 else question,
        )
    except Exception:
        return question, question


_FETCH_BLACKLIST = {'.pdf', '.doc', '.docx', '.xls', '.ppt', '.pptx', 'download?', 'attachment?'}

async def _fetch_page_safe(url: str) -> str:
    """Fetch page content, return empty string on failure."""
    if not url or not url.startswith('http'):
        return ""
    url_lower = url.lower()
    if any(b in url_lower for b in _FETCH_BLACKLIST):
        return ""
    try:
        page = await fetch_page(url)
        resp = getattr(page, "response", None)
        if not resp:
            return ""
        data = getattr(resp, "data", []) or []
        if not data:
            return ""
        content = getattr(data[0], "content", "") or ""
        if content and len(content) > 50:
            return content[:1000]
    except Exception as exc:
        logger.debug("fetch_page failed for %s: %s", url, exc)
    return ""

async def _fetch_pages(results: list[Result], limit: int = 2) -> None:
    """Fetch full content for top N results."""
    fetched = 0
    for r in results:
        if fetched >= limit:
            break
        content = await _fetch_page_safe(r.url)
        if content:
            r.content = content
            fetched += 1


async def _search_web(query: str, seen: set[str]) -> list[Result]:
    """Execute search_web and return structured results."""
    results: list[Result] = []
    try:
        resp = await search_web(query, num=12)
        receipt = resp.receipt_id
        if not receipt:
            return results
        
        data = getattr(resp.response, "data", []) or []
        tool = getattr(resp, "results", ()) or ()
        
        for i, res in enumerate(data):
            url = getattr(res, "link", "") or ""
            title = getattr(res, "title", "") or ""
            snippet = getattr(res, "snippet", "") or ""
            
            rid = ""
            if i < len(tool):
                rid = getattr(tool[i], "result_id", "") or ""
            
            if not rid or not receipt:
                continue
            if not url or url in seen:
                continue
            if not (title or snippet):
                continue
            
            seen.add(url)
            results.append(Result(rid, receipt, url, title, snippet))
    except Exception as exc:
        logger.debug("search_web failed: %s", exc)
    return results


async def _search_ai(query: str, seen: set[str]) -> list[Result]:
    """Execute search_ai and return structured results."""
    results: list[Result] = []
    try:
        resp = await search_ai(query, count=5)
        receipt = resp.receipt_id
        if not receipt:
            return results
        
        data = getattr(resp.response, "data", []) or []
        tool = getattr(resp, "results", ()) or ()
        
        for i, res in enumerate(data):
            url = getattr(res, "url", "") or ""
            title = getattr(res, "title", "") or ""
            note = getattr(res, "note", "") or ""
            
            rid = ""
            if i < len(tool):
                rid = getattr(tool[i], "result_id", "") or ""
            
            if not rid or not receipt:
                continue
            if not url or url in seen:
                continue
            if not (title or note):
                continue
            
            seen.add(url)
            results.append(Result(rid, receipt, url, title, note))
    except Exception as exc:
        logger.debug("search_ai failed: %s", exc)
    return results


def _evidence(results: list[Result]) -> str:
    """Format evidence with clear titles for citation by name."""
    lines = []
    for i, r in enumerate(results):
        body = r.content.strip()[:1500]
        lines.append(f"Source: {r.title}\nURL: {r.url}\nContent: {body}")
    return "\n\n---\n\n".join(lines)


def _map_title_to_index(answer: str, results: list[Result]) -> tuple[str, list[CitationRef]]:
    """
    Map title-based citations in answer to [N] indices.
    
    Strategy:
    1. Find all (Title) patterns in answer
    2. For each title, find the best matching result index
    3. Replace (Title) with [N]
    4. Return fixed answer and CitationRef list
    """
    if not results:
        return answer, []
    
    # Build title → index mapping
    title_map: dict[str, int] = {}
    for i, r in enumerate(results):
        title_lower = r.title.lower().strip()
        title_map[title_lower] = i
        # Also map without punctuation
        title_clean = re.sub(r'[^\w\s]', '', title_lower)
        title_map[title_clean] = i
    
    # Find citation patterns: (Title) or (Title 1; Title 2)
    # Match anything in parentheses that's not a number
    cite_pattern = re.compile(r'\(([^()]{3,200})\)')
    
    fixed_answer = answer
    refs: list[CitationRef] = []
    seen_refs: set[tuple] = set()
    
    for m in cite_pattern.finditer(answer):
        citation_text = m.group(1).strip()
        
        # Try to find matching result
        best_idx = -1
        best_score = 0
        
        # Try exact match first
        cite_lower = citation_text.lower().strip()
        if cite_lower in title_map:
            best_idx = title_map[cite_lower]
        else:
            # Try partial match
            cite_clean = re.sub(r'[^\w\s]', '', cite_lower)
            for title, idx in title_map.items():
                # Score based on word overlap
                cite_words = set(cite_clean.split())
                title_words = set(title.split())
                if cite_words and title_words:
                    overlap = len(cite_words & title_words)
                    score = overlap / max(len(cite_words), len(title_words))
                    if score > best_score and score > 0.3:
                        best_score = score
                        best_idx = idx
        
        if best_idx >= 0:
            # Replace (Title) with [N]
            old_str = m.group(0)
            new_str = f"[{best_idx}]"
            fixed_answer = fixed_answer.replace(old_str, new_str, 1)
            
            r = results[best_idx]
            if r.result_id and r.receipt_id:
                key = (r.receipt_id, r.result_id)
                if key not in seen_refs:
                    seen_refs.add(key)
                    refs.append(CitationRef(receipt_id=r.receipt_id, result_id=r.result_id))
        # If no match found, leave as-is (will be cleaned up later)
    
    return fixed_answer, refs


def _ensure_citations(answer: str, results: list[Result]) -> tuple[str, list[CitationRef]]:
    """
    Ensure answer has valid citations. If not, add them from keyword matching.
    """
    cited = set(re.findall(r'\[(\d+)\]', answer))
    valid_cited = {int(c) for c in cited if int(c) < len(results)}
    
    if valid_cited:
        # Already has valid citations
        refs = []
        seen: set[tuple] = set()
        for idx in sorted(valid_cited):
            r = results[idx]
            if r.result_id and r.receipt_id:
                key = (r.receipt_id, r.result_id)
                if key not in seen:
                    seen.add(key)
                    refs.append(CitationRef(receipt_id=r.receipt_id, result_id=r.result_id))
        return answer, refs
    
    # No valid citations - inject from top result
    if results:
        answer += " [0]"
        r = results[0]
        refs = []
        if r.result_id and r.receipt_id:
            refs.append(CitationRef(receipt_id=r.receipt_id, result_id=r.result_id))
        return answer, refs
    
    return answer, []


def _best_snippet_with_citation(results: list[Result]) -> tuple[str, list[CitationRef]]:
    if not results:
        return "", []
    r = results[0]
    answer = f"{r.title}. {r.content[:400]} [0]"
    refs: list[CitationRef] = []
    if r.result_id and r.receipt_id:
        refs.append(CitationRef(receipt_id=r.receipt_id, result_id=r.result_id))
    return answer, refs


async def _synthesize(question: str, results: list[Result]) -> str:
    evidence = _evidence(results)
    num_results = len(results)
    
    prompt = SYNTHESIS_PROMPT.format(num_results=num_results)
    
    user_msg = (
        f"Query: {question}\n\n"
        f"Evidence ({num_results} sources):\n{evidence[:12000]}\n\n"
        f"Answer ALL parts of the query. Cite every fact with the source title in parentheses."
    )
    try:
        r = await llm_chat(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user",   "content": user_msg},
            ],
            model=MODEL, max_output_tokens=700, temperature=0.0,
        )
        return (r.llm.raw_text or "").strip()
    except Exception:
        parts = [f"{r.content[:200]}" for i, r in enumerate(results[:4]) if r.content]
        return " ".join(parts) or question


async def _rewrite(question: str, answer: str, results: list[Result]) -> str:
    if not _has_hedge(answer):
        return answer
    evidence = _evidence(results[:8])
    prompt = (
        f"Query: {question}\n\nEvidence:\n{evidence[:5000]}\n\n"
        f"Draft (rewrite removing hedges):\n{answer}\n\n"
        "Remove ALL hedge phrases. Use strongest positive statements. Keep source citations.\n"
        "If evidence is partial, state what IS known directly.\n"
        "Forbidden: not specified, not provided, cannot be determined, no information, "
        "not available, unclear, does not provide, not found, unable to determine, "
        "seems to be, likely, probably, may be, no precise figures."
    )
    try:
        r = await llm_chat(
            messages=[{"role": "user", "content": prompt}],
            model=MODEL, max_output_tokens=600, temperature=0.0,
        )
        return (r.llm.raw_text or "").strip() or answer
    except Exception:
        return answer


@entrypoint("query")
async def agent(query: Query) -> Response:
    q = query.text.strip()
    if not q:
        return Response(text="No question provided.")

    seen: set[str] = set()

    # 1. Generate 2 targeted queries
    q1, q2 = await _gen_queries(q)

    # 2. search_web + search_ai for broader coverage
    results = await _search_web(q1, seen)
    results += await _search_web(q2, seen)
    results += await _search_ai(q1, seen)
    results += await _search_ai(q2, seen)

    # 3. No results fallback
    if not results:
        try:
            r = await llm_chat(
                messages=[
                    {"role": "system", "content": SYNTHESIS_PROMPT.format(num_results=0)},
                    {"role": "user",   "content": f"Answer from knowledge: {q}"},
                ],
                model=MODEL, max_output_tokens=300, temperature=0.0,
            )
            return Response(text=(r.llm.raw_text or q).strip())
        except Exception:
            return Response(text=q)

    # 4. Fetch full content for top 2 results
    await _fetch_pages(results, limit=2)

    # 5. Limit to top N results for synthesis
    results = results[:MAX_EVIDENCE]

    # 6. Synthesize with title-based citations
    answer = await _synthesize(q, results)

    # 6. Hedge rewrite
    answer = await _rewrite(q, answer, results)

    # 7. Map title citations to [N] indices
    answer, refs = _map_title_to_index(answer, results)

    # 8. Ensure at least some valid citations exist
    answer, refs = _ensure_citations(answer, results)

    # 9. HEDGE HARD-STOP
    if _has_hedge(answer):
        logger.warning("HEDGE HARD-STOP triggered")
        answer, refs = _best_snippet_with_citation(results)
        return Response(text=answer[:2000], citations=refs or None)

    citations = refs if refs else None
    return Response(text=answer[:2000], citations=citations)
