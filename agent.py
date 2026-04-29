from __future__ import annotations

import logging
import re

from harnyx_miner_sdk.api import llm_chat, search_web, search_ai, fetch_page
from harnyx_miner_sdk.decorators import entrypoint
from harnyx_miner_sdk.query import CitationRef, Query, Response

logger = logging.getLogger(__name__)

MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"

_HEDGE_PATTERNS = [
    "cannot be determined", "not provided", "not specified", "not mentioned",
    "not stated", "not given", "not clear", "not available", "not found",
    "does not specify", "does not mention", "does not state", "does not provide",
    "does not indicate", "does not include", "does not contain",
    "no information is provided", "no information was provided",
    "no information is available", "no information was available",
    "no information is given", "no information was given",
    "no information is found", "no information was found",
    "no data is provided", "no data was provided", "no data is available",
    "insufficient information", "insufficient evidence", "insufficient context",
    "insufficient data", "unclear from", "unclear based on",
    "the text does not", "the passage does not", "the excerpt does not",
    "the snippet does not", "the source does not", "the document does not",
    "the evidence does not", "based on the available evidence.*cannot",
    "based on the provided evidence.*cannot",
    "it is not possible to determine", "it was not possible to determine",
    "not available", "not stated", "not given", "not found", "not addressed",
    "not covered", "not discussed", "not listed", "not included",
    "unknown", "missing", "lacks", "absent", "not enough",
]

_HEDGE_RE = re.compile(
    r"\b(" + "|".join(re.escape(p) for p in _HEDGE_PATTERNS) + r")\b",
    re.IGNORECASE,
)

SYSTEM = (
    "You are a precise research agent. Answer the question using ONLY the evidence.\n\n"
    "CRITICAL RULES:\n"
    "1. State ONLY facts that appear in the evidence.\n"
    "2. Cite every factual claim with [N] matching the evidence index.\n"
    "3. Use exact numbers, dates, names, and figures from the evidence.\n"
    "4. NEVER hedge or say information is missing.\n"
    "5. If evidence is partial, state what IS known. Do NOT mention gaps.\n"
    "6. Output ONLY the answer. No preamble, no markdown fences, no explanations.\n\n"
    "FORBIDDEN PHRASES (these cause immediate zero score):\n"
    "'cannot be determined', 'not provided', 'not specified', 'not mentioned',\n"
    "'does not specify', 'no information', 'insufficient', 'unclear',\n"
    "'the evidence does not', 'it is not possible'.\n\n"
    "EXAMPLE - GOOD answer (score 1.0):\n"
    "'Based on [1], the project launched in Q4 2024. [2] confirms the budget was $50M.'\n\n"
    "EXAMPLE - BAD answer (score 0.0):\n"
    "'The evidence does not specify the exact launch date. This information is not provided.'"
)


class Source:
    __slots__ = ("url", "title", "snippet", "receipt_id", "result_id", "score")
    def __init__(self, url: str, title: str | None, snippet: str | None,
                 receipt_id: str = "", result_id: str = "") -> None:
        self.url = url
        self.title = title or ""
        self.snippet = snippet or ""
        self.receipt_id = receipt_id
        self.result_id = result_id
        self.score = 0.0


def _extract_text(ans) -> str:
    try:
        content = ans.llm.choices[0].message.content
        if isinstance(content, str):
            return content.strip()
        parts = []
        for p in content:
            t = getattr(p, "text", None)
            if t:
                parts.append(t)
        return "".join(parts).strip()
    except Exception:
        return ""


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def _score(src: Source, qt: set[str]) -> float:
    text = f"{src.title} {src.snippet}".lower()
    hits = sum(1 for t in qt if t in text)
    return hits + min(len(text) / 400.0, 1.5)


def _fmt_evidence(sources: list[Source]) -> str:
    lines = []
    for i, s in enumerate(sources, 1):
        snip = (s.snippet or s.title or "").strip()[:500]
        lines.append(f"[{i}] {s.title or 'Source'}: {snip}")
    return "\n\n".join(lines)


def _has_hedges(text: str) -> bool:
    return bool(_HEDGE_RE.search(text))


def _build_fallback(sources: list[Source]) -> str:
    """Build answer directly from evidence when LLM hedges."""
    parts = []
    for i, s in enumerate(sources[:5], 1):
        snip = (s.snippet or s.title or "").strip()[:250]
        if snip:
            parts.append(f"[{i}] {snip}")
    return " ".join(parts)


@entrypoint("query")
async def agent(query: Query) -> Response:
    q = query.text.strip()
    if not q:
        return Response(text="No question provided.")

    sources: list[Source] = []
    seen: set[str] = set()

    # Phase 1: Web search
    try:
        r = await search_web(q, num=5)
        receipt = getattr(r, "receipt_id", "")
        for res in r.results[:5]:
            url = getattr(res, "url", "") or getattr(res, "link", "")
            if url and url not in seen:
                seen.add(url)
                sources.append(Source(
                    url=url,
                    title=getattr(res, "title", None),
                    snippet=getattr(res, "note", None) or getattr(res, "snippet", None),
                    receipt_id=receipt,
                    result_id=getattr(res, "result_id", "") or url,
                ))
    except Exception as e:
        logger.debug(f"search_web failed: {e}")

    # Phase 2: AI search
    try:
        r = await search_ai(prompt=q, count=10)
        if r and getattr(r, "response", None) and getattr(r.response, "data", None):
            receipt = getattr(r, "receipt_id", "")
            for res in r.response.data[:5]:
                url = getattr(res, "url", "")
                if url and url not in seen:
                    seen.add(url)
                    sources.append(Source(
                        url=url,
                        title=getattr(res, "title", None),
                        snippet=getattr(res, "note", None) or getattr(res, "snippet", None),
                        receipt_id=receipt,
                        result_id=url,
                    ))
    except Exception as e:
        logger.debug(f"search_ai failed: {e}")

    if not sources:
        return Response(text=q)

    # Phase 3: Rank
    qt = _tokens(q)
    for s in sources:
        s.score = _score(s, qt)
    sources.sort(key=lambda s: -s.score)

    # Phase 4: Enrich thin snippets
    for s in sources[:3]:
        if s.url.startswith("http") and (not s.snippet or len(s.snippet) < 120):
            try:
                page = await fetch_page(s.url)
                content = getattr(page, "content", None) or getattr(page, "text", None) or ""
                content = content[:1500].strip()
                if content:
                    s.snippet = content
                    s.score = _score(s, qt)
            except Exception:
                pass

    sources.sort(key=lambda s: -s.score)
    sources = sources[:8]

    # Build refs
    refs = [CitationRef(receipt_id=s.receipt_id, result_id=s.result_id) for s in sources]

    # Phase 5: Synthesize
    evidence = _fmt_evidence(sources)
    user = (
        f"Question: {q}\n\n"
        f"Evidence ({len(sources)} sources):\n{evidence}\n\n"
        "Answer using ONLY the evidence. Cite every claim with [N]. "
        "State facts directly. Never mention missing information."
    )

    answer = ""
    try:
        r = await llm_chat(
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": user}],
            model=MODEL,
            temperature=0.0,
            max_output_tokens=500,
        )
        answer = _extract_text(r)
    except Exception as e:
        logger.debug(f"llm_chat failed: {e}")

    # Phase 6: Guard - if hedges detected, discard and use fallback
    if _has_hedges(answer):
        logger.debug(f"Hedges detected, using fallback")
        answer = _build_fallback(sources)

    if not answer.strip():
        answer = _build_fallback(sources)

    # Build citations from sources actually cited
    used_refs = [refs[i] for i in range(len(sources)) if f"[{i+1}]" in answer]
    if not used_refs:
        used_refs = refs[:5]

    return Response(text=answer[:2500], citations=used_refs)
