from __future__ import annotations

import logging
import re

from harnyx_miner_sdk.api import llm_chat, search_web, search_ai, fetch_page
from harnyx_miner_sdk.decorators import entrypoint
from harnyx_miner_sdk.query import CitationRef, Query, Response

logger = logging.getLogger(__name__)

MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"

_HEDGE_WORDS = {
    "cannot be determined", "not provided", "not specified", "not mentioned",
    "not stated", "not given", "not clear", "not available", "not found",
    "does not specify", "does not mention", "does not state", "does not provide",
    "does not indicate", "does not include", "does not contain",
    "no information", "insufficient", "unclear", "the text does not",
    "the evidence does not", "it is not possible",
}


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
        snip = (s.snippet or s.title or "").strip()[:600]
        lines.append(f"[{i}] {s.title or 'Source'}: {snip}")
    return "\n\n".join(lines)


def _strip_hedges(text: str) -> str:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    kept = []
    for sent in sentences:
        s = sent.strip()
        if not s:
            continue
        if any(h in s.lower() for h in _HEDGE_WORDS) and len(s) < 150:
            continue
        kept.append(s)
    result = " ".join(kept).strip()
    return result if result else text


@entrypoint("query")
async def agent(query: Query) -> Response:
    q = query.text.strip()
    if not q:
        return Response(text="No question provided.")

    sources: list[Source] = []
    seen: set[str] = set()

    # Phase 1: Web search (10 results)
    try:
        r = await search_web(q, num=10)
        receipt = getattr(r, "receipt_id", "")
        for res in r.results[:10]:
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

    # Phase 2: AI search (10 results)
    try:
        r = await search_ai(prompt=q, count=10)
        if r and getattr(r, "response", None) and getattr(r.response, "data", None):
            receipt = getattr(r, "receipt_id", "")
            for res in r.response.data[:10]:
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

    # Phase 4: Fetch pages for top 5 sources
    for s in sources[:5]:
        if s.url.startswith("http"):
            try:
                page = await fetch_page(s.url)
                content = getattr(page, "content", None) or getattr(page, "text", None) or ""
                content = content[:2000].strip()
                if content:
                    s.snippet = content
                    s.score = _score(s, qt)
            except Exception:
                pass

    sources.sort(key=lambda s: -s.score)
    sources = sources[:10]

    # Build refs
    refs = [CitationRef(receipt_id=s.receipt_id, result_id=s.result_id) for s in sources]

    # Phase 5: Extract facts with LLM
    evidence = _fmt_evidence(sources)
    extract_prompt = (
        f"Question: {q}\n\n"
        f"Evidence ({len(sources)} sources):\n{evidence}\n\n"
        "Extract EVERY specific fact from the evidence that answers the question. "
        "Format as bullet points. Each bullet must include [N] citation. "
        "Include exact numbers, dates, names, and percentages. "
        "Never write 'not specified' or 'not provided'. "
        "Only include facts that ARE in the evidence."
    )

    facts = ""
    try:
        r = await llm_chat(
            messages=[{"role": "user", "content": extract_prompt}],
            model=MODEL,
            temperature=0.0,
            max_output_tokens=600,
        )
        facts = _extract_text(r)
    except Exception as e:
        logger.debug(f"fact extraction failed: {e}")

    # Phase 6: Synthesize from facts
    if facts and len(facts) > 50:
        synth_prompt = (
            f"Question: {q}\n\n"
            f"Extracted facts:\n{facts}\n\n"
            "Write a direct, comprehensive answer using ONLY these facts. "
            "Cite every claim with [N]. Include all specific numbers and dates. "
            "Never mention missing information."
        )
        try:
            r = await llm_chat(
                messages=[{"role": "user", "content": synth_prompt}],
                model=MODEL,
                temperature=0.0,
                max_output_tokens=600,
            )
            answer = _extract_text(r)
        except Exception:
            answer = facts
    else:
        answer = facts

    # Phase 7: Guard
    answer = _strip_hedges(answer)

    # Build citations
    used_refs = [refs[i] for i in range(len(sources)) if f"[{i+1}]" in answer]

    # Fallback
    if not answer.strip():
        parts = []
        for i, s in enumerate(sources[:5], 1):
            snip = (s.snippet or s.title or "").strip()[:200]
            parts.append(f"[{i}] {snip}")
        answer = " ".join(parts)
        used_refs = refs[:5]

    return Response(text=answer[:2500], citations=used_refs or None)
