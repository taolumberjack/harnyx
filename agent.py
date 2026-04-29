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


def _strip_hedge_sentences(text: str) -> str:
    """Remove sentences that are PURE hedge (no facts). Keep mixed sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    kept = []
    for sent in sentences:
        s = sent.strip()
        if not s:
            continue
        # Pure hedge sentence: all text matches hedge words + short
        hedge_count = sum(1 for h in _HEDGE_WORDS if h.lower() in s.lower())
        if hedge_count > 0 and len(s) < 120:
            # Mixed sentence - keep it but remove the hedge clause
            for h in _HEDGE_WORDS:
                s = re.sub(r'(?i)' + re.escape(h) + r'[^.]*', '', s)
            s = s.strip()
            if len(s) > 30:
                kept.append(s)
        else:
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

    # Phase 4: Enrich top 5 thin snippets
    for s in sources[:5]:
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
    sources = sources[:12]

    # Build refs
    refs = [CitationRef(receipt_id=s.receipt_id, result_id=s.result_id) for s in sources]

    # Phase 5: Synthesize with strong prompt
    evidence = _fmt_evidence(sources)
    system = (
        "You are a precise research agent. Answer using ONLY the provided evidence.\n\n"
        "RULES:\n"
        "1. Answer EVERY part of the question comprehensively.\n"
        "2. Cite every claim with [N]. Use multiple citations when supported.\n"
        "3. Include exact numbers, dates, names from evidence.\n"
        "4. NEVER say information is missing or not provided.\n"
        "5. If evidence is partial, state what IS known and make brief inference.\n"
        "6. Output ONLY the answer. No preamble."
    )
    user = (
        f"Question: {q}\n\n"
        f"Evidence ({len(sources)} sources):\n{evidence}\n\n"
        "Write a thorough, specific answer. Cover all parts. Cite every claim."
    )

    answer = ""
    try:
        r = await llm_chat(
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            model=MODEL,
            temperature=0.0,
            max_output_tokens=600,
        )
        answer = _extract_text(r)
    except Exception as e:
        logger.debug(f"llm_chat failed: {e}")

    # Phase 6: Strip pure hedge sentences, keep mixed ones
    answer = _strip_hedge_sentences(answer)

    # Build citations
    used_refs = [refs[i] for i in range(len(sources)) if f"[{i+1}]" in answer]

    # Fallback: return top evidence if answer is empty
    if not answer.strip():
        parts = []
        for i, s in enumerate(sources[:5], 1):
            snip = (s.snippet or s.title or "").strip()[:200]
            parts.append(f"[{i}] {snip}")
        answer = " ".join(parts)
        used_refs = refs[:5]

    return Response(text=answer[:2500], citations=used_refs or None)
