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
    "the evidence does not", "it is not possible", "no evidence",
}


class Source:
    __slots__ = ("url", "title", "snippet", "receipt_id", "result_id", "score", "full_text")
    def __init__(self, url, title, snippet, receipt_id="", result_id=""):
        self.url = url
        self.title = title or ""
        self.snippet = snippet or ""
        self.receipt_id = receipt_id
        self.result_id = result_id
        self.score = 0.0
        self.full_text = ""


def _extract_text(ans):
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


def _tokens(text):
    return set(re.findall(r"\w+", text.lower()))


def _score(src, qt):
    text = f"{src.title} {src.snippet} {src.full_text}".lower()
    hits = sum(1 for t in qt if t in text)
    return hits + min(len(text) / 400.0, 1.5)


def _strip_hedges(text):
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

    sources = []
    seen = set()

    # Phase 1: Web search (15 results)
    try:
        r = await search_web(q, num=15)
        receipt = getattr(r, "receipt_id", "")
        for res in r.results[:15]:
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

    # Phase 4: Fetch pages for top 8
    for s in sources[:8]:
        if s.url.startswith("http"):
            try:
                page = await fetch_page(s.url)
                content = getattr(page, "content", None) or getattr(page, "text", None) or ""
                s.full_text = content[:3000].strip()
                s.score = _score(s, qt)
            except Exception:
                pass

    sources.sort(key=lambda s: -s.score)
    sources = sources[:12]

    # Build refs
    refs = [CitationRef(receipt_id=s.receipt_id, result_id=s.result_id) for s in sources]

    # Phase 5: Two-step fact extraction
    # Step 5a: Extract raw facts from each source
    evidence_parts = []
    for i, s in enumerate(sources, 1):
        text = s.full_text or s.snippet or s.title or ""
        text = text[:800].strip()
        evidence_parts.append(f"[{i}] {s.title or 'Source'}\n{text}")

    evidence = "\n\n".join(evidence_parts)

    extract_prompt = (
        f"Question: {q}\n\n"
        f"Evidence from {len(sources)} sources:\n{evidence}\n\n"
        "STEP 1 - EXTRACT FACTS:\n"
        "List every specific fact from the evidence that answers the question. "
        "Format each fact as a bullet with the source number [N]. "
        "Include exact numbers, dates, names, and percentages. "
        "Do NOT include facts that are NOT in the evidence. "
        "Do NOT write 'not specified' or 'not provided'.\n\n"
        "STEP 2 - ANSWER:\n"
        "Write a concise paragraph answering the question using ONLY the extracted facts. "
        "Cite every claim with [N]. Never mention missing information."
    )

    try:
        r = await llm_chat(
            messages=[{"role": "user", "content": extract_prompt}],
            model=MODEL,
            temperature=0.0,
            max_output_tokens=800,
        )
        answer = _extract_text(r)
    except Exception:
        answer = ""

    # Guard: strip hedge sentences
    answer = _strip_hedges(answer)

    # Build used citations
    used_refs = [refs[i] for i in range(len(sources)) if f"[{i+1}]" in answer]

    # Fallback
    if not answer.strip():
        parts = []
        for i, s in enumerate(sources[:5], 1):
            snip = (s.full_text or s.snippet or s.title or "").strip()[:200]
            parts.append(f"[{i}] {snip}")
        answer = " ".join(parts)
        used_refs = refs[:5]

    return Response(text=answer[:2500], citations=used_refs or None)
