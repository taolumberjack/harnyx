from __future__ import annotations

import logging

from harnyx_miner_sdk.api import llm_chat, search_web
from harnyx_miner_sdk.decorators import entrypoint
from harnyx_miner_sdk.query import CitationRef, Query, Response

logger = logging.getLogger(__name__)

MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"


class Source:
    __slots__ = ("url", "title", "snippet", "receipt_id", "result_id")
    def __init__(self, url: str, title: str | None, snippet: str | None,
                 receipt_id: str = "", result_id: str = "") -> None:
        self.url = url
        self.title = title or ""
        self.snippet = snippet or ""
        self.receipt_id = receipt_id
        self.result_id = result_id


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


@entrypoint("query")
async def agent(query: Query) -> Response:
    q = query.text.strip()
    if not q:
        return Response(text="No question provided.")

    sources: list[Source] = []
    seen: set[str] = set()

    # Phase 1: Search (4 results, like champion)
    try:
        r = await search_web(q, num=4)
        receipt = getattr(r, "receipt_id", "")
        for res in r.results[:4]:
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

    if not sources:
        return Response(text=q)

    # Build refs
    refs = [CitationRef(receipt_id=s.receipt_id, result_id=s.result_id) for s in sources]

    # Phase 2: Single LLM synthesis (like champion)
    evidence_lines = []
    for i, s in enumerate(sources, 1):
        snippet = (s.snippet or s.title or "").strip()[:1500]
        evidence_lines.append(f"[{i}] {s.title or 'Source'}: {snippet}")

    evidence = "\n".join(evidence_lines)

    system_prompt = (
        "You are writing the definitive reference answer to the given question.\n\n"
        "CRITICAL: Respond in the EXACT same language as the question — never switch languages.\n\n"
        "Search evidence is provided below. Use it to:\n"
        "- Back any time-sensitive or current-status claims with the evidence\n"
        "- Cover all key aspects: causes, effects, facts, names, dates, numbers\n"
        "- Lead with a direct answer, then cover each major sub-topic concisely\n"
        "- Stable well-known facts do not need evidence\n"
        "- No padding, no filler, no opinions\n\n"
        "Cite every claim with [N] matching the evidence number."
    )

    user_prompt = (
        f"Evidence:\n{evidence}\n\n"
        f"Question: {q}"
    )

    try:
        r = await llm_chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=MODEL,
            temperature=0.0,
            max_output_tokens=800,
        )
        answer = _extract_text(r)
    except Exception as e:
        logger.debug(f"llm_chat failed: {e}")
        answer = ""

    # Build used citations
    used_refs = [refs[i] for i in range(len(sources)) if f"[{i+1}]" in answer]

    # Fallback
    if not answer.strip():
        parts = []
        for i, s in enumerate(sources, 1):
            snip = (s.snippet or s.title or "").strip()[:200]
            parts.append(f"[{i}] {snip}")
        answer = " ".join(parts)
        used_refs = refs

    return Response(text=answer[:2500], citations=used_refs or None)
