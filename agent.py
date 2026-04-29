from __future__ import annotations

import asyncio
import logging

from harnyx_miner_sdk.api import llm_chat, search_web, search_ai
from harnyx_miner_sdk.decorators import entrypoint
from harnyx_miner_sdk.query import CitationRef, Query, Response

logger = logging.getLogger(__name__)

MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"

SYSTEM = (
    "You extract facts from evidence. State what IS known. "
    "Cite with [1], [2], etc. Never say what is missing. "
    "Never use phrases like 'not specified', 'cannot be determined', "
    "'does not mention', 'insufficient information', 'not found', "
    "'not provided', 'unable to', 'unknown', 'lacks', 'absent', "
    "'missing', 'not listed'. Be concise."
)


@entrypoint("query")
async def agent(query: Query) -> Response:
    q = query.text.strip()
    if not q:
        return Response(text="No question provided.")

    lines, refs = await _search(q)
    if not lines:
        return Response(text=q)

    evidence = "\n\n".join(lines)
    answer = await _extract(q, evidence)

    if not answer.strip():
        # LLM failed — return first evidence item as fallback
        answer = lines[0] if lines else q
        used_refs = [refs[0]] if refs else None
        return Response(text=answer, citations=used_refs)

    used_refs = [ref for i, ref in enumerate(refs, 1) if f"[{i}]" in answer]
    return Response(text=answer, citations=used_refs or None)


async def _search(q: str) -> tuple[list[str], list[CitationRef]]:
    all_lines, all_refs = [], []
    seen = set()
    counter = 1

    async def do_web():
        nonlocal counter
        try:
            r = await search_web(q, num=5)
            lines, refs = [], []
            for res in r.results[:5]:
                url = getattr(res, "url", "") or getattr(res, "link", "")
                if url in seen:
                    continue
                seen.add(url)
                body = getattr(res, "note", "") or getattr(res, "snippet", "") or ""
                title = getattr(res, "title", "") or ""
                if body.strip():
                    lines.append(f"[{counter}] {title}: {body}")
                    refs.append(CitationRef(receipt_id=r.receipt_id, result_id=res.result_id))
                    counter += 1
            return lines, refs
        except Exception as e:
            logger.debug(f"search_web failed: {e}")
            return [], []

    async def do_ai():
        nonlocal counter
        try:
            r = await search_ai(prompt=q, count=10)
            if not r or not r.response or not r.response.data:
                return [], []
            lines, refs = [], []
            for res in r.response.data[:5]:
                url = getattr(res, "url", "")
                if url in seen:
                    continue
                seen.add(url)
                note = getattr(res, "note", "") or ""
                title = getattr(res, "title", "") or ""
                if note.strip():
                    lines.append(f"[{counter}] {title}: {note}")
                    refs.append(CitationRef(receipt_id=r.receipt_id, result_id=url or str(counter)))
                    counter += 1
            return lines, refs
        except Exception as e:
            logger.debug(f"search_ai failed: {e}")
            return [], []

    (w1, r1), (w2, r2) = await asyncio.gather(do_web(), do_ai())
    return w1 + w2, r1 + r2


async def _extract(q: str, evidence: str) -> str:
    msg = (
        f"Question: {q}\n\n"
        f"Evidence:\n{evidence}\n\n"
        "Answer using only the evidence. Cite [1], [2], etc. "
        "State what IS known. Never mention what is missing."
    )
    try:
        r = await llm_chat(
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": msg}],
            model=MODEL, temperature=0.0, max_output_tokens=400,
        )
        return _extract_text(r)
    except Exception:
        return ""


def _extract_text(ans) -> str:
    if ans.llm and ans.llm.choices:
        content = ans.llm.choices[0].message.content
        if isinstance(content, str):
            return content.strip()
        elif isinstance(content, list):
            return "".join(p.text for p in content if getattr(p, "text", None)).strip()
    return ""
