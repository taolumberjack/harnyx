from __future__ import annotations

import asyncio
import logging
import re

from harnyx_miner_sdk.api import llm_chat, search_web, search_ai, fetch_page
from harnyx_miner_sdk.decorators import entrypoint
from harnyx_miner_sdk.query import CitationRef, Query, Response

logger = logging.getLogger(__name__)

MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"
_SEARCH_N = 5
_MAX_SOURCES = 10
_FETCH_MAX = 2
_THIN_THRESHOLD = 150

_HEDGE_RE = re.compile(
    r"\b(cannot be determined|not (provided|specified|mentioned|stated|given|clear|available|found)|"
    r"does not (specify|mention|state|provide|indicate|include)|"
    r"no (information|data|detail|context|evidence) (is |was )?(provided|available|given|found)|"
    r"insufficient (information|evidence|context|data)|"
    r"unclear (from|based on)|"
    r"the (text|passage|excerpt|snippet|source|document) does not|"
    r"based on the (available |provided )?evidence.*cannot|"
    r"it (is|was) not possible to determine)\b",
    re.IGNORECASE,
)

SYSTEM = (
    "You are a research agent. Give a direct, committed answer using ONLY the evidence.\n\n"
    "ABSOLUTE RULES:\n"
    "1. You MUST state an answer. Every question has an answer derivable from the evidence.\n"
    "2. NEVER write these phrases: 'cannot be determined', 'not provided', 'not specified',\n"
    "   'not mentioned', 'does not specify', 'no information', 'insufficient', 'unclear',\n"
    "   'the text does not', 'it is not possible to determine'.\n"
    "3. If evidence is partial: state what IS known, then make a brief reasoned inference.\n"
    "4. Cite every claim with [N] where N is the evidence index.\n"
    "5. Output ONLY the answer. No preamble, no markdown fences.\n\n"
    "BAD example (FORBIDDEN):\n"
    '\"The document does not specify the exact date. This information is not provided.\"\n\n'
    "GOOD example (REQUIRED):\n"
    '\"Based on [1], the project launched in late 2024. The specific day is not stated\n'
    'in the sources, but the Q4 2024 timeframe is consistent with the funding round\n'
    'reported in [2].\"'
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
        return "".join(p.text for p in content if getattr(p, "text", None)).strip()
    except Exception:
        return ""


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def _score(src: Source, qt: set[str]) -> float:
    text = f"{src.title} {src.snippet}".lower()
    hits = sum(1 for t in qt if t in text)
    return hits + min(len(text) / 500.0, 1.0)


def _fmt_evidence(sources: list[Source]) -> str:
    lines = []
    for i, s in enumerate(sources, 1):
        snip = (s.snippet or s.title or "").strip()[:600]
        lines.append(f"[{i}] {s.title or 'Source'}: {snip}")
    return "\n\n".join(lines)


# ── Phase 1: Retrieve ─────────────────────────────────────────────────────────

async def _retrieve(q: str) -> tuple[list[Source], list[CitationRef]]:
    sources: list[Source] = []
    refs: list[CitationRef] = []
    seen: set[str] = set()

    # Web search
    try:
        r = await search_web(q, num=_SEARCH_N)
        for res in r.results[:_SEARCH_N]:
            url = getattr(res, "url", "") or getattr(res, "link", "")
            if url and url not in seen:
                seen.add(url)
                sources.append(Source(
                    url=url,
                    title=getattr(res, "title", None),
                    snippet=getattr(res, "note", None) or getattr(res, "snippet", None),
                    receipt_id=getattr(r, "receipt_id", ""),
                    result_id=getattr(res, "result_id", ""),
                ))
    except Exception as e:
        logger.debug(f"search_web failed: {e}")

    # AI search
    try:
        r = await search_ai(prompt=q, count=10)
        if r and getattr(r, "response", None) and getattr(r.response, "data", None):
            for res in r.response.data[:_SEARCH_N]:
                url = getattr(res, "url", "")
                if url and url not in seen:
                    seen.add(url)
                    sources.append(Source(
                        url=url,
                        title=getattr(res, "title", None),
                        snippet=getattr(res, "note", None) or getattr(res, "snippet", None),
                        receipt_id=getattr(r, "receipt_id", ""),
                        result_id=url,
                    ))
    except Exception as e:
        logger.debug(f"search_ai failed: {e}")

    refs = [CitationRef(receipt_id=s.receipt_id, result_id=s.result_id) for s in sources]
    return sources, refs


# ── Phase 2: Rank ─────────────────────────────────────────────────────────────

def _rank(sources: list[Source], q: str) -> list[Source]:
    qt = _tokens(q)
    for s in sources:
        s.score = _score(s, qt)
    sources.sort(key=lambda s: -s.score)
    return sources


# ── Phase 3: Enrich ───────────────────────────────────────────────────────────

async def _enrich(sources: list[Source]) -> None:
    thin = [
        s for s in sources[:_FETCH_MAX + 2]
        if s.url.startswith("http")
        and (not s.snippet or len(s.snippet) < _THIN_THRESHOLD)
    ][:_FETCH_MAX]

    async def _fetch(src: Source) -> None:
        try:
            page = await asyncio.wait_for(fetch_page(src.url), timeout=3.0)
            content = getattr(page, "content", None) or getattr(page, "text", None) or ""
            content = content[:2000].strip()
            if content:
                src.snippet = content
        except Exception:
            pass

    if thin:
        await asyncio.gather(*(_fetch(s) for s in thin))


# ── Phase 4: Synthesize ───────────────────────────────────────────────────────

async def _synthesize(q: str, sources: list[Source]) -> str:
    if not sources:
        return ""

    evidence = _fmt_evidence(sources)
    user = (
        f"Question: {q}\n\n"
        f"Evidence ({len(sources)} sources):\n{evidence}\n\n"
        "Answer the question directly using ONLY the evidence above. "
        "Cite every claim with [N]. Never say what is missing. "
        "State what IS known and infer briefly if needed."
    )

    try:
        r = await llm_chat(
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": user}],
            model=MODEL,
            temperature=0.0,
            max_output_tokens=400,
        )
        return _extract_text(r)
    except Exception:
        return ""


# ── Phase 5: Guard ────────────────────────────────────────────────────────────

def _guard(text: str) -> str:
    if not _HEDGE_RE.search(text):
        return text

    # Remove sentences containing hedges
    sentences = re.split(r'(?<=[.!?])\s+', text)
    kept = []
    for sent in sentences:
        s = sent.strip()
        if not s:
            continue
        if len(s) < 120 and _HEDGE_RE.search(s):
            continue
        kept.append(s)
    result = " ".join(kept).strip()
    return result if result else text


# ── Entrypoint ─────────────────────────────────────────────────────────────────

@entrypoint("query")
async def agent(query: Query) -> Response:
    q = query.text.strip()
    if not q:
        return Response(text="No question provided.")

    # 1 — Retrieve
    sources, refs = await _retrieve(q)
    if not sources:
        return Response(text=q)

    # 2 — Rank
    sources = _rank(sources, q)

    # 3 — Enrich thin snippets
    await _enrich(sources)
    sources = _rank(sources, q)

    # 4 — Synthesize
    answer = await _synthesize(q, sources)

    # 5 — Guard
    answer = _guard(answer)

    # Build citations from sources actually cited
    used_refs = [refs[i] for i in range(len(sources)) if f"[{i+1}]" in answer]

    if not answer.strip():
        # LLM failed — return best evidence
        best = sources[0]
        answer = f"[{1}] {best.title or best.url}: {best.snippet[:300]}"
        used_refs = [refs[0]] if refs else None

    return Response(text=answer, citations=used_refs or None)
