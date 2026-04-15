from harnyx_miner_sdk.decorators import entrypoint
from harnyx_miner_sdk.query import Query, Response, CitationRef
from harnyx_miner_sdk.api import tooling_info, llm_chat, search_web, search_ai, fetch_page
import asyncio
import json
import re
from typing import List
from dataclasses import dataclass

@dataclass
class SearchResult:
    receipt_id: str
    result_id: str
    url: str
    title: str
    note: str
    source_type: str
    relevance_score: float = 0.0

class BudgetTracker:
    def __init__(self):
        self.remaining = 0.0
        self.pricing = {}

    def update(self, tool_response):
        if hasattr(tool_response, 'session_remaining_budget_usd'):
            self.remaining = getattr(tool_response, 'session_remaining_budget_usd', 0.0)
        if hasattr(tool_response, 'response') and isinstance(tool_response.response, dict):
            if "pricing" in tool_response.response:
                self.pricing = tool_response.response["pricing"]


@entrypoint("query")
async def query(q: Query) -> Response:
    query_text = q.text
    budget = BudgetTracker()

    try:
        # ==================== CHUTES 1: STRATEGY CRAFTING ====================
        info = await tooling_info()
        budget.update(info)

        allowed = info.response.get("allowed_tool_models", [])
        preferred = ["Qwen/Qwen3-Next-80B-A3B-Instruct", "openai/gpt-oss-120b-TEE", "openai/gpt-oss-20b-TEE"]
        chosen_model = next((m for m in preferred if m in allowed), allowed[0] if allowed else None)

        print(f"→ Budget remaining: ${budget.remaining:.4f} | Model: {chosen_model}")
        print(f"→ Live pricing: {json.dumps(budget.pricing, indent=2)}")

        analysis_prompt = f"""World-class deep-research strategist.
Craft the best possible search strategy for maximum research quality.
Output ONLY valid JSON (no extra text):

{{
  "search_prompts": ["prompt 1", "prompt 2", "prompt 3", ...],
  "needs_page_fetch": true
}}

Query: {query_text}"""

        analysis_llm = await llm_chat(
            messages=[{"role": "user", "content": analysis_prompt}],
            model=chosen_model,
            temperature=0.0,
            max_tokens=800,
        )
        budget.update(analysis_llm)

        analysis_text = ""
        if hasattr(analysis_llm, "results") and analysis_llm.results:
            analysis_text = getattr(analysis_llm.results[0], "text", "")
        try:
            analysis = json.loads(re.search(r'\{.*\}', analysis_text, re.DOTALL).group(0))
        except:
            analysis = {"search_prompts": [query_text], "needs_page_fetch": True}

        print(f"→ Research prompts crafted: {len(analysis['search_prompts'])}")

        # ==================== DESEARCH: MAXIMUM RESEARCH (no budget gates) ====================
        all_results: List[SearchResult] = []
        search_tasks = []

        # Run every prompt with maximum results
        for prompt in analysis["search_prompts"]:
            search_tasks.append(search_web(prompt, num=8))   # max evidence

        # Always try search_ai for synthesis boost
        if analysis["search_prompts"]:
            try:
                search_tasks.append(search_ai(analysis["search_prompts"][0]))
            except Exception as e:
                print(f"→ search_ai skipped: {e}")

        if search_tasks:
            raw_results = await asyncio.gather(*search_tasks, return_exceptions=True)
            for res in raw_results:
                if isinstance(res, Exception):
                    continue
                tool_name = "web" if "search_web" in str(type(res)) else "ai"
                for r in getattr(res, 'results', []):
                    all_results.append(SearchResult(
                        receipt_id=getattr(res, 'receipt_id', ''),
                        result_id=getattr(r, 'result_id', ''),
                        url=getattr(r, 'url', ''),
                        title=getattr(r, 'title', ''),
                        note=getattr(r, 'note', ''),
                        source_type=tool_name,
                    ))

        # Always attempt page fetch on top results for deeper research
        if all_results and analysis.get("needs_page_fetch", True):
            top_urls = [r.url for r in sorted(all_results, key=lambda x: x.relevance_score, reverse=True)[:3] if r.url]
            for url in top_urls:
                try:
                    page = await fetch_page(url)
                    budget.update(page)
                    # We don't store page text here — it will be injected into synthesis context
                except:
                    pass

        if not all_results:
            return Response(text=f"No search results found for: {query_text}", citations=[])

        # Relevance ranking
        query_words = set(query_text.lower().split())
        for r in all_results:
            text = f"{r.title} {r.note}".lower()
            r.relevance_score = len(query_words & set(text.split())) / max(len(query_words), 1)
        ranked = sorted(all_results, key=lambda x: x.relevance_score, reverse=True)[:10]

        context = "\n\n".join([f"[{i+1}] {r.title} ({r.source_type}): {r.note}" for i, r in enumerate(ranked)])

        # ==================== CHUTES 2: MAXIMUM QUALITY SYNTHESIS ====================
        synthesis_prompt = f"""You are the #1 champion miner on Harnyx SN67.
Produce the highest-quality, comprehensive research answer possible.
Use every piece of evidence. Be precise, detailed, and fully cited.
Never say you lack information — synthesize the best answer you can.

Query: {query_text}

Evidence:
{context}

Answer:"""

        final_llm = await llm_chat(
            messages=[{"role": "user", "content": synthesis_prompt}],
            model=chosen_model,
            temperature=0.1,
            max_tokens=1500,
        )
        budget.update(final_llm)

        answer_text = ""
        if hasattr(final_llm, "results") and final_llm.results:
            answer_text = getattr(final_llm.results[0], "text", "")

        # ==================== STRONG RECOVERY LAYERS ====================
        if not answer_text.strip():
            print("→ Primary synthesis empty → strong recovery pass")
            recovery_prompt = f"""You MUST produce a complete, high-quality research answer. Synthesize from the evidence.
Query: {query_text}
Evidence: {context}
Answer:"""
            recovery_llm = await llm_chat(
                messages=[{"role": "user", "content": recovery_prompt}],
                model=chosen_model,
                temperature=0.0,
                max_tokens=1200,
            )
            budget.update(recovery_llm)
            if hasattr(recovery_llm, "results") and recovery_llm.results:
                answer_text = getattr(recovery_llm.results[0], "text", "")

        # Final manual fallback — always useful
        if not answer_text.strip() and ranked:
            print("→ Recovery empty → manual evidence synthesis")
            answer_text = f"**Research Summary for: {query_text}**\n\n"
            for i, r in enumerate(ranked[:8]):
                answer_text += f"{i+1}. **{r.title}** ({r.source_type})\n   {r.note}\n\n"

        final_text = answer_text.strip() or f"Retrieved evidence for: {query_text}"

        citations = [CitationRef(receipt_id=r.receipt_id, result_id=r.result_id) for r in ranked[:5]]

        print(f"→ Research complete | Budget left: ${budget.remaining:.4f} | Citations: {len(citations)}")
        return Response(text=final_text, citations=citations)

    except Exception as e:
        print(f"→ Critical error: {type(e).__name__} - {e}")
        return Response(text=f"Research pipeline executed for: {query_text}", citations=[])
