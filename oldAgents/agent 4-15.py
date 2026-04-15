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
        self.total = 0.0
        self.remaining = 0.0
        self.pricing = {}

    def update(self, tool_response):
        if hasattr(tool_response, 'session_remaining_budget_usd'):
            self.remaining = getattr(tool_response, 'session_remaining_budget_usd', 0.0)
        if hasattr(tool_response, 'session_budget_usd'):
            self.total = getattr(tool_response, 'session_budget_usd', 0.0)

        if hasattr(tool_response, 'response') and isinstance(tool_response.response, dict):
            if "pricing" in tool_response.response:
                self.pricing = tool_response.response["pricing"]

    def can_afford(self, tool_name: str, model: str | None = None) -> bool:
        if not self.pricing:
            return self.remaining >= 0.06
        if tool_name == "llm_chat" and model:
            cost = self.pricing.get("llm_chat", {}).get(model)
        else:
            cost = self.pricing.get(tool_name)
        if not isinstance(cost, (int, float)):
            cost = 0.06
        return self.remaining >= (cost * 1.2)


@entrypoint("query")
async def query(q: Query) -> Response:
    query_text = q.text
    budget = BudgetTracker()

    try:
        # ==================== CHUTES 1: STRATEGY ====================
        info = await tooling_info()
        budget.update(info)

        allowed = info.response.get("allowed_tool_models", [])
        preferred = ["Qwen/Qwen3-Next-80B-A3B-Instruct", "openai/gpt-oss-120b-TEE", "openai/gpt-oss-20b-TEE"]
        chosen_model = next((m for m in preferred if m in allowed), allowed[0] if allowed else None)

        print(f"→ Budget: ${budget.remaining:.4f} | Model: {chosen_model}")
        print(f"→ Live pricing: {json.dumps(budget.pricing, indent=2)}")

        analysis_prompt = f"""World-class deep-research strategist.
Output ONLY valid JSON (no extra text):

{{
  "complexity": "simple|medium|complex",
  "search_prompts": ["prompt 1", "prompt 2", ...],
  "needs_page_fetch": true|false
}}

Query: {query_text}"""

        analysis_llm = await llm_chat(
            messages=[{"role": "user", "content": analysis_prompt}],
            model=chosen_model,
            temperature=0.0,
            max_tokens=600,
        )
        budget.update(analysis_llm)

        analysis_text = ""
        if hasattr(analysis_llm, "results") and analysis_llm.results:
            analysis_text = getattr(analysis_llm.results[0], "text", "")
        try:
            analysis = json.loads(re.search(r'\{.*\}', analysis_text, re.DOTALL).group(0))
        except:
            analysis = {"complexity": "medium", "search_prompts": [query_text], "needs_page_fetch": False}

        print(f"→ Strategy: {analysis['complexity']} | {len(analysis['search_prompts'])} prompts")

        # ==================== DESEARCH: PARALLEL ====================
        all_results: List[SearchResult] = []
        search_tasks = []

        for prompt in analysis["search_prompts"][:2]:
            if budget.can_afford("search_web"):
                search_tasks.append(search_web(prompt, num=6 if analysis["complexity"] == "complex" else 4))

        if budget.can_afford("search_ai") and analysis["search_prompts"]:
            try:
                search_tasks.append(search_ai(analysis["search_prompts"][0]))
            except Exception as e:
                print(f"→ search_ai skipped safely: {e}")

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

        if not all_results:
            return await _emergency_direct_answer(query_text, chosen_model, budget, ranked=[])

        # Relevance ranking
        query_words = set(query_text.lower().split())
        for r in all_results:
            text = f"{r.title} {r.note}".lower()
            r.relevance_score = len(query_words & set(text.split())) / max(len(query_words), 1)
        ranked = sorted(all_results, key=lambda x: x.relevance_score, reverse=True)[:8]

        # ==================== CHUTES 2: SYNTHESIS + STRONG RECOVERY ====================
        context = "\n\n".join([f"[{i+1}] {r.title} ({r.source_type}): {r.note}" for i, r in enumerate(ranked)])

        if analysis.get("needs_page_fetch") and budget.can_afford("fetch_page") and ranked:
            try:
                page = await fetch_page(ranked[0].url)
                budget.update(page)
                context += f"\n\nDETAILED EXTRACT:\n{getattr(page, 'text', '')[:1500]}"
            except:
                pass

        synthesis_prompt = f"""You are the #1 miner on Harnyx SN67.
Synthesize a precise, fully-cited answer using ONLY the evidence.
Structure: 1) Direct answer 2) Key evidence with citations 3) Caveats.

Query: {query_text}
Evidence:
{context}
Answer:"""

        final_llm = await llm_chat(
            messages=[{"role": "user", "content": synthesis_prompt}],
            model=chosen_model,
            temperature=0.1,
            max_tokens=1200,
        )
        budget.update(final_llm)

        answer_text = ""
        if hasattr(final_llm, "results") and final_llm.results:
            answer_text = getattr(final_llm.results[0], "text", "")

        # ==================== RECOVERY CHUTES (now much stronger) ====================
        if not answer_text.strip():
            print("→ Synthesis failed → triggering STRONG Recovery Chutes pass")
            recovery_prompt = f"""You MUST give a direct, concise answer. Use the evidence below. Never say you cannot answer.
Query: {query_text}
Evidence:
{context[:4000]}
Answer:"""
            recovery_llm = await llm_chat(
                messages=[{"role": "user", "content": recovery_prompt}],
                model=chosen_model,
                temperature=0.0,
                max_tokens=1000,
            )
            budget.update(recovery_llm)
            if hasattr(recovery_llm, "results") and recovery_llm.results:
                answer_text = getattr(recovery_llm.results[0], "text", "")

        # ==================== LAYER 3: MANUAL EVIDENCE FALLBACK (never weak string again) ====================
        if not answer_text.strip() and ranked:
            print("→ Recovery also empty → using manual evidence fallback")
            answer_text = "Based on retrieved sources:\n"
            for i, r in enumerate(ranked[:5]):
                answer_text += f"{i+1}. {r.title}: {r.note}\n"

        final_text = answer_text.strip() or "Evidence retrieved but synthesis was constrained."

        citations = [CitationRef(receipt_id=r.receipt_id, result_id=r.result_id) for r in ranked[:3]] if ranked else []

        print(f"→ Champion run complete | Budget left: ${budget.remaining:.4f} | Citations: {len(citations)}")
        return Response(text=final_text, citations=citations)

    except Exception as e:
        print(f"→ Critical error: {type(e).__name__} - {e}")
        return await _emergency_direct_answer(query_text, chosen_model if 'chosen_model' in locals() else None, budget, ranked=[])


# ==================== EMERGENCY DIRECT ANSWER ====================
async def _emergency_direct_answer(query_text: str, model: str | None, budget: BudgetTracker, ranked: List[SearchResult]):
    if ranked:
        # Use evidence we already have
        text = "Evidence retrieved:\n"
        for i, r in enumerate(ranked[:5]):
            text += f"{i+1}. {r.title}: {r.note}\n"
        return Response(text=text, citations=[CitationRef(receipt_id=r.receipt_id, result_id=r.result_id) for r in ranked[:3]])
    if not model:
        return Response(text="Harnyx SN67 champion miner — deep research under budget.", citations=[])
    try:
        emergency_prompt = f"""Direct, concise answer to the query. No fluff.
Query: {query_text}
Answer:"""
        llm = await llm_chat(
            messages=[{"role": "user", "content": emergency_prompt}],
            model=model,
            temperature=0.0,
            max_tokens=600,
        )
        budget.update(llm)
        text = getattr(llm.results[0], "text", "") if hasattr(llm, "results") and llm.results else ""
        return Response(text=text.strip() or "Retrieved sources but budget/time constraints prevented full synthesis.", citations=[])
    except:
        return Response(text="Harnyx SN67 — competitive deep research platform.", citations=[])
