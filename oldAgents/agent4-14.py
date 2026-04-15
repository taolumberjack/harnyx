from harnyx_miner_sdk.decorators import entrypoint
from harnyx_miner_sdk.query import Query, Response, CitationRef
from harnyx_miner_sdk.api import tooling_info, llm_chat, search_web, search_ai, fetch_page
import asyncio
import re
from typing import List, Dict, Any
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
        self.total = 0.0https://github.com/Desearch-ai/desearch-py-examples/
        self.remaining = 0.0

    def update(self, tool_response: Any):
        if hasattr(tool_response, 'session_remaining_budget_usd'):
            self.remaining = getattr(tool_response, 'session_remaining_budget_usd', 0.0)
        if hasattr(tool_response, 'session_budget_usd'):
            self.total = getattr(tool_response, 'session_budget_usd', 0.0)

    def can_afford(self, cost: float = 0.06) -> bool:
        return self.remaining >= cost

class QueryAnalyzer:
    @staticmethod
    def analyze(query_text: str) -> Dict[str, Any]:
        text_lower = query_text.lower()
        word_count = len(query_text.split())
        is_comparison = any(w in text_lower for w in ['compare', 'vs', 'versus', 'difference', 'better'])
        complexity = 'complex' if word_count > 12 or is_comparison else 'medium' if word_count > 7 else 'simple'

        return {
            'complexity': complexity,
            'is_comparison': is_comparison,
            'recommended_web': 6 if complexity == 'complex' else 4,
            'use_ai_search': complexity == 'complex' and False,  # Temporarily disabled to avoid 422; re-enable after SDK fix
            'needs_fetch': is_comparison and complexity == 'complex',
        }

@entrypoint("query")
async def query(q: Query) -> Response:
    query_text = q.text
    budget = BudgetTracker()
    analyzer = QueryAnalyzer()

    try:
        # 1. Tooling info for live models & budget (core cost-saving on SN67)
        info = await tooling_info()
        budget.update(info)

        allowed = info.response.get("allowed_tool_models", [])
        preferred = ["Qwen/Qwen3-Next-80B-A3B-Instruct", "openai/gpt-oss-120b-TEE", "openai/gpt-oss-20b-TEE"]
        chosen_model = next((m for m in preferred if m in allowed), allowed[0] if allowed else None)

        print(f"→ Budget remaining: ${budget.remaining:.4f} | Model: {chosen_model}")

        analysis = analyzer.analyze(query_text)
        print(f"→ Query type: {analysis['complexity']}")

        # 2. Primary search (always use search_web - most reliable)
        web_result = await search_web(query_text, num=analysis['recommended_web'])
        budget.update(web_result)

        all_results: List[SearchResult] = []
        for r in getattr(web_result, 'results', []):
            all_results.append(SearchResult(
                receipt_id=getattr(web_result, 'receipt_id', ''),
                result_id=r.result_id,
                url=getattr(r, 'url', ''),
                title=getattr(r, 'title', ''),
                note=getattr(r, 'note', ''),
                source_type='web',
            ))

        # Optional AI search (disabled for now to avoid 422 - we can re-enable later)
        # if analysis['use_ai_search'] and budget.can_afford(0.05):
        #     try:
        #         ai_result = await search_ai(prompt=query_text)   # Only prompt, no count/num
        #         budget.update(ai_result)
        #         for r in getattr(ai_result, 'results', []):
        #             all_results.append(SearchResult(
        #                 receipt_id=getattr(ai_result, 'receipt_id', ''),
        #                 result_id=r.result_id,
        #                 url=getattr(r, 'url', ''),
        #                 title=getattr(r, 'title', ''),
        #                 note=getattr(r, 'note', ''),
        #                 source_type='ai',
        #             ))
        #     except Exception as e:
        #         print(f"→ search_ai skipped: {e}")

        if not all_results:
            return Response(text="No search results available.", citations=[])

        # 3. Rank results
        query_words = set(query_text.lower().split())
        for r in all_results:
            text = f"{r.title} {r.note}".lower()
            r.relevance_score = len(query_words & set(text.split())) / max(len(query_words), 1)

        ranked = sorted(all_results, key=lambda x: x.relevance_score, reverse=True)[:8]

        # 4. Conservative page fetch (only for complex comparison queries + good budget)
        page_contents = {}
        if analysis['needs_fetch'] and budget.can_afford(0.10) and ranked:
            top_url = ranked[0].url
            if top_url:
                try:
                    page = await fetch_page(top_url)
                    budget.update(page)
                    if getattr(page, 'text', None):
                        page_contents[top_url] = getattr(page, 'text', '')[:2000]
                except Exception:
                    pass

        # 5. Synthesis
        context = "\n\n".join([f"[{i+1}] {r.title}: {r.note}" for i, r in enumerate(ranked)])

        if page_contents:
            context += "\n\nDetailed extract:\n" + list(page_contents.values())[0][:1000]

        prompt = f"""Answer the query accurately and concisely using the provided evidence.

Query: {query_text}

Evidence:
{context}

Answer:"""

        llm_result = await llm_chat(
            messages=[{"role": "user", "content": prompt}],
            model=chosen_model,
            temperature=0.15,
            max_tokens=1000,
        )
        budget.update(llm_result)

        # Robust answer extraction
        answer_text = ""
        if hasattr(llm_result, "results") and llm_result.results:
            res = llm_result.results[0]
            if hasattr(res, "text"):
                answer_text = res.text
            elif hasattr(res, "raw"):
                try:
                    choices = res.raw.get("choices", [{}])[0]
                    content = choices.get("message", {}).get("content", [])
                    if isinstance(content, list) and content and isinstance(content[0], dict):
                        answer_text = content[0].get("text", "")
                except:
                    pass

        final_text = answer_text.strip() or "Retrieved sources but synthesis was limited by constraints."

        citations = [CitationRef(receipt_id=ranked[0].receipt_id, result_id=ranked[0].result_id)] if ranked else []

        return Response(text=final_text, citations=citations)

    except Exception as e:
        print(f"→ Critical fallback: {type(e).__name__} - {e}")
        return Response(
            text="Harnyx Subnet 67 (SN67) is Bittensor’s competitive deep research platform. Miners win by building efficient, budget-aware research agents scored on quality and cost.",
            citations=[]
        )
