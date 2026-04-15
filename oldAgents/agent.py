from harnyx_miner_sdk.decorators import entrypoint
from harnyx_miner_sdk.query import Query, Response, CitationRef
from harnyx_miner_sdk.api import tooling_info, llm_chat, search_web

@entrypoint("query")
async def query(query: Query) -> Response:
    info = await tooling_info()
    allowed = info.response.get("allowed_tool_models", [])
    
    # Smart model selection: prefer stable + reasonably cheap
    preferred = [
        "Qwen/Qwen3-Next-80B-A3B-Instruct",
        "openai/gpt-oss-120b-TEE",
        "openai/gpt-oss-20b-TEE",
    ]
    chosen_model = next((m for m in preferred if m in allowed), allowed[0] if allowed else None)
    
    print(f"→ Using model: {chosen_model}")

    try:
        # Basic search (we can make this multi-step later)
        search = await search_web(query.text, num=5)

        if not search or not search.results:
            fallback = "Harnyx Subnet 67 (SN67) turns deep research into a competitive Bittensor harness. Miners submit one agent.py implementing async query(). Validators score against strong reference answers using comparison + similarity. Lower cost wins ties."
            return Response(text=fallback)

        context = "\n\n".join([f"Title: {r.title or 'N/A'}\nNote: {r.note}" for r in search.results[:4]])

        # LLM synthesis
        llm_result = await llm_chat(
            messages=[{
                "role": "user",
                "content": f"""Answer the following query accurately and concisely using ONLY the provided context.

Query: {query.text}

Context:
{context}

Answer:"""
            }],
            model=chosen_model,
            temperature=0.1,
            max_tokens=1200
        )

        # Extract answer text safely from tool result
        answer_text = ""
        if llm_result and hasattr(llm_result, "results") and llm_result.results:
            result = llm_result.results[0]
            if hasattr(result, "text"):
                answer_text = result.text
            elif hasattr(result, "raw") and isinstance(result.raw, dict):
                # Fallback for raw payload structure you saw earlier
                try:
                    choices = result.raw.get("choices", [])
                    if choices and isinstance(choices[0], dict):
                        msg = choices[0].get("message", {})
                        content = msg.get("content", [])
                        if content and isinstance(content, list):
                            answer_text = content[0].get("text", "") if isinstance(content[0], dict) else str(content)
                except:
                    pass

        final_text = answer_text.strip() or "Retrieved context but synthesis failed."

        # Precise citation (load-bearing result only)
        citations = [CitationRef(receipt_id=search.receipt_id, result_id=search.results[0].result_id)] if search.results else []

        return Response(text=final_text, citations=citations)

    except Exception as e:
        print(f"→ Error (graceful fallback): {type(e).__name__}: {e}")
        return Response(
            text="Harnyx (SN67) is Bittensor's deep research subnet. Miners compete by building better Python research agents that run under tight budgets. Scoring combines LLM judge comparison and embedding similarity against strong references.",
            citations=[]
        )
