"""
LLM-based answer generator.

This is a thin wrapper around an LLM API. NOT an optimization target —
exists only to complete the RAG pipeline for demo purposes.
"""


def format_prompt(query: str, contexts: list[str]) -> str:
    """Format the RAG prompt with retrieved contexts."""
    context_block = "\n\n---\n\n".join(
        f"[Document {i+1}]\n{ctx}" for i, ctx in enumerate(contexts)
    )
    return (
        "Answer the question based on the provided context. "
        "If the context doesn't contain enough information, say so.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )


def generate_answer(query: str, contexts: list[str]) -> str:
    """
    Generate answer using an LLM.
    For this project, we use a simple placeholder.
    Replace with actual API call (OpenAI, Anthropic, local model) as needed.
    """
    prompt = format_prompt(query, contexts)

    # ── Placeholder: return the prompt for now ──
    # In production, you'd call an API here:
    #
    #   import anthropic
    #   client = anthropic.Anthropic()
    #   response = client.messages.create(
    #       model="claude-sonnet-4-20250514",
    #       max_tokens=512,
    #       messages=[{"role": "user", "content": prompt}],
    #   )
    #   return response.content[0].text

    return f"[LLM would answer based on {len(contexts)} retrieved documents]"
