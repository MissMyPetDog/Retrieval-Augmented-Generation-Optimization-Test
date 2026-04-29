"""
LLM-based answer generator -- baseline implementation.

Two modes:
  1. SIMULATED: Mimics API latency without real calls (default, no key needed)
  2. REAL API:  Calls Anthropic/OpenAI API for actual generation

This is the LAST step in the RAG pipeline:
  Retrieved docs + Query -> [Generator] -> Answer

Optimization targets:
  - Week 9 (Concurrency):  asyncio for non-blocking API calls
  - Week 9 (Concurrency):  Batch multiple queries concurrently
  - Week 9 (Concurrency):  Streaming responses (start showing answer before it's complete)
  - Week 2 (Performance):  Prompt optimization to reduce token count -> faster response
"""
import time
import numpy as np
from typing import Optional

import config


# ==============================================
# PROMPT FORMATTING
# ==============================================

def format_prompt(query: str, contexts: list[str], max_context_chars: int = 3000) -> str:
    """
    Format the RAG prompt with retrieved contexts.

    +-------------------------------------------------------------+
    | OPTIMIZATION OPPORTUNITY [Week 2 - Performance Tips]:        |
    |                                                              |
    | Fewer tokens in the prompt = faster LLM response.            |
    | - Truncate long contexts intelligently                       |
    | - Remove redundant/overlapping passages                      |
    | - Reorder: most relevant first (LLMs attend more to start)  |
    +-------------------------------------------------------------+
    """
    # Truncate contexts to fit within budget
    truncated = []
    total_chars = 0
    for i, ctx in enumerate(contexts):
        remaining = max_context_chars - total_chars
        if remaining <= 0:
            break
        if len(ctx) > remaining:
            ctx = ctx[:remaining] + "..."
        truncated.append(ctx)
        total_chars += len(ctx)

    context_block = "\n\n---\n\n".join(
        f"[Document {i+1}]\n{ctx}" for i, ctx in enumerate(truncated)
    )

    return (
        "Answer the question based on the provided context. "
        "If the context doesn't contain enough information, say so.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )


# ==============================================
# BASELINE GENERATOR (sequential, blocking)
# ==============================================

class BaselineGenerator:
    """
    Sequential LLM generator. Each call blocks until the full response is received.

    +-------------------------------------------------------------+
    | OPTIMIZATION OPPORTUNITIES:                                  |
    |                                                              |
    | 1. [Week 9] asyncio -- non-blocking generation:               |
    |    While waiting for LLM response (~500ms-2s), the program  |
    |    is idle. With async, you can:                             |
    |    - Process the next query's retrieval while waiting        |
    |    - Send multiple generation requests concurrently          |
    |                                                              |
    | 2. [Week 9] Streaming -- progressive response:                |
    |    Instead of waiting for the FULL response, stream tokens   |
    |    as they arrive. User sees the answer building in          |
    |    real-time. Perceived latency drops from 2s to ~100ms.    |
    |                                                              |
    | 3. [Week 9] ThreadPoolExecutor -- batch generation:           |
    |    Given N queries, send all generation requests to a        |
    |    thread pool. Since API calls are IO-bound, threads        |
    |    work well here (GIL released during network wait).        |
    |                                                              |
    | 4. [Week 2] Prompt optimization:                              |
    |    Shorter prompts = fewer input tokens = faster response.   |
    |    Also reduces API cost.                                    |
    +-------------------------------------------------------------+
    """

    def __init__(
        self,
        api_provider: str = "simulated",
        api_key: str = None,
        model_name: str = None,
        simulated_latency_ms: float = 500.0,
        max_tokens: int = 256,
        base_url: str = None,
        extra_headers: dict = None,
    ):
        self.api_provider = api_provider
        self.api_key = api_key
        self.model_name = model_name or self._default_model()
        self.simulated_latency_ms = simulated_latency_ms
        self.max_tokens = max_tokens
        self.base_url = base_url
        self.extra_headers = extra_headers or {}

        # Stats
        self.total_requests = 0
        self.total_latency_ms = 0.0

    def _default_model(self):
        if self.api_provider == "anthropic":
            return "claude-sonnet-4-20250514"
        elif self.api_provider == "openai":
            return "gpt-4o-mini"
        return "simulated"

    def generate(self, query: str, contexts: list[str]) -> dict:
        """
        Generate an answer for a single query with retrieved contexts.

        Returns dict with:
          - answer: the generated text
          - latency_ms: time taken
          - prompt_tokens: approximate token count sent
        """
        prompt = format_prompt(query, contexts)
        prompt_tokens = len(prompt.split()) * 1.3  # rough estimate

        t0 = time.perf_counter()

        if self.api_provider == "simulated":
            answer = self._simulated_generate(query, contexts)

        elif self.api_provider == "anthropic":
            answer = self._anthropic_generate(prompt)

        elif self.api_provider == "openai":
            answer = self._openai_generate(prompt)

        else:
            raise ValueError(f"Unknown provider: {self.api_provider}")

        latency_ms = (time.perf_counter() - t0) * 1000
        self.total_requests += 1
        self.total_latency_ms += latency_ms

        return {
            "answer": answer,
            "latency_ms": latency_ms,
            "prompt_tokens": int(prompt_tokens),
        }

    def generate_batch(self, items: list[tuple[str, list[str]]]) -> list[dict]:
        """
        Generate answers for multiple (query, contexts) pairs.
        Baseline: sequential, one at a time.

        +-------------------------------------------------------------+
        | THIS IS THE MAIN OPTIMIZATION TARGET FOR GENERATION.         |
        |                                                              |
        | Baseline: N queries ? 500ms each = N ? 0.5s total           |
        |                                                              |
        | With ThreadPoolExecutor (N threads):                         |
        |   All N requests sent simultaneously -> ~500ms total          |
        |   Speedup: ~Nx                                               |
        |                                                              |
        | With asyncio:                                                |
        |   Same idea but with async/await syntax                      |
        |   Even lower overhead than threads for many concurrent calls |
        +-------------------------------------------------------------+
        """
        results = []
        for i, (query, contexts) in enumerate(items):
            result = self.generate(query, contexts)
            results.append(result)
        return results

    def _simulated_generate(self, query: str, contexts: list[str]) -> str:
        """Simulate LLM API with realistic latency."""
        # Simulate variable latency (real APIs have jitter)
        jitter = np.random.uniform(0.8, 1.2)
        time.sleep(self.simulated_latency_ms * jitter / 1000.0)

        return (
            f"Based on the {len(contexts)} retrieved documents, "
            f"the answer to '{query[:50]}...' is: "
            f"[simulated response with ~{self.simulated_latency_ms:.0f}ms latency]"
        )

    def _anthropic_generate(self, prompt: str) -> str:
        """Call Anthropic Claude API."""
        import anthropic
        client = anthropic.Anthropic(api_key=self.api_key)
        response = client.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def _openai_generate(self, prompt: str) -> str:
        """Call OpenAI-compatible API (supports custom base_url and headers, e.g., NYU ChatGPT API)."""
        import openai
        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        if self.extra_headers:
            client_kwargs["default_headers"] = self.extra_headers
        client = openai.OpenAI(**client_kwargs)
        response = client.chat.completions.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    # ----------------------------------------------------------------
    # Streaming generation (Week 9 - Concurrency; TTFT optimization)
    # ----------------------------------------------------------------
    def generate_stream(self, query: str, contexts: list[str]) -> dict:
        """
        Same as .generate() but uses streaming responses when possible.

        Returns dict with:
          - answer:        full text
          - ttft_ms:       time to FIRST token (perceived latency)
          - total_ms:      time until full response complete
          - prompt_tokens: rough input-token estimate

        Non-streaming providers (simulated baseline path) set ttft_ms == total_ms.
        """
        prompt = format_prompt(query, contexts)
        prompt_tokens = int(len(prompt.split()) * 1.3)

        if self.api_provider == "simulated":
            return self._simulated_generate_stream(query, contexts, prompt_tokens)
        elif self.api_provider == "openai":
            return self._openai_generate_stream(prompt, prompt_tokens)
        elif self.api_provider == "anthropic":
            # Anthropic also supports streaming; add later if needed.
            raise NotImplementedError("Streaming not implemented for Anthropic here.")
        raise ValueError(f"Unknown provider: {self.api_provider}")

    def _simulated_generate_stream(self, query: str, contexts: list[str],
                                   prompt_tokens: int) -> dict:
        """Fake streaming: first-token delay + per-token delay."""
        t0 = time.perf_counter()
        # Simulate roundtrip + model loading for the first chunk
        initial = (self.simulated_latency_ms * 0.2) / 1000.0 * np.random.uniform(0.8, 1.2)
        time.sleep(initial)
        ttft_ms = (time.perf_counter() - t0) * 1000
        # Simulate the rest of the tokens
        rest = (self.simulated_latency_ms * 0.8) / 1000.0 * np.random.uniform(0.8, 1.2)
        time.sleep(rest)
        total_ms = (time.perf_counter() - t0) * 1000
        return {
            "answer": f"[simulated streamed response for '{query[:40]}...']",
            "ttft_ms": ttft_ms,
            "total_ms": total_ms,
            "prompt_tokens": prompt_tokens,
        }

    def _openai_generate_stream(self, prompt: str, prompt_tokens: int) -> dict:
        """OpenAI-compatible streaming call (works with NYU ChatGPT API)."""
        import openai
        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        if self.extra_headers:
            client_kwargs["default_headers"] = self.extra_headers
        client = openai.OpenAI(**client_kwargs)

        t0 = time.perf_counter()
        stream = client.chat.completions.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )

        ttft_ms: Optional[float] = None
        parts: list[str] = []
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - t0) * 1000
                parts.append(content)

        total_ms = (time.perf_counter() - t0) * 1000
        return {
            "answer": "".join(parts),
            "ttft_ms": ttft_ms if ttft_ms is not None else total_ms,
            "total_ms": total_ms,
            "prompt_tokens": prompt_tokens,
        }


# ==============================================
# Quick self-test
# ==============================================

if __name__ == "__main__":
    print("=== Single Generation (simulated, 500ms) ===")
    gen = BaselineGenerator(api_provider="simulated", simulated_latency_ms=500)

    result = gen.generate(
        query="What causes climate change?",
        contexts=["Carbon dioxide traps heat...", "Greenhouse gases include..."],
    )
    print(f"  Answer: {result['answer'][:100]}...")
    print(f"  Latency: {result['latency_ms']:.0f}ms")

    print(f"\n=== Batch Generation (10 queries, sequential) ===")
    items = [
        (f"Question {i}?", [f"Context for question {i}"])
        for i in range(10)
    ]

    t0 = time.perf_counter()
    results = gen.generate_batch(items)
    total = (time.perf_counter() - t0) * 1000

    print(f"  10 queries ? ~500ms = {total:.0f}ms total")
    print(f"  Average per query: {total / 10:.0f}ms")
    print(f"\n  -> With async/threading optimization:")
    print(f"    All 10 queries in parallel = ~500ms total")
    print(f"    Expected speedup: ~{total / 500:.0f}x")
