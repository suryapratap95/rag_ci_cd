"""RAG agent — retrieves context and calls the LLM."""
import os
import time

from anthropic import Anthropic

from .retriever import retrieve

_client = Anthropic()
MODEL = "claude-sonnet-4-5"

SYSTEM_PROMPT = (
    "You are a banking assistant. Use ONLY the context provided. "
    "Cite document IDs in square brackets like [D001]. "
    "If the context does not contain the answer, say so plainly."
)


def answer(question: str, top_k: int = 3) -> dict:
    """Return {answer, sources, latency_ms, input_tokens, output_tokens}."""
    docs = retrieve(question, top_k=top_k)
    context = "\n\n".join(f"[{d['id']}] {d['title']}\n{d['text']}" for d in docs)

    start = time.time()
    resp = _client.messages.create(
        model=MODEL, max_tokens=400, system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"}],
    )
    latency_ms = int((time.time() - start) * 1000)

    return {
        "answer": resp.content[0].text.strip(),
        "sources": [d["id"] for d in docs],
        "latency_ms": latency_ms,
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
    }
