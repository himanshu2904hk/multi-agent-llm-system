import time
import hashlib
from tools.base import ToolResult, FailureMode

# Stub web search returning structured results with source URLs and relevance scores
STUB_INDEX = {
    "python": [
        {"title": "Python official docs", "url": "https://docs.python.org", "snippet": "Python is a high-level programming language.", "relevance": 0.95},
        {"title": "Python Wikipedia", "url": "https://en.wikipedia.org/wiki/Python_(programming_language)", "snippet": "Python was created by Guido van Rossum.", "relevance": 0.88},
    ],
    "machine learning": [
        {"title": "ML Overview", "url": "https://en.wikipedia.org/wiki/Machine_learning", "snippet": "Machine learning is a subset of AI that enables systems to learn from data.", "relevance": 0.93},
        {"title": "Scikit-learn docs", "url": "https://scikit-learn.org/stable/", "snippet": "Scikit-learn provides simple tools for predictive data analysis.", "relevance": 0.87},
    ],
    "docker": [
        {"title": "Docker docs", "url": "https://docs.docker.com", "snippet": "Docker is a platform for developing, shipping, and running applications.", "relevance": 0.96},
        {"title": "Docker Hub", "url": "https://hub.docker.com", "snippet": "Docker Hub is the world's largest container image registry.", "relevance": 0.82},
    ],
    "llm": [
        {"title": "Large language models survey", "url": "https://arxiv.org/abs/2303.18223", "snippet": "LLMs are neural networks trained on large corpora of text.", "relevance": 0.94},
        {"title": "OpenAI GPT overview", "url": "https://openai.com/research/gpt-4", "snippet": "GPT-4 is a large multimodal model that accepts image and text inputs.", "relevance": 0.91},
    ],
    "fastapi": [
        {"title": "FastAPI docs", "url": "https://fastapi.tiangolo.com", "snippet": "FastAPI is a modern, fast web framework for building APIs with Python.", "relevance": 0.97},
    ],
    "default": [
        {"title": "Wikipedia", "url": "https://en.wikipedia.org", "snippet": "General knowledge resource.", "relevance": 0.5},
        {"title": "Google", "url": "https://google.com", "snippet": "Search engine.", "relevance": 0.4},
    ],
}


def web_search(query: str, top_k: int = 3, timeout_ms: int = 5000) -> ToolResult:
    start = time.time()

    if not query or not query.strip():
        return ToolResult(
            success=False,
            failure_mode=FailureMode.malformed_input,
            error_message="Query must be a non-empty string.",
            latency_ms=0.0,
        )

    if timeout_ms <= 0:
        return ToolResult(
            success=False,
            failure_mode=FailureMode.timeout,
            error_message="Simulated timeout: timeout_ms must be positive.",
            latency_ms=float(timeout_ms),
        )

    query_lower = query.lower()
    matched = STUB_INDEX.get("default", [])
    for key, results in STUB_INDEX.items():
        if key in query_lower:
            matched = results
            break

    results = sorted(matched, key=lambda x: x["relevance"], reverse=True)[:top_k]
    latency = (time.time() - start) * 1000

    if not results:
        return ToolResult(
            success=False,
            failure_mode=FailureMode.empty_results,
            error_message="No results found for query.",
            latency_ms=latency,
        )

    return ToolResult(
        success=True,
        data={"query": query, "results": results, "result_count": len(results)},
        latency_ms=latency,
    )
