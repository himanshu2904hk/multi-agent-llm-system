import logging
from agents.llm_client import chat_json

logger = logging.getLogger(__name__)

COMPRESSION_PROMPT = """Summarize the following context to reduce token count.
Rules:
- Preserve ALL structured data: tool outputs, scores, citations, chunk IDs, numerical values
- You may compress only conversational filler and verbose prose
- Return JSON: {"compressed": "..."}
"""


def compress(text: str, max_tokens: int = 1000) -> str:
    """Lossy compression for prose, lossless for structured data."""
    if len(text) // 4 <= max_tokens:
        return text
    try:
        result, _ = chat_json([
            {"role": "system", "content": COMPRESSION_PROMPT},
            {"role": "user", "content": f"Compress this context (target ~{max_tokens} tokens):\n\n{text}"},
        ])
        return result.get("compressed", text[:max_tokens * 4])
    except Exception as e:
        logger.error(f"[compression] Error: {e}")
        return text[:max_tokens * 4]
