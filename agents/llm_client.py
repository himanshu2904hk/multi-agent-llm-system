import os
import json
import logging
import time
from groq import Groq

logger = logging.getLogger(__name__)

_client = None


def get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _client


def chat(
    messages: list,
    model: str = None,
    temperature: float = 0.2,
    max_tokens: int = 2048,
    response_format: dict = None,
) -> tuple[str, int]:
    """Returns (content, total_tokens)"""
    model = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    client = get_client()

    kwargs = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if response_format:
        kwargs["response_format"] = response_format

    start = time.time()
    resp = client.chat.completions.create(**kwargs)
    latency = (time.time() - start) * 1000
    content = resp.choices[0].message.content
    tokens = resp.usage.total_tokens if resp.usage else 0
    logger.debug(f"[llm] model={model} tokens={tokens} latency={latency:.0f}ms")
    return content, tokens


def chat_json(messages: list, model: str = None, temperature: float = 0.1, max_tokens: int = 2048) -> tuple[dict, int]:
    """Chat expecting JSON response."""
    content, tokens = chat(
        messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    try:
        return json.loads(content), tokens
    except json.JSONDecodeError:
        # Attempt to extract JSON from markdown code blocks
        import re
        m = re.search(r"```(?:json)?\s*([\s\S]+?)```", content)
        if m:
            return json.loads(m.group(1).strip()), tokens
        raise
