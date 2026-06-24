"""Search helpers for the teaching deep-research graph."""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request

from .state import Source


def search_web(query: str, *, max_results: int = 4, offline: bool = False) -> list[Source]:
    """Search the web with Tavily when configured, otherwise return a traceable stub.

    The fallback keeps notebooks and tests runnable without paid search keys. It is
    intentionally explicit, so students can see when evidence is simulated.
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    if offline or not api_key:
        return [
            Source(
                title=f"Offline research note for: {query}",
                url="offline://local-research-note",
                snippet=(
                    "No TAVILY_API_KEY was configured. Treat this as a placeholder "
                    "and connect a real search API before using the report as evidence."
                ),
            )
        ]

    payload = json.dumps(
        {
            "query": query,
            "max_results": max_results,
            "include_answer": False,
            "include_raw_content": False,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        "https://api.tavily.com/search",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        data = json.loads(response.read().decode("utf-8"))

    sources: list[Source] = []
    for item in data.get("results", [])[:max_results]:
        sources.append(
            Source(
                title=item.get("title") or urllib.parse.urlparse(item.get("url", "")).netloc,
                url=item.get("url", ""),
                snippet=item.get("content", ""),
            )
        )
    return sources


def format_sources(sources: list[Source]) -> str:
    """Format sources for a model prompt."""
    if not sources:
        return "No sources found."
    return "\n".join(
        f"- {source.title}\n  URL: {source.url}\n  Snippet: {source.snippet}"
        for source in sources
    )
