"""Minimal MCP server for the prompt-engineering course.

Run from repository root:
    uv run --with "mcp>=1.27,<2" python prompt-engineering/mcp/course_mcp_server.py

This server is intentionally small for teaching:
- `search_course_files` searches text files under prompt-engineering/
- `read_course_file` reads one allowed course/adoption file
"""

from __future__ import annotations

from pathlib import Path

from mcp.server.fastmcp import FastMCP


REPO_ROOT = Path(__file__).resolve().parents[2]
COURSE_ROOT = REPO_ROOT / "prompt-engineering"
ALLOWED_SUFFIXES = {".md", ".py", ".txt", ".json", ".toml", ".ipynb"}
ALLOWED_SINGLE_FILES = {
    (REPO_ROOT / "AGENT.md").resolve(),
    (REPO_ROOT / ".mcp.example.json").resolve(),
    (REPO_ROOT / "docs" / "agent-integration.md").resolve(),
}

mcp = FastMCP("prompt-engineering-course")


def _safe_course_path(path: str) -> Path:
    target = (REPO_ROOT / path).resolve()
    if not target.is_relative_to(COURSE_ROOT) and target not in ALLOWED_SINGLE_FILES:
        raise ValueError("Only prompt-engineering/ files and selected adoption docs are allowed.")
    if target.suffix not in ALLOWED_SUFFIXES:
        raise ValueError(f"Unsupported file type: {target.suffix}")
    return target


@mcp.tool()
def search_course_files(query: str, max_results: int = 8) -> dict:
    """Search course files under prompt-engineering/ for a keyword."""
    query_lower = query.lower()
    matches: list[dict] = []

    for path in sorted(COURSE_ROOT.rglob("*")):
        if not path.is_file() or path.suffix not in ALLOWED_SUFFIXES:
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        index = text.lower().find(query_lower)
        if index == -1:
            continue
        snippet = text[max(0, index - 80) : index + 160].replace("\n", " ")
        matches.append({"path": rel, "snippet": snippet})
        if len(matches) >= max_results:
            break

    return {"query": query, "matches": matches}


@mcp.tool()
def read_course_file(path: str, max_chars: int = 4000) -> dict:
    """Read a course file or selected agent adoption doc."""
    target = _safe_course_path(path)
    if not target.exists() or not target.is_file():
        raise FileNotFoundError(path)

    text = target.read_text(encoding="utf-8", errors="ignore")
    rel = target.relative_to(REPO_ROOT).as_posix()
    return {
        "path": rel,
        "truncated": len(text) > max_chars,
        "text": text[:max_chars],
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
