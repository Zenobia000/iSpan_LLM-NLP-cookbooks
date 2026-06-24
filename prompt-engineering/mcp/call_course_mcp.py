"""Teaching client that calls the local course MCP server over stdio.

Run from repository root:
    uv run --with "mcp>=1.27,<2" python prompt-engineering/mcp/call_course_mcp.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client


REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_PATH = REPO_ROOT / "prompt-engineering" / "mcp" / "course_mcp_server.py"


def _text_result(result: types.CallToolResult) -> str:
    parts: list[str] = []
    for item in result.content:
        if isinstance(item, types.TextContent):
            parts.append(item.text)
    return "\n".join(parts)


async def main() -> None:
    server = StdioServerParameters(
        command="uv",
        args=["run", "--with", "mcp>=1.27,<2", "python", str(SERVER_PATH)],
    )

    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("Available MCP tools:")
            for tool in tools.tools:
                print(f"- {tool.name}: {tool.description}")

            search_result = await session.call_tool(
                "search_course_files",
                arguments={"query": "Agentic RAG", "max_results": 3},
            )
            print("\nsearch_course_files result:")
            print(_text_result(search_result))

            read_result = await session.call_tool(
                "read_course_file",
                arguments={
                    "path": "prompt-engineering/agent-skills/agent-harness-review/SKILL.md",
                    "max_chars": 1200,
                },
            )
            print("\nread_course_file result:")
            print(_text_result(read_result))


if __name__ == "__main__":
    asyncio.run(main())
