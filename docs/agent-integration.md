# Agent Integration Guide

This repository now includes a concrete agent adoption layer for AGENT.md, MCP, and Skills.

## Files

| File | Purpose |
|------|---------|
| `AGENT.md` | Persistent project rules for coding agents. |
| `.mcp.example.json` | Example MCP host configuration for the local course MCP server plus optional GitHub read-only server. |
| `prompt-engineering/mcp/course_mcp_server.py` | Minimal local MCP server with course search/read tools. |
| `prompt-engineering/mcp/call_course_mcp.py` | Teaching client that launches the server and calls tools over stdio. |
| `prompt-engineering/mcp/course-agent.mcp.example.json` | Course-specific MCP host config. |
| `prompt-engineering/mcp/README.md` | Teaching scenario, run command, and MCP tool catalog used by the lesson notebook. |
| `prompt-engineering/agent-skills/course-notebook-author/SKILL.md` | Workflow for creating or revising course notebooks. |
| `prompt-engineering/agent-skills/agent-harness-review/SKILL.md` | Workflow for reviewing agent harness, MCP, Skills, RAG-agent, and tool-safety lessons. |

## How To Use

1. Point your agent host at this repository root so it can read `AGENT.md`.
2. For classroom demonstration, run the local client first:
   `uv run --with "mcp>=1.27,<2" python prompt-engineering/mcp/call_course_mcp.py`
3. For an MCP-capable host, copy `.mcp.example.json` or `prompt-engineering/mcp/course-agent.mcp.example.json` into the host's MCP configuration location.
4. Replace `${GITHUB_TOKEN}` with an environment variable in your shell or secret manager, not a hard-coded token.
5. Start with read-only workflows. Confirm the agent can call `search_course_files` and `read_course_file` before enabling write actions.
6. When asking the agent to perform repeated course work, name the relevant skill:
   - `course-notebook-author`
   - `agent-harness-review`
7. In teaching, use the tool catalog in `prompt-engineering/mcp/README.md` to explain how the agent chooses MCP tools.

## Recommended Permission Boundary

- Local course MCP server: read-only access under `prompt-engineering/`.
- GitHub: read issues, PRs, review comments, and checks by default.
- External writes: require human confirmation for comments, branch pushes, merge operations, email, database writes, and file deletion.

## Example Prompts

```text
Use the course-notebook-author skill to add a notebook for agentic RAG evaluation under prompt-engineering/07-calibration-eval.
```

```text
Use the agent-harness-review skill to check whether the MCP lesson explains trust boundaries and actual project adoption.
```

```text
Use the agent-harness-review skill and the course MCP example to explain how an agent chooses course-files.search before editing a notebook.
```

```text
Run the local course MCP client, then explain how search_course_files differs from ordinary function calling.
```

## Validation

For notebook edits, run a syntax check on every modified notebook:

```bash
python - <<'PY'
import ast, json, pathlib
paths = [
    pathlib.Path("prompt-engineering/05-agent-harness/10-agent-md-mcp-skills.ipynb")
]
for p in paths:
    nb = json.loads(p.read_text())
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") == "code":
            src = "".join(cell.get("source", [])) if isinstance(cell.get("source"), list) else cell.get("source", "")
            ast.parse(src, filename=f"{p}:cell_{i}")
    print(f"{p}: ok")
PY
```
