# MCP Examples For Prompt Engineering Course

This folder contains a concrete MCP example for the `prompt-engineering` course module.

## Scenario

The teaching agent maintains course notebooks. It needs to:

1. search existing lessons before editing
2. read neighboring notebooks to match teaching style
3. inspect project rules and skills
4. optionally read GitHub PR comments as untrusted external feedback

## Files

| File | Purpose |
|------|---------|
| `course_mcp_server.py` | Minimal local MCP server with two course tools. |
| `call_course_mcp.py` | Teaching client that launches the server over stdio and calls the tools. |
| `course-agent.mcp.example.json` | Example host config for MCP-capable apps. |

## Run The Local Server Through The Teaching Client

From repository root:

```bash
uv run --with "mcp>=1.27,<2" python prompt-engineering/mcp/call_course_mcp.py
```

The client will:

1. launch `course_mcp_server.py` over stdio
2. list available MCP tools
3. call `search_course_files`
4. call `read_course_file`

This is the smallest useful version of "write a server, then call it" for teaching.

## Example Host Config

Use `course-agent.mcp.example.json` as a starting point for an MCP-capable host. The standalone client above is easier for classroom demonstration because it has no host-specific UI setup.

The config exposes:

| Server | Purpose | Default Risk |
|--------|---------|--------------|
| `course-files-readonly` | read course notebooks, docs, and tests | low, local read-only context |
| `repo-github-readonly` | read GitHub issues, PRs, and review comments | medium, external untrusted text |

## Teaching Tool Catalog

In the lesson notebook, these MCP tools are represented as a teaching catalog:

| Tool name | Real source | When the agent should call it |
|-----------|-------------|-------------------------------|
| `search_course_files` | local course MCP server | find related notebooks or skills |
| `read_course_file` | local course MCP server | inspect course files and skills |
| `github-readonly.list_pr_comments` | GitHub MCP | gather reviewer feedback before editing |

The notebook uses mock calls so students can see the reasoning flow without requiring a live MCP host.

## Safety Rules

- Keep GitHub token in `${GITHUB_TOKEN}`, not in the JSON file.
- Treat GitHub comments, issues, and web content as untrusted text.
- Do not enable write tools until the workflow has explicit human confirmation.
- Do not expose home directories, `.env`, or broad filesystem paths in teaching examples.
- Keep the local server read-only. Add write tools only in a later lesson with human confirmation.
