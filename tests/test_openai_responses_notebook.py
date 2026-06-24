import json
from pathlib import Path


NOTEBOOK = Path("prompt-engineering/05-agent-harness/09-responses-api.ipynb")


def _notebook_text() -> str:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
    )


def test_responses_api_notebook_uses_current_openai_entrypoint():
    text = _notebook_text()

    assert "client.responses.create" in text
    assert "client.chat.completions" not in text
    assert "OPENAI_MODEL" in text
    assert 'model="gpt-4o"' not in text
    assert 'model="gpt-4o-mini"' not in text


def test_responses_api_notebook_explains_state_boundaries():
    text = _notebook_text()

    assert "previous_response_id" in text
    assert "Conversations API" in text
    assert "Assistants API 已棄用" in text
    assert "Chat Completions 仍可用於簡單聊天" in text
