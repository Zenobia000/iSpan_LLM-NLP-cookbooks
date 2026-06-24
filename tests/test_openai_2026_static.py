import json
from pathlib import Path


MODERN_OPENAI_NOTEBOOKS = [
    Path("prompt-engineering/01-uncontrollability/01-openai-api-intro.ipynb"),
    Path("prompt-engineering/01-uncontrollability/02-sampling-and-uncertainty.ipynb"),
    Path("prompt-engineering/01-uncontrollability/04-multimodal.ipynb"),
    Path("prompt-engineering/02-intent-convergence/01-prompt-basics.ipynb"),
    Path("prompt-engineering/02-intent-convergence/02-few-shot-and-reasoning.ipynb"),
    Path("prompt-engineering/02-intent-convergence/03-prompt-chaining.ipynb"),
    Path("prompt-engineering/02-intent-convergence/04-spec-writing.ipynb"),
    Path("prompt-engineering/03-structured-output/01-json-mode.ipynb"),
    Path("prompt-engineering/03-structured-output/02-function-calling-basics.ipynb"),
    Path("prompt-engineering/03-structured-output/03-structured-extraction.ipynb"),
    Path("prompt-engineering/03-structured-output/04-classification-gradio.ipynb"),
    Path("prompt-engineering/04-knowledge-rag/01-embedding.ipynb"),
    Path("prompt-engineering/04-knowledge-rag/02-vanilla-rag.ipynb"),
    Path("prompt-engineering/04-knowledge-rag/03-similarity-and-relevance.ipynb"),
    Path("prompt-engineering/04-knowledge-rag/04-vector-db-rag.ipynb"),
    Path("prompt-engineering/04-knowledge-rag/05-dynamic-few-shot.ipynb"),
    Path("prompt-engineering/04-knowledge-rag/06-advanced-rag.ipynb"),
    Path("prompt-engineering/04-knowledge-rag/07-pdf-parsing.ipynb"),
    Path("prompt-engineering/05-agent-harness/01-function-calling-agents.ipynb"),
    Path("prompt-engineering/05-agent-harness/02-react-loop.ipynb"),
    Path("prompt-engineering/05-agent-harness/04-function-calling-rag.ipynb"),
    Path("prompt-engineering/05-agent-harness/05-shop-guardrails.ipynb"),
    Path("prompt-engineering/05-agent-harness/06-prompt-injection.ipynb"),
    Path("prompt-engineering/05-agent-harness/07-plugin-tools.ipynb"),
    Path("prompt-engineering/05-agent-harness/08-chatbot.ipynb"),
    Path("prompt-engineering/05-agent-harness/09-responses-api.ipynb"),
    Path("prompt-engineering/06-multi-agent/03-cross-model-review.ipynb"),
    Path("prompt-engineering/07-calibration-eval/01-rag-evaluation.ipynb"),
    Path("prompt-engineering/07-calibration-eval/02-feedback-loop.ipynb"),
]


def _notebook_text(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
    )


def test_modern_openai_notebooks_use_responses_api_as_main_entrypoint():
    for notebook in MODERN_OPENAI_NOTEBOOKS:
        text = _notebook_text(notebook)

        assert "client.responses.create" in text, notebook
        assert "client.chat.completions" not in text, notebook
        assert "OPENAI_MODEL" in text, notebook


def test_modern_openai_notebooks_do_not_teach_legacy_response_format():
    for notebook in MODERN_OPENAI_NOTEBOOKS:
        text = _notebook_text(notebook)

        assert "response_format" not in text, notebook


def test_responses_tool_calling_notebook_uses_current_tool_items():
    text = _notebook_text(
        Path("prompt-engineering/03-structured-output/02-function-calling-basics.ipynb")
    )

    assert '"type": "function_call_output"' in text
    assert 'item.type == "function_call"' in text
    assert '"type": "function",' in text
    assert '"function": {' not in text
    assert '"role": "tool"' not in text


def test_modern_openai_notebooks_do_not_pin_legacy_4o_models():
    for notebook in MODERN_OPENAI_NOTEBOOKS:
        text = _notebook_text(notebook)

        assert 'model="gpt-4o"' not in text, notebook
        assert 'model="gpt-4o-mini"' not in text, notebook
        assert '"model": "gpt-4o"' not in text, notebook
        assert '"model": "gpt-4o-mini"' not in text, notebook
