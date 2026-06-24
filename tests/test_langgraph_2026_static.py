"""Static guardrails for the 2026 LangGraph multi-agent implementation."""

from pathlib import Path


ROOT = (
    Path(__file__).resolve().parent.parent
    / "Langchain_scratch"
    / "Multi-agent-system"
    / "framework"
    / "LangGraph_2026"
)

LONGFORM_ROOT = (
    Path(__file__).resolve().parent.parent
    / "Langchain_scratch"
    / "Multi-agent-system"
    / "應用專案-多智能體長文寫作"
)


def test_langgraph_2026_uses_langgraph_and_universal_model_init():
    graph_py = (ROOT / "deep_research" / "graph.py").read_text(encoding="utf-8")

    assert "init_chat_model" in graph_py
    assert "StateGraph" in graph_py
    assert "Send(" in graph_py
    assert "ChatOpenAI" not in graph_py
    assert "CrewAI" not in graph_py


def test_langgraph_2026_documents_open_deep_research_reference():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "open_deep_research" in readme
    assert "plan_research" in readme
    assert "write_report" in readme


def test_longform_framework_notebooks_use_langchain_langgraph_not_crewai():
    framework_dir = LONGFORM_ROOT / "03_框架實作"
    notebooks = sorted(path.name for path in framework_dir.glob("*.ipynb"))

    assert notebooks == [
        "01_LangChain_LCEL_長文寫作基礎.ipynb",
        "02_LangGraph_STORM_長文寫作系統.ipynb",
        "03_延伸_OpenDeepResearch_長文研究.ipynb",
    ]

    combined = "\n".join(
        path.read_text(encoding="utf-8")
        for path in framework_dir.glob("*.ipynb")
    )

    assert "crewai" not in combined.lower()
    assert "ChatOpenAI" not in combined
    assert "gpt-3.5-turbo" not in combined
    assert "init_chat_model" in combined
    assert "StateGraph" in combined
    assert "deep_research.graph" in combined


def test_longform_notebooks_do_not_use_legacy_model_entrypoints():
    combined = "\n".join(
        path.read_text(encoding="utf-8")
        for path in LONGFORM_ROOT.rglob("*.ipynb")
    )

    assert "from crewai" not in combined
    assert "import crewai" not in combined
    assert "ChatOpenAI" not in combined
    assert "gpt-3.5-turbo" not in combined
    assert "langchain_openai" not in combined
