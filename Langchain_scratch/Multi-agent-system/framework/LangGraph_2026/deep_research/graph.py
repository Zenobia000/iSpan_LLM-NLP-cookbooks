"""LangGraph implementation inspired by LangChain's Open Deep Research."""

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph

from .configuration import ResearchConfig
from .prompts import PLANNER_PROMPT, RESEARCHER_PROMPT, WRITER_PROMPT
from .state import ResearchFinding, ResearchPlan, ResearchState, ResearcherState
from .tools import format_sources, search_web


def _model(model_name: str):
    """Create a LangChain-native chat model."""
    return init_chat_model(model_name)


def plan_research(state: ResearchState, config: RunnableConfig | None = None):
    """Convert the user topic into a structured research brief and sub-tasks."""
    cfg = ResearchConfig.from_runnable_config(config)
    topic = state.get("topic") or state["messages"][-1].content
    planner = _model(cfg.planner_model).with_structured_output(ResearchPlan)
    plan = planner.invoke([HumanMessage(content=PLANNER_PROMPT.format(topic=topic))])
    units = plan.units[: cfg.max_research_units]
    return {
        "topic": topic,
        "research_brief": plan.research_brief,
        "research_units": units,
    }


def fan_out_research(state: ResearchState):
    """Run one researcher subgraph per planned research unit."""
    return [
        Send("researcher", {"unit": unit})
        for unit in state.get("research_units", [])
    ]


def researcher(state: ResearcherState, config: RunnableConfig | None = None):
    """Search and compress one research unit."""
    cfg = ResearchConfig.from_runnable_config(config)
    unit = state["unit"]
    sources = search_web(
        unit.topic,
        max_results=cfg.search_results_per_query,
        offline=cfg.offline_search,
    )
    prompt = RESEARCHER_PROMPT.format(
        unit_topic=unit.topic,
        rationale=unit.rationale,
        evidence=format_sources(sources),
    )
    response = _model(cfg.researcher_model).invoke([HumanMessage(content=prompt)])
    finding = ResearchFinding(
        topic=unit.topic,
        summary=str(response.content),
        sources=sources,
    )
    return {"findings": [finding]}


def write_report(state: ResearchState, config: RunnableConfig | None = None):
    """Synthesize all findings into the final report."""
    cfg = ResearchConfig.from_runnable_config(config)
    findings = "\n\n".join(
        f"## {finding.topic}\n{finding.summary}\n"
        + "\n".join(f"- {source.title}: {source.url}" for source in finding.sources)
        for finding in state.get("findings", [])
    )
    prompt = WRITER_PROMPT.format(
        research_brief=state.get("research_brief", ""),
        findings=findings,
    )
    response = _model(cfg.writer_model).invoke([HumanMessage(content=prompt)])
    return {
        "final_report": str(response.content),
        "messages": [AIMessage(content=str(response.content))],
    }


def build_graph():
    """Build the LangGraph workflow."""
    builder = StateGraph(ResearchState)
    builder.add_node("plan_research", plan_research)
    builder.add_node("researcher", researcher)
    builder.add_node("write_report", write_report)

    builder.add_edge(START, "plan_research")
    builder.add_conditional_edges("plan_research", fan_out_research, ["researcher"])
    builder.add_edge("researcher", "write_report")
    builder.add_edge("write_report", END)
    return builder.compile()


graph = build_graph()
