"""Runtime configuration for the teaching deep-research graph."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from langchain_core.runnables import RunnableConfig


@dataclass(frozen=True)
class ResearchConfig:
    """Small config surface aligned with LangChain v1 `init_chat_model`."""

    planner_model: str = "openai:gpt-4o-mini"
    researcher_model: str = "openai:gpt-4o-mini"
    writer_model: str = "openai:gpt-4o-mini"
    max_research_units: int = 3
    search_results_per_query: int = 4
    offline_search: bool = False

    @classmethod
    def from_runnable_config(cls, config: RunnableConfig | None = None) -> "ResearchConfig":
        """Read LangGraph config values, falling back to environment variables."""
        configurable: dict[str, Any] = {}
        if config:
            configurable = dict(config.get("configurable", {}) or {})

        def get(name: str, env: str, default: Any) -> Any:
            return configurable.get(name, os.environ.get(env, default))

        return cls(
            planner_model=str(get("planner_model", "COURSE_MODEL", cls.planner_model)),
            researcher_model=str(get("researcher_model", "COURSE_MODEL", cls.researcher_model)),
            writer_model=str(get("writer_model", "COURSE_MODEL", cls.writer_model)),
            max_research_units=int(get("max_research_units", "MAX_RESEARCH_UNITS", cls.max_research_units)),
            search_results_per_query=int(
                get("search_results_per_query", "SEARCH_RESULTS_PER_QUERY", cls.search_results_per_query)
            ),
            offline_search=str(get("offline_search", "OFFLINE_SEARCH", "false")).lower()
            in {"1", "true", "yes"},
        )
