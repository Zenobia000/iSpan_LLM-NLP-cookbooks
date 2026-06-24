"""State and structured-output schemas for the LangGraph research workflow."""

from __future__ import annotations

import operator
from typing import Annotated

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class ResearchUnit(BaseModel):
    """One focused research assignment."""

    topic: str = Field(description="A standalone research task.")
    rationale: str = Field(description="Why this task is needed for the final report.")


class ResearchPlan(BaseModel):
    """Structured plan generated from the user request."""

    research_brief: str = Field(description="A precise research brief in the user's language.")
    units: list[ResearchUnit] = Field(description="Independent research units to investigate.")


class Source(BaseModel):
    """A source found by the search layer."""

    title: str
    url: str
    snippet: str


class ResearchFinding(BaseModel):
    """Compressed finding from one researcher."""

    topic: str
    summary: str
    sources: list[Source] = Field(default_factory=list)


class ResearchState(MessagesState):
    """Main graph state."""

    topic: str
    research_brief: str
    research_units: list[ResearchUnit]
    findings: Annotated[list[ResearchFinding], operator.add]
    final_report: str


class ResearcherState(TypedDict):
    """Subgraph state for one research unit."""

    unit: ResearchUnit
    finding: ResearchFinding
