"""Command-line runner for the LangGraph deep-research teaching workflow."""

from __future__ import annotations

import argparse

from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

from deep_research.graph import graph


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("topic", help="Research topic or question.")
    parser.add_argument("--model", default=None, help="Override all model roles.")
    parser.add_argument("--offline-search", action="store_true", help="Do not call Tavily.")
    args = parser.parse_args()

    configurable = {"offline_search": args.offline_search}
    if args.model:
        configurable.update(
            {
                "planner_model": args.model,
                "researcher_model": args.model,
                "writer_model": args.model,
            }
        )

    result = graph.invoke(
        {"messages": [HumanMessage(content=args.topic)], "topic": args.topic},
        config={"configurable": configurable},
    )
    print(result["final_report"])


if __name__ == "__main__":
    main()
