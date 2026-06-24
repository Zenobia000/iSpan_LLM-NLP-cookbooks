"""Course-wide helpers shared by every notebook.

Goal: keep example code provider-agnostic. Students set ONE environment
variable (`COURSE_MODEL`) and every notebook works against OpenAI, Anthropic,
a local Ollama model, or anything else `init_chat_model` understands.

Usage inside a notebook
-----------------------
    import sys, pathlib
    sys.path.append(str(pathlib.Path.cwd().parents[1] / "_shared"))  # reach _shared/
    from course_utils import get_model, get_embeddings, load_env

    load_env()                 # read the repo-root .env
    model = get_model()        # provider-agnostic chat model
"""
from __future__ import annotations

import os
from pathlib import Path

# Sensible defaults. Override via .env without touching notebook code.
DEFAULT_MODEL = "openai:gpt-4o-mini"
DEFAULT_EMBEDDINGS = "openai:text-embedding-3-small"


def load_env() -> None:
    """Load a `.env` from the curriculum root if python-dotenv is installed.

    Searches upward from the current working directory so it works no matter
    which module folder a notebook is launched from.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:  # dotenv is optional; env vars may be set another way
        return
    here = Path.cwd()
    for parent in [here, *here.parents]:
        candidate = parent / ".env"
        if candidate.exists():
            load_dotenv(candidate)
            return
    load_dotenv()  # fall back to default search


def get_model(model: str | None = None, **kwargs):
    """Return a chat model via the v1 provider-agnostic factory.

    `model` accepts the `provider:name` form, e.g. "anthropic:claude-haiku-4-5"
    or "openai:gpt-4o-mini". When omitted, reads $COURSE_MODEL, else DEFAULT_MODEL.
    """
    from langchain.chat_models import init_chat_model

    name = model or os.environ.get("COURSE_MODEL", DEFAULT_MODEL)
    return init_chat_model(name, **kwargs)


def get_embeddings(model: str | None = None, **kwargs):
    """Return an embeddings model via the v1 provider-agnostic factory."""
    from langchain.embeddings import init_embeddings

    name = model or os.environ.get("COURSE_EMBEDDINGS", DEFAULT_EMBEDDINGS)
    return init_embeddings(name, **kwargs)
