#!/usr/bin/env python3
"""Percent-format .py -> Jupyter .ipynb converter (no nbformat dependency).

Cell markers (jupytext "percent" format):
    # %% [markdown]   -> a markdown cell (body lines are comments `# ...`)
    # %%              -> a code cell
    # %% [raw]        -> a raw cell

Usage:
    python build_notebooks.py <root_dir>
        Convert every `lab.py` found under <root_dir> into a sibling `lab.ipynb`.
    python build_notebooks.py <file.py> [out.ipynb]
        Convert a single file.

The emitted JSON conforms to nbformat 4.5 (cells carry stable string ids).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _strip_md(lines: list[str]) -> list[str]:
    """Markdown cell bodies are written as `# text`; strip one leading '# '."""
    out = []
    for ln in lines:
        if ln.startswith("# "):
            out.append(ln[2:])
        elif ln.rstrip() == "#":
            out.append("")
        else:
            out.append(ln)
    return out


def _src(lines: list[str]) -> list[str]:
    """nbformat stores source as a list of lines, each ending in \n except last."""
    # Drop trailing blank lines for tidiness.
    while lines and lines[-1].strip() == "":
        lines.pop()
    if not lines:
        return []
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]]


def parse(text: str) -> list[dict]:
    cells: list[dict] = []
    cur_type = "code"
    cur: list[str] = []
    started = False

    def flush():
        nonlocal cur, cur_type
        if not started:
            return
        body = _strip_md(cur) if cur_type == "markdown" else cur
        src = _src(list(body))
        if cur_type == "code" and not src:
            cur = []
            return
        cells.append({"_type": cur_type, "_src": src})
        cur = []

    for raw in text.splitlines():
        stripped = raw.strip()
        if stripped.startswith("# %%"):
            flush()
            started = True
            tag = stripped[len("# %%"):].strip()
            if tag.startswith("[markdown]"):
                cur_type = "markdown"
            elif tag.startswith("[raw]"):
                cur_type = "raw"
            else:
                cur_type = "code"
            cur = []
        else:
            if started:
                cur.append(raw)
    flush()
    return cells


def to_nb(cells: list[dict]) -> dict:
    nb_cells = []
    for i, c in enumerate(cells):
        cell_id = f"cell-{i:03d}"
        if c["_type"] == "markdown":
            nb_cells.append({
                "cell_type": "markdown",
                "id": cell_id,
                "metadata": {},
                "source": c["_src"],
            })
        elif c["_type"] == "raw":
            nb_cells.append({
                "cell_type": "raw",
                "id": cell_id,
                "metadata": {},
                "source": c["_src"],
            })
        else:
            nb_cells.append({
                "cell_type": "code",
                "id": cell_id,
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": c["_src"],
            })
    return {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def convert_file(src: Path, dst: Path) -> int:
    cells = parse(src.read_text(encoding="utf-8"))
    nb = to_nb(cells)
    dst.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return len(cells)


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 1
    target = Path(argv[1])
    if target.is_dir():
        total = 0
        for py in sorted(target.rglob("lab.py")):
            n = convert_file(py, py.with_suffix(".ipynb"))
            print(f"  [ok] {py.relative_to(target)} -> lab.ipynb ({n} cells)")
            total += 1
        print(f"Converted {total} notebook(s).")
    else:
        out = Path(argv[2]) if len(argv) > 2 else target.with_suffix(".ipynb")
        n = convert_file(target, out)
        print(f"[ok] {target} -> {out} ({n} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
