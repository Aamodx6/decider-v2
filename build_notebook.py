"""Rebuild decider-train/sagemaker_train.ipynb from the per-cell sources in _nbcells/.

Cell files are named  NN-name.(md|.py): the numeric prefix is execution order, the suffix the
nbformat cell_type. The committed .ipynb must remain byte-stable for a given _nbcells/ tree, so
this builder is deliberately trivial (a sort + a fold): edit the cell file, re-run, git-diff.

    python build_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parent
SRC_DIR = ROOT / "_nbcells"
OUT = ROOT / "decider-train" / "sagemaker_train.ipynb"


def _id(path, seen):
    """nbformat >= 4.5 requires a unique cell id; the file stem is already unique."""
    raw = path.stem.replace("-", "-").replace("_", "")
    cid = raw[:32]
    while cid in seen:
        cid += "-"
    seen.add(cid)
    return cid


def build():
    cells = []
    seen = set()
    for path in sorted(SRC_DIR.iterdir(), key=lambda p: p.name):
        if path.suffix == ".md":
            cells.append({"cell_type": "markdown", "id": _id(path, seen), "metadata": {},
                          "source": path.read_text(encoding="utf-8").splitlines(keepends=True)})
        elif path.suffix == ".py":
            cells.append({"cell_type": "code", "id": _id(path, seen), "metadata": {},
                          "execution_count": None, "outputs": [],
                          "source": path.read_text(encoding="utf-8").splitlines(keepends=True)})
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.14"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    # newline="\n" keeps the committed notebook byte-stable across platforms (on Windows the
    # default would turn every line into CRLF).
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8",
                   newline="\n")
    print(f"wrote {OUT} ({len(cells)} cells)")


if __name__ == "__main__":
    build()
