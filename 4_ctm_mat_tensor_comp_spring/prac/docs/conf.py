"""Sphinx configuration for the matcomp documentation."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

project = "matcomp"
author = "Yegor Podlesov"
copyright = "2026, Yegor Podlesov"  # noqa: A001 - sphinx convention
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "numpydoc",
    "sphinx_autodoc_typehints",
]

# numpydoc / napoleon settings tuned for clean, NumPy-style API pages.
numpydoc_show_class_members = False
numpydoc_show_inherited_class_members = False
napoleon_google_docstring = False
napoleon_numpy_docstring = True

# Show types from annotations rather than from docstrings.
autodoc_typehints = "description"
autodoc_member_order = "bysource"

# MathJax v3 — used for rendering LaTeX in docstrings.
mathjax3_config = {
    "tex": {
        "inlineMath": [["\\(", "\\)"], ["$", "$"]],
        "displayMath": [["\\[", "\\]"], ["$$", "$$"]],
    },
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

html_theme = "furo"
html_title = "matcomp"
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
