from __future__ import annotations

from importlib.metadata import version
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

project = "quantitativeactuarial"
author = "Quantitative Actuarial contributors"
copyright = "2026, Quantitative Actuarial contributors"

try:
    release = version("quantitativeactuarial")
except Exception:
    release = "0.1.0"

version = release

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = "quantitativeactuarial"
html_static_path = ["_static"]

html_theme_options = {
    "source_repository": "https://github.com/Walls16/Quantitative-Actuarial/",
    "source_branch": "main",
    "source_directory": "docs/",
}

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
]
