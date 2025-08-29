import os
import sys

# -- Path setup ---------------------------------------------------------------
sys.path.insert(0, os.path.abspath(".."))


# -- Project information -----------------------------------------------------
project = "causal-falsify"
author = "Rickard Karlsson"
release = "0.2.0"
copyright = "2025, Rickard Karlsson"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",  # Pull in docstrings
    "sphinx.ext.napoleon",  # Google/NumPy style
    "sphinx.ext.autosummary",  # Generate summary tables
    "sphinx.ext.viewcode",  # Link to source code
    "sphinx.ext.todo",  # TODOs
    "sphinx_autodoc_typehints",  # Type hints
]

autosummary_generate = True
templates_path = ["_templates"]
exclude_patterns = []

# -- HTML output -------------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_theme_options = {
    "collapse_navigation": False,  # Expand full sidebar
    "navigation_depth": 4,  # Show submodules
    "titles_only": False,
}


# -- Automatically run sphinx-apidoc at build time -------------------------------
import subprocess


def run_apidoc(app):
    here = os.path.dirname(__file__)
    package_dir = os.path.abspath(os.path.join(here, "..", "..", "causal_falsify"))
    output_dir = os.path.join(here, "api")
    # Clear and regenerate
    subprocess.call(
        [
            "sphinx-apidoc",
            "-o",
            output_dir,
            "--separate",
            "--module-first",
            package_dir,
            "--force",
        ]
    )


def setup(app):
    app.connect("builder-inited", run_apidoc)
