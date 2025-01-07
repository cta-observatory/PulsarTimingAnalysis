# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import datetime
import os
import sys
from pathlib import Path

if sys.version_info < (3, 11):
    import tomli as tomllib
else:
    import tomllib


pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
pyproject = tomllib.loads(pyproject_path.read_text())

import ptiming_ana

project = pyproject["project"]["name"]
author = ", ".join([author["name"] for author in pyproject["project"]["authors"]])
copyright = f'{author}.  Last updated {datetime.datetime.now().strftime("%d %b %Y %H:%M")}'
python_requires = pyproject["project"]["requires-python"]

# make some variables available to each page
rst_epilog = f"""
.. |python_requires| replace:: {python_requires}
"""

version = ptiming_ana.__version__
# The full version, including alpha/beta/rc tags.
release = version


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = [
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "numpydoc",
    "sphinx_changelog",
    "sphinx_automodapi.automodapi",
    "sphinx_gallery.gen_gallery"
]

numpydoc_show_class_members = False
numpydoc_class_members_toctree = False

exclude_patterns = []

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
}

sphinx_gallery_conf = {
    'examples_dirs': '../tutorial',
    'gallery_dirs': 'auto_examples',
    'filename_pattern': r'.*\.py',
    'copyfile_regex': r'.*\.png',
    'promote_jupyter_magic': True,
    'line_numbers': True,
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'

html_title = "Pulsar timing analysis"

html_theme_options = {
    "github_url": "https://github.com/cta-observatory/PulsarTimingAnalysis",
    "navbar_end": ["navbar-icon-links", "theme-switcher"],
    "icon_links_label": "Quick Links",
}