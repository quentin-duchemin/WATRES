# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'WATRES'
copyright = '2024, Q.Duchemin and P.Benettin'
author = 'Q.Duchemin and P.Benettin'

# The full version, including alpha/beta/rc tags
release = '1'

import sys
from pathlib import Path



HERE = Path(__file__).parent
sys.path[:0] = [str(HERE.parent.parent), str(HERE.parent), str(HERE / "extensions")]


import WATRES

HERE = Path(__file__).parent
sys.path[:0] = [str(HERE.parent), str(HERE / "extensions")]

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.githubpages",
    "sphinx.ext.viewcode",
    "nbsphinx",
    "nbsphinx_link",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",  # needs to be after napoleon
    "sphinx.ext.autosummary",
    "scanpydoc.elegant_typehints",
    "scanpydoc.definition_list_typed_field",
       'sphinx.ext.autosummary',
#    "scanpydoc.autosummary_generate_imported",
    *[p.stem for p in (HERE / "extensions").glob("*.py")],
    "sphinx_copybutton",
    "sphinx_gallery.load_style",
    "sphinx_tabs.tabs",
]



# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
source_suffix = ['.rst','.md']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "**.ipynb_checkpoints"]



# The master toctree document.
master_doc = "index"

intersphinx_mapping = dict(
    anndata=("https://anndata.readthedocs.io/en/stable/", None),
    ipython=("https://ipython.readthedocs.io/en/stable/", None),
    matplotlib=("https://matplotlib.org/", None),
    numpy=("https://numpy.org/doc/stable/", None),
    pandas=("https://pandas.pydata.org/docs/", None),
    python=("https://docs.python.org/3", None),
    scipy=("https://docs.scipy.org/doc/scipy/reference/", None),
    sklearn=("https://scikit-learn.org/stable/", None),
    torch= ('https://pytorch.org/docs/stable/', None),
    pytorch_lightning=("https://pytorch-lightning.readthedocs.io/en/stable/", None),
    pyro=("http://docs.pyro.ai/en/stable/", None),
)

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_show_sourcelink = True
html_theme = "pydata_sphinx_theme"

html_theme_options = {
    # Toc options
    'collapse_navigation': True
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = ["css/override.css", "css/sphinx_gallery.css", "css/custom_logo.css"]
html_title = "WATRES"
html_logo = "_static/logo.png"

nbsphinx_thumbnails = {
    "tutorials/notebooks/generating-simulated-data": "_static/tutorials/simulated_data.png",
    "tutorials/notebooks/visualizing-results": "_static/tutorials/nb_visualization.png",
    "tutorials/notebooks/running-the-model": "_static/tutorials/quickstart-on-simulated-data.png",
    "tutorials/notebooks/overview": "_static/watres_readme.png",
}


autosummary_generate = True
autodoc_member_order = "bysource"
bibtex_reference_style = "author_year"
napoleon_google_docstring = True  # for pytorch lightning
napoleon_numpy_docstring = True  # use numpydoc style
napoleon_include_init_with_doc = False
napoleon_use_rtype = True  # having a separate entry generally helps readability
napoleon_use_param = True
napoleon_custom_sections = [("Params", "Parameters")]
todo_include_todos = False
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
    "html_admonition",
]
myst_url_schemes = ("http", "https", "mailto")
nb_output_stderr = "remove"
nb_execution_mode = "off"
nbsphinx_execute = 'never'

nb_merge_streams = True
typehints_defaults = "braces"



nbsphinx_allow_errors = True
