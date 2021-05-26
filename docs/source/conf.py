# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Workaround --------------------------------------------------------------
# Issue https://github.com/sphinx-contrib/googleanalytics/issues/2
# Note that a warning still will be issued "unsupported object from its setup() function"
# Remove this workaround when the issue has been resolved upstream
import sphinx.application
import sphinx.errors
sphinx.application.ExtensionError = sphinx.errors.ExtensionError

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
sys.path.insert(0, os.path.abspath('../../'))


# -- Project information -----------------------------------------------------

project = 'Sparse Tensor Classifier'
copyright = '2020, E Guidotti, A Ferrara'
author = 'E Guidotti, A Ferrara'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
	'sphinx.ext.autodoc',
	'sphinx.ext.autosummary',
	'sphinx.ext.githubpages',
	'sphinx_autodoc_typehints',
	'sphinxcontrib.googleanalytics',
	'sphinx_sitemap',
	'sphinxext.opengraph',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# Google Analytics
googleanalytics_id = 'UA-197829786-1'

# Open Graph
ogp_site_url = 'https://sparsetensorclassifier.org'
ogp_description_length = 0
ogp_custom_meta_tags = [
	'<meta property="og:description" content="An Explainable Probabilistic Classifier for Categorical Data Inspired to Quantum Physics" />',
	'<meta name="twitter:card" content="summary_large_image" />',
	'<meta property="og:image" content="https://sparsetensorclassifier.org/_static/img/logo-1280x640.png" />',
	'<meta property="og:image" content="https://sparsetensorclassifier.org/_static/img/logo-512x512.png" />',
]

# Sitemap
html_baseurl = 'https://sparsetensorclassifier.org'
sitemap_filename = "sitemap.xml"
html_extra_path = ['robots.txt']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'press'
html_favicon = 'favicon.png'
html_logo = '_static/img/logo.svg'
html_sidebars = {'**': ['util/sidetoc.html']}
html_theme_options = {
  "external_links": [
      ("Github", "https://github.com/SparseTensorClassifier")
  ]
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = [
    'css/custom.css',
]
