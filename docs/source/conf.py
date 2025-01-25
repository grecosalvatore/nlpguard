# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'NLPGuard'
copyright = '2025, Salvatore Greco'
author = 'Salvatore Greco'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',  # Enables automatic documentation
    'sphinx_autodoc_typehints'  # Adds type hint documentation support
]

autodoc_default_options = {
    'members': True,  # Include members of classes and functions
    'undoc-members': True,  # Include members without docstrings
    'private-members': True,  # Include private members
}

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"

html_static_path = ['_static']
