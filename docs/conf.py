# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html



# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'ztfimg'
copyright = '2022, Mickael Rigault'
author = 'Mickael Rigault'





import os
import sys

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../..'))

for x in os.walk(f'../{project}'):
  sys.path.insert(0, x[0])


from ztfimg import *



# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon', # numpy or google docstring
#    'sphinx.ext.viewcode',
    'matplotlib.sphinxext.plot_directive', # matplotlib plotting in the doc
    # extra
    "numpydoc",
#    'myst_nb',
    "nbsphinx",
    'sphinx_copybutton'
    ]
    

# Class to another class
inheritance_node_attrs = dict(shape='ellipse', fontsize=13, height=0.75,
                              color='sienna', style='filled', imagepos='tc')

inheritance_graph_attrs = dict(rankdir="LR", size='""')


nbsphinx_execute = 'never'


autoclass_content = "both"              # Insert class and __init__ docstrings
autodoc_member_order = "bysource"

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
    'matplotlib': ('https://matplotlib.org/', None),
    #'emcee': ('https://emcee.readthedocs.io/en/latest', None),
}

    
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# = from jax docs =
# The suffix(es) of source filenames.
# Note: important to list ipynb before md here: we have both md and ipynb
# copies of each notebook, and myst will choose which to convert based on
# the order in the source_suffix list. Notebooks which are not executed have
# outputs stored in ipynb but not in md, so we must convert the ipynb.
#source_suffix = ['.rst', '.ipynb', '.md']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'alabaster'
html_theme = 'sphinx_book_theme'
html_theme_options = {
    'show_toc_level': 2,
    'repository_url': 'https://github.com/MickaelRigault/ztfimg',
    'use_repository_button': True,     # add a "link to repository" button
}
html_static_path = ['_static']

#html_permalinks_icon = '<span>#</span>'
#html_theme = 'sphinxawesome_theme'
