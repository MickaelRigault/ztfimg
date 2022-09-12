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

extensions = [# Standard extensions
    "numpydoc",
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.mathjax',       # or pngmath
    'sphinx.ext.viewcode',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.graphviz',
    'sphinx.ext.extlinks',
    # 'sphinx.ext.doctest',
    # 'sphinx.ext.coverage',
    # Other extensions  
]


# Class to another class
inheritance_node_attrs = dict(shape='ellipse', fontsize=13, height=0.75,
                              color='sienna', style='filled', imagepos='tc')

inheritance_graph_attrs = dict(rankdir="LR", size='""')


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



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

#html_permalinks_icon = '<span>#</span>'
#html_theme = 'sphinxawesome_theme'
