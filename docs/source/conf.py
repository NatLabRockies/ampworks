# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://sphinx-doc.org/en/master/usage/configuration.html#project-information

import ampworks as amp

project = 'ampworks'
author = 'Corey R. Randall'
copyright = 'Alliance for Energy Innovation, LLC'

version = amp.__version__
release = amp.__version__

json_url = 'https://ampworks.readthedocs.io/en/latest/_static/switcher.json'


# -- General configuration ---------------------------------------------------
# https://sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.todo',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'myst_nb',
    'sphinx_design',
    'sphinx_copybutton',
    'autoapi.extension',
]

templates_path = ['_templates']

exclude_patterns = [
    'build',
    'Thumbs.db',
    '.DS_Store',
    '__pycache__',
    '*.ipynb_checkpoints',
]

source_suffix = {
    '.myst': 'myst-nb',
    '.ipynb': 'myst-nb',
    '.rst': 'restructuredtext',
}

default_role = 'literal'  # allow single backticks for inline code refs
highlight_language = 'console'


# -- Options for HTML output -------------------------------------------------
# https://sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
# https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/layout.html

html_theme = 'pydata_sphinx_theme'

html_context = {'default_mode': 'dark'}

html_static_path = ['_static']

html_js_files = ['custom.js']
html_css_files = ['custom.css']

html_sidebars = {'index': [], '**': ['sidebar-nav-bs']}

html_theme_options = {
    'icon_links': [
        {
            'name': 'GitHub',
            'url': 'https://github.com/NatLabRockies/ampworks',
            'icon': 'fa-brands fa-github',
        },
        {
            'name': 'PyPI',
            'url': 'https://pypi.org/project/ampworks',
            'icon': 'fa-solid fa-box',
        },
    ],
    'navbar_start': ['navbar-logo'],
    'navbar_align': 'content',
    'header_links_before_dropdown': 5,
    'footer_start': ['copyright'],
    'footer_end': ['sphinx-version'],
    'navbar_persistent': ['search-button-field'],
    'primary_sidebar_end': ['sidebar-ethical-ads'],
    'secondary_sidebar_items': ['page-toc'],
    'search_bar_text': 'Search...',
    'show_prev_next': False,
    'collapse_navigation': True,
    'show_toc_level': 0,
    'pygments_light_style': 'tango',
    'show_version_warning_banner': True,
    'switcher': {
        'json_url': json_url,
        'version_match': version,
    }
}

# -- Options for napoleon ----------------------------------------------------
# https://sphinx-doc.org/en/master/usage/extensions/napoleon.html

napoleon_use_ivar = True
napoleon_use_rtype = False
napoleon_numpy_docstring = True
napoleon_custom_sections = [
    'TODO',
    'Summary',
    'Accessing the documentation',
]


# -- Options for autoapi -----------------------------------------------------
# https://sphinx-autoapi.readthedocs.io/en/latest/reference/config.html

autoapi_root = 'api'
autoapi_type = 'python'
autoapi_keep_files = True
autodoc_typehints = 'none'
autoapi_member_order = 'groupwise'
autoapi_python_class_content = 'both'
autoapi_dirs = ['../../src/ampworks']
autoapi_options = [
    'members',
    'imported-members',
    'inherited-members',
    'show-module-summary',
]


# -- Options for myst --------------------------------------------------------
# https://myst-nb.readthedocs.io/en/latest/configuration.html

nb_execution_timeout = 300
nb_number_source_lines = True
myst_enable_extensions = ['amsmath', 'dollarmath']


# -- Custom options ----------------------------------------------------------
# The Dataset class inherits from pd.DataFrame, but it is not worth it to keep
# all of the inherited docstrings. Rather, the user should be pointed to the
# pandas documentation for this. The following provides a list of which methods
# to keep in the ampworks documentation. The rest is excluded.

dataset_keep = [
    'zero_below',
    'downsample',
    'interactive_xy_plot',
]

richres_keep = ['copy']
progbar_keep = ['set_progress', 'reset', 'close']


def skip_util_classes(app, what, name, obj, skip, options):
    
    what_kind = ['method', 'attribute', 'property']
                    
    if what in what_kind and '.Dataset.' in name:  # no DataFrame methods
        if name.split('.')[-1] not in dataset_keep:
            skip = True
            
    elif what in what_kind and '.RichResult.' in name:  # no dict methods
        if name.split('.')[-1] not in richres_keep:
            skip = True
            
    elif what in what_kind and '.ProgressBar.' in name:  # no tqdm methods
        if name.split('.')[-1] not in progbar_keep:
            skip = True
            
    return skip
            

def setup(sphinx):
   sphinx.connect('autoapi-skip-member', skip_util_classes)

