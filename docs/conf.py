# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'sika'
copyright = '%Y, Sage Santomenna'
author = ''
# author = 'Sage Santomenna'
release = '0.0.1'
root_doc = 'index'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_automodapi.automodapi',
    # 'nbsphinx'
]


modindex_common_prefix = ["sika"]
highlight_language = 'python'

numpydoc_show_class_members = False

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__, __call__, _call'
}

autodoc_typehints = "both"
# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']

html_title = "sika documentation"

html_theme_options = {
    "repository_url": "https://github.com/sagesanto/sika",
    "use_repository_button": True,
}




def resolve_type_aliases(app, env, node, contnode):
    """Resolve :class: references to our type aliases as :attr: instead."""

    # A sphinx bug means that TypeVar doesn't work:
    # https://github.com/sphinx-doc/sphinx/issues/10785

    if (
        node["refdomain"] == "py"
        and node["reftype"] == "class"
        and "sika_typing" in node["reftarget"]
    ):
        print(node.__dict__)
        print("meets criteria!")
        resolved = app.env.get_domain("py").resolve_xref(
            env, node["refdoc"], app.builder, "attr", node["reftarget"], node, contnode
        )
        print(resolved)
        print()
        return resolved


def setup(app):
    app.connect("missing-reference", resolve_type_aliases)