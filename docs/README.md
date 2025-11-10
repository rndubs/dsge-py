# DSGE Documentation

This directory contains the documentation for the DSGE project, built using [Sphinx](https://www.sphinx-doc.org/) with [MyST](https://myst-parser.readthedocs.io/) for Markdown support.

## Building the Documentation

### Prerequisites

All Python dependencies are managed by `uv` and should be installed as dev dependencies:

```bash
uv sync
```

For PDF generation, you'll also need LaTeX installed on your system:

- **Ubuntu/Debian**: `sudo apt-get install texlive-latex-base texlive-latex-extra texlive-fonts-recommended latexmk`
- **macOS**: Install [MacTeX](https://www.tug.org/mactex/)
- **Windows**: Install [MiKTeX](https://miktex.org/)

### Build HTML Documentation

To build the HTML documentation:

```bash
cd docs
uv run sphinx-build -b html . _build/html
```

Or using the Makefile:

```bash
cd docs
make html
```

The HTML output will be in `_build/html/`. Open `_build/html/index.html` in your browser to view the documentation.

### Build PDF Documentation

To build the PDF documentation:

```bash
cd docs
uv run sphinx-build -b latex . _build/latex
cd _build/latex
make
```

Or using the Makefile:

```bash
cd docs
make latexpdf
```

The PDF output will be in `_build/latex/dsge.pdf`.

### Clean Build Artifacts

To clean all build artifacts:

```bash
cd docs
make clean
```

## Documentation Structure

- `index.md` - Main documentation page
- `api.md` - API reference documentation (auto-generated from docstrings)
- `references.bib` - Bibliography file for citations
- `conf.py` - Sphinx configuration
- `_static/` - Static assets (images, CSS, etc.)
- `_build/` - Build output directory (git-ignored)

## Writing Documentation

All documentation is written in Markdown using the MyST parser. MyST extends CommonMark with Sphinx-specific features:

- Use standard Markdown syntax for most content
- Use MyST directives with ` ```{directive}` syntax
- Math equations use `$...$` for inline and `$$...$$` for display math
- Cross-references use `{ref}` syntax

See the [MyST documentation](https://myst-parser.readthedocs.io/) for more details.
