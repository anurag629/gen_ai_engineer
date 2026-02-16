#!/usr/bin/env python3
"""Convert README.md books and .py files to Jupyter notebooks with Colab badges."""

import json
import re
import os

BASE_DIR = "/Users/cops/MY/learn_something/gen_ai_engineer"
GITHUB_RAW = "https://raw.githubusercontent.com/anurag629/gen_ai_engineer/main"
COLAB_BASE = "https://colab.research.google.com/github/anurag629/gen_ai_engineer/blob/main"

# ── Notebook helpers ──────────────────────────────────────────────────

def nb_metadata():
    return {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0",
            "mimetype": "text/x-python",
            "file_extension": ".py",
            "codemirror_mode": {"name": "ipython", "version": 3},
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3"
        },
        "colab": {
            "provenance": [],
            "toc_visible": True
        }
    }


def to_source_lines(text):
    """Convert a string to a list of lines ending with \\n (except last)."""
    lines = text.split('\n')
    result = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            result.append(line + '\n')
        else:
            result.append(line)
    return result


def md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": to_source_lines(text)
    }


def code_cell(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": to_source_lines(text)
    }


def write_notebook(cells, path):
    nb = {
        "cells": cells,
        "metadata": nb_metadata(),
        "nbformat": 4,
        "nbformat_minor": 5
    }
    with open(path, 'w') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"  Created: {path}")


# ── README.md → Notebook ─────────────────────────────────────────────

def parse_readme(content):
    """Parse markdown into list of ('markdown', text) or ('code', text) tuples."""
    lines = content.split('\n')
    chunks = []
    buffer = []
    in_code = False
    is_python = False

    for line in lines:
        stripped = line.strip()

        if not in_code:
            if stripped.startswith('```'):
                lang = stripped[3:].strip().lower()
                if lang in ('python', 'py', 'python3'):
                    # Flush markdown buffer
                    md = '\n'.join(buffer).strip()
                    if md:
                        chunks.append(('markdown', md))
                    buffer = []
                    in_code = True
                    is_python = True
                else:
                    # Non-python fence: keep as markdown
                    buffer.append(line)
                    in_code = True
                    is_python = False
            else:
                buffer.append(line)
        else:
            if stripped == '```':
                if is_python:
                    code_text = '\n'.join(buffer).strip()
                    if code_text:
                        chunks.append(('code', code_text))
                    buffer = []
                else:
                    buffer.append(line)
                in_code = False
                is_python = False
            else:
                buffer.append(line)

    remaining = '\n'.join(buffer).strip()
    if remaining:
        chunks.append(('markdown', remaining))

    return chunks


def readme_to_notebook(readme_path, output_path, day_rel_dir, deps=None, extra_setup=None):
    """Convert a README.md book to a Jupyter notebook."""
    with open(readme_path, 'r') as f:
        content = f.read()

    # Replace relative image paths with GitHub raw URLs
    raw_url = f"{GITHUB_RAW}/{day_rel_dir}"
    content = re.sub(
        r'!\[([^\]]*)\]\((?!http)([^)]+)\)',
        lambda m: f'![{m.group(1)}]({raw_url}/{m.group(2)})',
        content
    )

    chunks = parse_readme(content)
    cells = []

    # Colab badge
    nb_rel = output_path.replace(BASE_DIR + '/', '')
    colab_url = f"{COLAB_BASE}/{nb_rel}"
    badge = f'[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab_url})\n\n---'
    cells.append(md_cell(badge))

    # Install dependencies cell
    if deps:
        install_code = "# Run this cell first to install dependencies\n"
        install_code += f"!pip install -q {' '.join(deps)}"
        cells.append(code_cell(install_code))

    # Extra setup (e.g., download files)
    if extra_setup:
        cells.append(code_cell(extra_setup))

    # Convert chunks to cells
    for kind, text in chunks:
        if not text.strip():
            continue
        if kind == 'markdown':
            # Split very large markdown blocks at ## headings for better navigation
            sections = re.split(r'\n(?=## )', text)
            for section in sections:
                s = section.strip()
                if s:
                    cells.append(md_cell(s))
        else:
            cells.append(code_cell(text))

    write_notebook(cells, output_path)


# ── .py → Notebook ───────────────────────────────────────────────────

def py_to_notebook(py_path, output_path, deps=None, extra_setup=None):
    """Convert a .py file to a Jupyter notebook."""
    with open(py_path, 'r') as f:
        content = f.read()

    cells = []

    # Colab badge
    nb_rel = output_path.replace(BASE_DIR + '/', '')
    colab_url = f"{COLAB_BASE}/{nb_rel}"
    badge = f'[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab_url})\n\n---'
    cells.append(md_cell(badge))

    # Install dependencies
    if deps:
        install_code = "# Run this cell first to install dependencies\n"
        install_code += f"!pip install -q {' '.join(deps)}"
        cells.append(code_cell(install_code))

    # Extra setup
    if extra_setup:
        cells.append(code_cell(extra_setup))

    # Extract module docstring as title markdown
    docstring_match = re.match(r'^"""(.*?)"""', content, re.DOTALL)
    if docstring_match:
        docstring = docstring_match.group(1).strip()
        # Clean up Run: and Requires: lines
        doc_lines = []
        for line in docstring.split('\n'):
            if line.strip().startswith(('Run:', 'Requires:', 'Download:')):
                continue
            doc_lines.append(line)
        title_text = '\n'.join(doc_lines).strip()
        if title_text:
            # Make first line a heading if it isn't
            first_line = title_text.split('\n')[0]
            if not first_line.startswith('#'):
                title_text = f"# {first_line}\n" + '\n'.join(title_text.split('\n')[1:])
            cells.append(md_cell(title_text))
        # Remove docstring from content
        content = content[docstring_match.end():].strip()

    # Split by section markers: # ====... or # ----...
    section_pattern = r'\n# [=\-]{20,}\n# (.+?)\n# [=\-]{20,}\n'
    parts = re.split(section_pattern, content)

    # parts[0] = code before first section
    # parts[1] = section 1 title, parts[2] = section 1 code
    # parts[3] = section 2 title, parts[4] = section 2 code, ...

    # Handle pre-section code (imports, etc.)
    pre_code = parts[0].strip()
    if pre_code:
        # Fix __file__ references for Colab compatibility
        pre_code = re.sub(
            r'os\.path\.dirname\(os\.path\.abspath\(__file__\)\)',
            '"."',
            pre_code
        )
        cells.append(code_cell(pre_code))

    # Handle sections
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            section_title = parts[i].strip()
            section_code = parts[i + 1].strip()

            # Section header as markdown
            cells.append(md_cell(f"## {section_title}"))

            if section_code:
                # Split section code into logical blocks
                # If there's an `if __name__` guard, remove it and dedent
                section_code = re.sub(
                    r"if __name__ == ['\"]__main__['\"]:\n",
                    "# Main execution\n",
                    section_code
                )
                # Fix __file__ references for Colab compatibility
                section_code = re.sub(
                    r'os\.path\.dirname\(os\.path\.abspath\(__file__\)\)',
                    '"."',
                    section_code
                )

                # Split at blank-line separated blocks for readability
                blocks = re.split(r'\n\n\n+', section_code)
                for block in blocks:
                    b = block.strip()
                    if b:
                        # Check if block starts with a comment that looks like a subsection
                        if b.startswith('# ---') or b.startswith('# ~~~'):
                            # Extract the comment as markdown, rest as code
                            lines = b.split('\n')
                            header_lines = []
                            code_lines = []
                            in_header = True
                            for line in lines:
                                if in_header and line.strip().startswith('#'):
                                    # Clean the comment marker
                                    cleaned = line.strip().lstrip('#').strip().strip('-').strip('~').strip()
                                    if cleaned:
                                        header_lines.append(cleaned)
                                else:
                                    in_header = False
                                    code_lines.append(line)
                            if header_lines:
                                cells.append(md_cell(f"### {'  '.join(header_lines)}"))
                            code_text = '\n'.join(code_lines).strip()
                            if code_text:
                                cells.append(code_cell(code_text))
                        else:
                            cells.append(code_cell(b))

    write_notebook(cells, output_path)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("Converting files to Jupyter notebooks...\n")

    # ── Day 1 ──
    d1 = f"{BASE_DIR}/week1/day01_neural_networks_backpropagation"
    d1_rel = "week1/day01_neural_networks_backpropagation"
    print("Day 1: Neural Networks & Backpropagation")

    readme_to_notebook(
        f"{d1}/README.md",
        f"{d1}/Day_01_Neural_Networks_and_Backpropagation.ipynb",
        d1_rel,
        deps=["torch", "matplotlib", "numpy", "scikit-learn"],
    )

    py_to_notebook(
        f"{d1}/micrograd.py",
        f"{d1}/micrograd.ipynb",
        deps=None,  # pure python
    )

    py_to_notebook(
        f"{d1}/exercises.py",
        f"{d1}/exercises.ipynb",
        deps=["torch", "matplotlib", "numpy", "scikit-learn"],
        extra_setup=(
            "# Download micrograd.py (needed for imports)\n"
            "import urllib.request, os\n"
            "if not os.path.exists('micrograd.py'):\n"
            "    urllib.request.urlretrieve(\n"
            f"        '{GITHUB_RAW}/{d1_rel}/micrograd.py',\n"
            "        'micrograd.py'\n"
            "    )\n"
            "    print('Downloaded micrograd.py')"
        ),
    )

    py_to_notebook(
        f"{d1}/visualizations.py",
        f"{d1}/visualizations.ipynb",
        deps=["matplotlib", "numpy"],
    )

    # ── Day 2 ──
    d2 = f"{BASE_DIR}/week1/day02_language_modeling_nlp"
    d2_rel = "week1/day02_language_modeling_nlp"
    print("\nDay 2: Language Modeling & NLP")

    names_setup = (
        "# Download names.txt dataset\n"
        "import urllib.request, os\n"
        "if not os.path.exists('names.txt'):\n"
        "    urllib.request.urlretrieve(\n"
        "        'https://raw.githubusercontent.com/karpathy/makemore/master/names.txt',\n"
        "        'names.txt'\n"
        "    )\n"
        "    print('Downloaded names.txt')"
    )

    readme_to_notebook(
        f"{d2}/README.md",
        f"{d2}/Day_02_Language_Modeling_and_NLP.ipynb",
        d2_rel,
        deps=["torch", "matplotlib"],
        extra_setup=names_setup,
    )

    py_to_notebook(
        f"{d2}/bigram.py",
        f"{d2}/bigram.ipynb",
        deps=["torch", "matplotlib"],
        extra_setup=names_setup,
    )

    py_to_notebook(
        f"{d2}/bigram_neural.py",
        f"{d2}/bigram_neural.ipynb",
        deps=["torch"],
        extra_setup=names_setup,
    )

    py_to_notebook(
        f"{d2}/mlp_lm.py",
        f"{d2}/mlp_lm.ipynb",
        deps=["torch", "matplotlib"],
        extra_setup=names_setup,
    )

    # ── Day 3 ──
    d3 = f"{BASE_DIR}/week1/day03_transformers"
    d3_rel = "week1/day03_transformers"
    print("\nDay 3: Transformers")

    readme_to_notebook(
        f"{d3}/README.md",
        f"{d3}/Day_03_Transformers.ipynb",
        d3_rel,
        deps=["torch", "matplotlib", "numpy"],
    )

    py_to_notebook(
        f"{d3}/visualizations.py",
        f"{d3}/visualizations.ipynb",
        deps=["matplotlib", "numpy"],
    )

    print("\n✓ All notebooks created successfully!")


if __name__ == "__main__":
    main()
