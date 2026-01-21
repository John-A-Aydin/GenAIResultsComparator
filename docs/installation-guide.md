# Installation Guide

This guide provides comprehensive installation instructions for GAICo, including optional dependencies, Jupyter setup, and developer installation. _Use the table of contents in the left sidebar to navigate._

## Basic Installation

{{ readme("INSTALLATION_STANDARD_INTRO") }}

{{ readme("INSTALLATION_STANDARD_SETUP") }}

{{ readme("INSTALLATION_PYPI_BASIC") }}

## Optional Dependencies

{{ readme("INSTALLATION_OPTIONAL_INTRO") }}

{{ readme("INSTALLATION_OPTIONAL_FEATURES") }}

<!-- INSTALLATION_JUPYTER_GUIDE_START -->

## Using GAICo with Jupyter Notebooks/Lab

If you plan to use GAICo within Jupyter Notebooks or JupyterLab (recommended for exploring examples and interactive analysis), install them into the _same activated virtual environment_:

```shell
# (Ensure your 'gaico-env' is active)
pip install notebook  # For Jupyter Notebook
# OR
# pip install jupyterlab # For JupyterLab
```

Then, launch Jupyter from the same terminal where your virtual environment is active:

```shell
# (Ensure your 'gaico-env' is active)
jupyter notebook
# OR
# jupyter lab
```

New notebooks created in this session should automatically use the `gaico-env` Python environment.

### Troubleshooting Jupyter Kernels

If your notebooks don't recognize the GAICo installation:

1. **Install ipykernel** in your virtual environment:
   ```shell
   pip install ipykernel
   ```

2. **Register the environment as a Jupyter kernel:**
   ```shell
   python -m ipykernel install --user --name=gaico-env --display-name "Python (gaico-env)"
   ```

3. **Select the kernel** in Jupyter:
   - In Jupyter Notebook: Kernel → Change Kernel → Python (gaico-env)
   - In JupyterLab: Click the kernel name in the top-right corner and select Python (gaico-env)

For more troubleshooting, see our [FAQ](faq.md).

<!-- INSTALLATION_JUPYTER_GUIDE_END -->

<!-- INSTALLATION_SIZE_TABLE_INTRO_START -->

## Installation Size Comparison

<!-- INSTALLATION_SIZE_TABLE_INTRO_END -->

<!-- INSTALLATION_SIZE_TABLE_CONTENT_START -->

The following table provides an _estimated_ overview of the relative disk space impact of different installation options. Actual sizes may vary depending on your operating system, Python version, and existing packages. These are primarily to illustrate the relative impact of optional dependencies.

_Note:_ Core dependencies include: `levenshtein`, `matplotlib`, `numpy`, `pandas`, `rouge-score`, and `seaborn`.

| Installation Command                              | Dependencies                                                 | Estimated Total Size Impact |
| ------------------------------------------------- | ------------------------------------------------------------ | --------------------------- |
| `pip install gaico`                               | Core                                                         | 215 MB                      |
| `pip install 'gaico[audio]'`                      | Core + `scipy`, `soundfile`                                  | 330 MB                      |
| `pip install 'gaico[bertscore]'`                  | Core + `bert-score` (includes `torch`, `transformers`, etc.) | 800 MB                      |
| `pip install 'gaico[cosine]'`                     | Core + `scikit-learn`                                        | 360 MB                      |
| `pip install 'gaico[jsd]'`                        | Core + `scipy`, `nltk`                                       | 310 MB                      |
| `pip install 'gaico[audio,jsd,cosine,bertscore]'` | Core + all dependencies from above                           | 1.0 GB                      |

<!-- INSTALLATION_SIZE_TABLE_CONTENT_END -->

!!! tip "Managing Installation Size"
    If disk space is a concern, install only the optional dependencies you need. You can always add more later using `pip install 'gaico[feature]'`.

<!-- INSTALLATION_DEVELOPER_GUIDE_START -->

## Developer Installation

If you want to contribute to GAICo or install it from source for development:

1.  **Clone the repository:**

    ```shell
    git clone https://github.com/ai4society/GenAIResultsComparator.git
    cd GenAIResultsComparator
    ```

2.  **Set up a virtual environment and install dependencies:**

    _We recommend using [UV](https://docs.astral.sh/uv/#installation) for fast environment and dependency management._

    === "Using UV (Recommended)"

        ```shell
        # Create a virtual environment (Python 3.10-3.12 recommended)
        uv venv
        # Activate the environment
        source .venv/bin/activate  # On Windows: .venv\Scripts\activate
        # Install in editable mode with all development dependencies
        uv pip install -e ".[dev]"
        ```

    === "Using pip"

        ```shell
        # Create a virtual environment (Python 3.10-3.12 recommended)
        python3 -m venv .venv
        # Activate the environment
        source .venv/bin/activate  # On Windows: .venv\Scripts\activate
        # Install the package in editable mode with development extras
        pip install -e ".[dev]"
        ```

    The `dev` extra installs GAICo with all optional features, plus dependencies for testing, linting, and documentation.

3.  **Set up pre-commit hooks** (recommended for contributors):

    _Pre-commit hooks help maintain code quality by running checks automatically before you commit._

    ```shell
    pre-commit install
    ```

<!-- INSTALLATION_DEVELOPER_GUIDE_END -->

For more information on development workflows, see our [Developer Guide](developer-guide.md).

## Common Installation Issues

### Import Errors After Installation

If you get import errors after installing GAICo:

1. Verify installation: `pip list | grep gaico`
2. Check you're in the correct virtual environment
3. Try reinstalling: `pip install --force-reinstall gaico`

### Dependency Conflicts

If you encounter dependency conflicts:

1. Create a fresh virtual environment
2. Install GAICo first before other packages
3. If conflicts persist, see our [FAQ](faq.md) for potential solutions

### Platform-Specific Issues

- **Windows**: Use `gaico-env\Scripts\activate` to activate the virtual environment
- **macOS/Linux**: Use `source gaico-env/bin/activate`
- **M1/M2 Macs**: Some dependencies may need Rosetta. See [FAQ](faq.md) for details.

## Next Steps

- 📚 Read the [Quick Start Guide](index.md#quick-start)
- 📓 Explore [example notebooks](https://github.com/ai4society/GenAIResultsComparator/tree/main/examples)
- 🔧 Set up your [development environment](developer-guide.md)
