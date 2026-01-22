# Developer Guide

This guide covers everything you need to contribute to GAICo, including project structure, testing, code style, and development workflows. _Use the table of contents in the left sidebar to navigate._

For development setup, wee the [Developer Installation](installation-guide.md#developer-installation) section.

## Project Structure

The [project](https://github.com/ai4society/GenAIResultsComparator) is organized as follows:

```shell
.
├── README.md               # Project overview and quick start
├── LICENSE                 # MIT License
├── .gitignore              # Git ignore rules
├── uv.lock                 # UV dependency lock file
├── pyproject.toml          # Project metadata and dependencies
├── project_macros.py       # Used by mkdocs-macros-plugin (documentation)
├── PYPI_DESCRIPTION.MD     # PyPI package description
├── .pre-commit-config.yaml # Pre-commit hook configuration
├── mkdocs.yml              # MkDocs documentation configuration
├── gaico/                  # Main library code
│   ├── __init__.py         # Package initialization
│   ├── base.py             # BaseMetric abstract class
│   ├── experiment.py       # Experiment class for streamlined evaluation
│   ├── metrics/            # Individual metric implementations
│   │   ├── text/           # Text-based metrics
│   │   ├── structured/     # Structured data metrics
│   │   └── multimedia/     # Image and audio metrics
│   └── utils/              # Utility functions
├── examples/               # Jupyter notebook examples
│   ├── quickstart.ipynb    # Quick introduction
│   ├── example-1.ipynb     # Multiple models, single metric
│   ├── example-2.ipynb     # Single model, all metrics
│   └── data/               # Sample data for examples
├── tests/                  # Test suite
│   ├── test_metrics/       # Metric-specific tests
│   ├── test_experiment.py  # Experiment class tests
│   └── conftest.py         # Pytest configuration
├── docs/                   # Documentation source files
│   ├── index.md            # Documentation homepage
│   ├── installation-guide.md
│   ├── developer-guide.md
│   ├── resources.md
│   ├── faq.md
│   └── news.md
├── scripts/                # Utility scripts
│   ├── deploy-docs.sh      # Documentation deployment
│   └── generate-readme.py  # README generation
└── .github/workflows/      # CI/CD workflows
    ├── deploy-docs.yml     # Documentation deployment
    └── publish-pypi.yml    # PyPI publishing
```

## Running Tests

We use [Pytest](https://docs.pytest.org/) for testing. Tests are located in the `tests/` directory.

### Basic Test Commands

Navigate to the project root and use `uv` to run tests:

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run with coverage report
uv run pytest --cov=gaico --cov-report=html

# If pytest gives import errors, use:
uv run -m pytest
```

### Targeting Specific Tests

You can run or skip tests based on markers:

```bash
# Skip slow BERTScore tests
uv run pytest -m "not bertscore"

# Run ONLY BERTScore tests
uv run pytest -m bertscore

# Run tests for a specific file
uv run pytest tests/test_experiment.py

# Run a specific test function
uv run pytest tests/test_experiment.py::test_experiment_init
```

### Test Markers

We use the following pytest markers:

- `bertscore`: Tests for BERTScore metric (can be slow)
- `integration`: Integration tests that test multiple components
- `unit`: Fast unit tests for individual functions

### Writing Tests

When adding new features:

1. **Create tests first** (TDD approach recommended)
2. **Place tests** in the appropriate `tests/test_*.py` file
3. **Use descriptive names**: `test_feature_behavior_expected_result`
4. **Include edge cases**: Empty inputs, None values, type mismatches
5. **Add docstrings** explaining what the test validates

Example test structure:

```python
import pytest
from gaico.metrics import YourNewMetric

def test_your_metric_basic_functionality():
    """Test that YourNewMetric calculates scores correctly."""
    metric = YourNewMetric()
    generated = "test output"
    reference = "test reference"

    result = metric.calculate(generated, reference)

    assert 0 <= result <= 1, "Score should be between 0 and 1"
    assert isinstance(result, float), "Result should be a float"

@pytest.mark.parametrize("generated,reference,expected", [
    ("exact", "exact", 1.0),
    ("different", "words", 0.0),
])
def test_your_metric_edge_cases(generated, reference, expected):
    """Test edge cases for YourNewMetric."""
    metric = YourNewMetric()
    result = metric.calculate(generated, reference)
    assert result == pytest.approx(expected, rel=1e-2)
```

## Code Style

We maintain code quality using pre-commit hooks. Configuration is in `.pre-commit-config.yaml`.

### Pre-commit Hooks

**Setup** (run once after cloning):

```bash
pre-commit install
```

**Running hooks manually:**

```bash
# Run on all files
pre-commit run --all-files

# Run on staged files only
pre-commit run

# Run a specific hook
pre-commit run black --all-files
```

### Code Style Tools

Our pre-commit hooks include:

- **Black**: Code formatting (line length: 88)
- **isort**: Import sorting
- **Flake8**: Linting
- **mypy**: Type checking (optional)
- **trailing-whitespace**: Remove trailing spaces
- **end-of-file-fixer**: Ensure files end with newline

### Style Guidelines

- **Formatting**: Use Black's defaults (88 character line length)
- **Imports**: Group stdlib, third-party, local (enforced by isort)
- **Type hints**: Add type hints for public APIs
- **Docstrings**: Use Google-style docstrings
- **Variable names**:
  - `snake_case` for functions and variables
  - `PascalCase` for classes
  - `UPPER_CASE` for constants

Example:

```python
from typing import List, Union
import numpy as np

class ExampleMetric(BaseMetric):
    """Brief description of the metric.

    More detailed explanation of what this metric does,
    its use cases, and any important notes.

    Args:
        parameter1: Description of parameter1
        parameter2: Description of parameter2

    Attributes:
        attribute1: Description of attribute1
    """

    DEFAULT_THRESHOLD = 0.5  # Class constant

    def __init__(self, parameter1: str, parameter2: int = 10):
        """Initialize the metric."""
        self.parameter1 = parameter1
        self.parameter2 = parameter2

    def calculate(
        self,
        generated_texts: Union[str, List[str]],
        reference_texts: Union[str, List[str], None] = None
    ) -> Union[float, np.ndarray]:
        """Calculate the metric score.

        Args:
            generated_texts: Model-generated output(s)
            reference_texts: Reference output(s) or None

        Returns:
            Score(s) between 0 and 1

        Raises:
            ValueError: If inputs are invalid
        """
        # Implementation here
        pass
```

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Write code following style guidelines
- Add/update tests
- Update documentation if needed

### 3. Run Tests and Checks

```bash
# Run tests
uv run pytest

# Run pre-commit checks
pre-commit run --all-files

# Check if documentation builds
mkdocs serve
```

### 4. Commit Changes

```bash
git add .
git commit -m "Add: Brief description of changes"
```

Pre-commit hooks will run automatically. Fix any issues they report.

### 5. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## Adding a New Metric

See our [FAQ guide on adding custom metrics](faq.md#q-how-do-i-add-a-new-custom-metric) for detailed instructions.

Quick checklist:

1. Create new file in `gaico/metrics/[category]/`
2. Inherit from `BaseMetric`
3. Implement `calculate()` method
4. Add tests in `tests/test_metrics/`
5. Update documentation
6. Register in `__init__.py` if needed

## Building Documentation

We use [MkDocs](https://www.mkdocs.org/) with the Material theme.

### Local Documentation Server

```bash
# Install documentation dependencies (included in dev install)
pip install -e ".[dev]"

# Serve documentation locally
mkdocs serve
```

Visit `http://127.0.0.1:8000` to view the docs.

### Building Documentation

```bash
# Build static site
mkdocs build

# Build and deploy to GitHub Pages (maintainers only)
mkdocs gh-deploy
```

## Release Process

For maintainers releasing new versions:

1. **Update version** in `pyproject.toml`
2. **Update changelog** in `docs/news.md` and `docs/resources.md`
3. **Run full test suite**: `uv run pytest`
4. **Build package**: `uv build`
5. **Create git tag**: `git tag v0.x.x`
6. **Push tag**: `git push origin v0.x.x`
7. **GitHub Actions** will automatically publish to PyPI

## Getting Help

- 💬 Open an issue on [GitHub](https://github.com/ai4society/GenAIResultsComparator/issues)
- 📧 Email us at [ai4societyteam@gmail.com](mailto:ai4societyteam@gmail.com)
- 📖 Check the [FAQ](faq.md)

## Code of Conduct

We follow a standard code of conduct:

- Be respectful and inclusive
- Provide constructive feedback
- Focus on what's best for the community
- Show empathy towards other community members
