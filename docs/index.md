# Welcome to GAICo

<figure markdown="span">
  <img src="https://raw.githubusercontent.com/ai4society/GenAIResultsComparator/refs/heads/main/docs/misc/quickstart.gif" alt="GIF Showing GAICo's Quickstart" style="display: block; margin: auto; border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);">
  <figcaption><em>GAICo Quickstart Demonstration</em></figcaption>
</figure>

{{ readme("BADGES") }}

{{ readme("TAGLINE") }}

!!! success "🥳 Papers accepted at IAAI/AAAI 2026 and AAAI Demonstrations 2026!"
    We're pleased to announce our acceptance! Check out our materials:

    *   **Papers:** [Demo Paper (PDF)](https://ai4society.github.io/publications/papers_local/GAICO-Demo-AAAI2026.pdf) | [Main Paper (arXiv)](https://arxiv.org/abs/2508.16753)
    *   **Try it out:** [Interactive Demo App](https://gaico-demo.streamlit.app/)
    *   **Conference Tracks:** [IAAI-26 Call](https://aaai.org/conference/aaai/aaai-26/iaai-26-call/) | [AAAI-26 Demo Call](https://aaai.org/conference/aaai/aaai-26/demonstration-call/)

## What is GAICo?

<figure markdown="span">
  ![GAICo Overview](https://raw.githubusercontent.com/ai4society/GenAIResultsComparator/refs/heads/main/gaico.drawio.png){ width="500" }
  <figcaption><em>Overview of the workflow supported by the <i>GAICo</i> library</em></figcaption>
</figure>

{{ readme("DESCRIPTION_CORE_CONCEPT") }}

**Key capabilities:**

- **Batch processing**: Efficiently evaluate entire datasets with one-to-one or one-to-many comparisons
- **Flexible inputs**: Works with strings, lists, NumPy arrays, and Pandas Series
- **Extensible architecture**: Easily add custom metrics by inheriting from `BaseMetric`
- **Automated reporting**: Generate CSV reports and visualizations (bar charts, radar plots)

!!! note "Dataset Evaluation"
    The `Experiment` class evaluates model responses against a **single reference** at a time. For full dataset evaluation, either iterate with `Experiment` or use metric classes directly. See our [FAQ](faq.md) for details.

## Quick Navigation

<div class="grid cards" markdown>

-   :material-download:{ .lg .middle } **Installation**

    ---

    Get GAICo installed quickly with pip

    [:octicons-arrow-right-24: Installation Guide](installation-guide.md)

-   :material-rocket-launch:{ .lg .middle } **Quick Start**

    ---

    Start evaluating LLM outputs in 2 minutes

    [:octicons-arrow-right-24: Quick Start](quickstart.md)

-   :material-notebook:{ .lg .middle } **Examples**

    ---

    Explore Jupyter notebooks and demos

    [:octicons-arrow-right-24: Resources](resources.md)

-   :material-frequently-asked-questions:{ .lg .middle } **FAQ**

    ---

    Common questions and troubleshooting

    [:octicons-arrow-right-24: FAQ](faq.md)

</div>

## Quick Installation

{{ readme("INSTALLATION_STANDARD_INTRO") }}

{{ readme("INSTALLATION_STANDARD_SETUP") }}

{{ readme("INSTALLATION_PYPI_BASIC") }}

{{ readme("INSTALLATION_OPTIONAL_INTRO") }}

{{ readme("INSTALLATION_OPTIONAL_FEATURES") }}

!!! tip
    For detailed installation instructions including Jupyter setup, developer installation, and size comparisons, see our [Installation Guide](installation-guide.md).

## Quick Start

We demonstrate a simple example comparing outputs from multiple LLMs using two text similarity metrics: Jaccard and ROUGE. Sample data is from [https://arxiv.org/abs/2504.07995](https://arxiv.org/abs/2504.07995).

{{ readme("QUICKSTART_CODE") }}

**Explore complete examples:**

- [`quickstart.ipynb`](https://github.com/ai4society/GenAIResultsComparator/blob/main/examples/quickstart.ipynb) - Hands-on introduction [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai4society/GenAIResultsComparator/blob/main/examples/quickstart.ipynb)
- [`example-1.ipynb`](https://github.com/ai4society/GenAIResultsComparator/blob/main/examples/example-1.ipynb) - Multiple models, single metric [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai4society/GenAIResultsComparator/blob/main/examples/example-1.ipynb)
- [`example-2.ipynb`](https://github.com/ai4society/GenAIResultsComparator/blob/main/examples/example-2.ipynb) - Single model, all metrics [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai4society/GenAIResultsComparator/blob/main/examples/example-2.ipynb)

!!! tip
    More examples, videos, and interactive demos available on our [Resources page](resources.md).

## Features

- **Comprehensive Metric Library:**
    - Textual Similarity: Jaccard, Cosine, Levenshtein, Sequence Matcher.
    - N-gram Based: BLEU, ROUGE, JS Divergence.
    - Semantic Similarity: BERTScore.
    - Structured Data: Specialized metrics for planning sequences (`PlanningLCS`, `PlanningJaccard`) and time-series data (`TimeSeriesElementDiff`, `TimeSeriesDTW`).
    - Multimedia: Metrics for image similarity (`ImageSSIM`, `ImageAverageHash`, `ImageHistogramMatch`) and audio quality (`AudioSNRNormalized`, `AudioSpectrogramDistance`).
- **Streamlined Evaluation Workflow:** A high-level `Experiment` class to easily compare multiple models, apply thresholds, generate plots, and create CSV reports.
- **Enhanced Reporting:** A `summarize()` method for quick, aggregated overviews of model performance, including mean scores and pass rates.
- **Dynamic Metric Registration:** Easily extend the `Experiment` class by registering your own custom `BaseMetric` implementations at runtime.
- **Powerful Visualization:** Generate bar charts and radar plots to compare model performance using Matplotlib and Seaborn.
- **Efficient & Flexible:**
    - Supports batch processing for efficient computation on datasets.
    - Optimized for various input types (lists, NumPy arrays, Pandas Series).
    - Easily extensible architecture for adding new custom metrics.
- **Robust and Reliable:** Includes a comprehensive test suite using [Pytest](https://docs.pytest.org/en/stable/).

!!! tip "Want to add your own metric?"
    Check our [custom metrics guide](faq.md#q-how-do-i-add-a-new-custom-metric).

## Latest Updates

!!! info "Latest release information"
    Stay up to date with GAICo releases and news: [Release notes and version history →](news.md)

{{ readme("CITATION_SECTION") }}

{{ readme("ACKNOWLEDGMENTS_SECTION") }}

{{ readme("CONTACT_SECTION") }}

## Additional Resources

- 🎥 [Video Demo](resources.md#video-demo)
- 💻 [Interactive Demo](resources.md#interactive-demo)
- 📚 [All Examples](resources.md#example-notebooks)
- 🔧 [Developer Guide](developer-guide.md)
- 📖 [Installation Guide](installation-guide.md)
