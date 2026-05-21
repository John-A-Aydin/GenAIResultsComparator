# GAICo: GenAI Results Comparator

<!-- BADGES_START -->
<p align="center">
  <a href="https://pepy.tech/projects/gaico"><img src="https://static.pepy.tech/badge/gaico" alt="PyPI Downloads"></a>
  <a href="https://arxiv.org/abs/2508.16753"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2508.16753-b31b1b.svg?style=plastic"></a>
  <br>
  <a href="https://deepwiki.com/ai4society/GenAIResultsComparator"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki"></a>
  <a href="https://ai4society.github.io/projects/GenAIResultsComparator/"><img alt="Documentation" src="https://img.shields.io/badge/docs-mkdocs%20material-blue.svg?style=flat-square"></a>
  <br>
  <a href="https://pypi.org/project/GAICo/"><img alt="PyPI version" src="https://img.shields.io/pypi/v/GAICo.svg?style=flat-square"></a>
  <a href="https://pypi.org/project/GAICo/"><img alt="Python versions" src="https://img.shields.io/pypi/pyversions/GAICo.svg?style=flat-square"></a>
  <a href="https://github.com/ai4society/GenAIResultsComparator/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/ai4society/GenAIResultsComparator?style=flat-square"></a>
</p>
<!-- BADGES_END -->

<!-- TAGLINE_START -->

_GenAI Results Comparator (GAICo)_ helps you measure the quality of your Generative AI (LLM) outputs. It enables you to compare, analyze, and visualize results across text, images, audio, and structured data, helping you answer the question: "Which model performed better?"

<!-- TAGLINE_END -->

<!-- ACCEPTANCE_BANNER_START -->
<p align="center">
  <strong>🥳  Papers accepted at AAAI 2026!</strong><br/>
  <em>We're pleased to announce our acceptance! Check out our materials:<br>
  <strong>Papers:</strong> <a href="https://ai4society.github.io/publications/papers_local/GAICO-Demo-AAAI2026.pdf">Demo Paper (PDF)</a> | <a href="https://arxiv.org/abs/2508.16753">Main Paper (arXiv)</a><br/>
  <strong>Try it out:</strong> <a href="https://gaico-demo.streamlit.app/">Interactive Demo App</a><br/>
  <strong>Conference Tracks:</strong> <a href="https://aaai.org/conference/aaai/aaai-26/iaai-26-call/">IAAI-26 Call</a> | <a href="https://aaai.org/conference/aaai/aaai-26/demonstration-call/">AAAI-26 Demo Call</a>
  </em>
</p>
<!-- ACCEPTANCE_BANNER_END -->

> [!NOTE]
> This README provides a quick overview of GAICo. For detailed documentation, installation guides, examples, and developer resources, please visit our [Documentation Site](https://ai4society.github.io/projects/GenAIResultsComparator).

## Resources

📖 **[Documentation](https://ai4society.github.io/projects/GenAIResultsComparator)** · 📦 **[PyPI](https://pypi.org/project/gaico/)** · 📄 **[Technical Paper](https://arxiv.org/abs/2508.16753)** · 🎥 **[Video Demo](https://drive.google.com/file/d/1m93agi6H4-HDpIvZCiYOOw6RpYEIQbvD/view)** · 🤔 **[FAQ](https://ai4society.github.io/projects/GenAIResultsComparator/faq)** · 🗞️ **[News & Releases](https://ai4society.github.io/projects/GenAIResultsComparator/news)**

## What is GAICo?

<p align="center">
  <img src="https://raw.githubusercontent.com/ai4society/GenAIResultsComparator/refs/heads/main/gaico.drawio.png" alt="GAICo Overview" width="60%">
  <br/>
  <em>Overview of the workflow supported by the GAICo library</em>
</p>

<!-- DESCRIPTION_CORE_CONCEPT_START -->

At its core, the library provides a set of metrics for evaluating various types of outputs, from plain text strings to structured data like planning sequences and time-series, and multimedia content such as images and audio. While the `Experiment` class streamlines evaluation for text-based and structured string outputs, individual metric classes offer direct control for all data types, including binary or array-based multimedia. These metrics produce normalized scores (typically 0 to 1), where 1 indicates a perfect match, enabling robust analysis and visualization of LLM performance.

<!-- DESCRIPTION_CORE_CONCEPT_END -->

**Key capabilities:**

- **Batch processing**: Efficiently evaluate entire datasets with one-to-one or one-to-many comparisons
- **Flexible inputs**: Works with strings, lists, NumPy arrays, and Pandas Series
- **Extensible architecture**: Easily add custom metrics by inheriting from `BaseMetric`
- **Automated reporting**: Generate CSV reports and visualizations (bar charts, radar plots)

> [!NOTE]
> The `Experiment` class evaluates model responses against a **single reference** at a time. For full dataset evaluation, either iterate with `Experiment` or use metric classes directly. See our [FAQ](https://ai4society.github.io/projects/GenAIResultsComparator/faq) for details.

## Installation

> [!IMPORTANT]
> We recommend using a [Python virtual environment](https://docs.python.org/3/tutorial/venv.html) to manage dependencies.

<!-- INSTALLATION_STANDARD_INTRO_START -->

GAICo can be installed using pip.

<!-- INSTALLATION_STANDARD_INTRO_END -->

<!-- INSTALLATION_STANDARD_SETUP_START -->

**Create and activate a virtual environment:**

```shell
python3 -m venv gaico-env
source gaico-env/bin/activate  # On macOS/Linux
# gaico-env\Scripts\activate   # On Windows
```

<!-- INSTALLATION_STANDARD_SETUP_END -->

<!-- INSTALLATION_PYPI_BASIC_START -->

**Install GAICo:**

```shell
pip install gaico
```

This installs the core GAICo library with essential metrics.

<!-- INSTALLATION_PYPI_BASIC_END -->

<!-- INSTALLATION_OPTIONAL_INTRO_START -->

**Optional dependencies** for specialized metrics:

<!-- INSTALLATION_OPTIONAL_INTRO_END -->

<!-- INSTALLATION_OPTIONAL_FEATURES_START -->

```shell
pip install 'gaico[audio]'                       # Audio metrics
pip install 'gaico[bertscore]'                   # BERTScore metric
pip install 'gaico[cosine]'                      # Cosine similarity
pip install 'gaico[jsd]'                         # JS Divergence
pip install 'gaico[audio,bertscore,cosine,jsd]'  # All features
```

<!-- INSTALLATION_OPTIONAL_FEATURES_END -->

> [!TIP]
> For detailed installation instructions including Jupyter setup, developer installation, installation size comparisons, and troubleshooting, see our [Installation Guide](https://ai4society.github.io/projects/GenAIResultsComparator/installation-guide).

## Quick Start

Get started with GAICo in under 2 minutes:

<!-- QUICKSTART_CODE_START -->

```python
from gaico import Experiment

# Sample LLM responses comparing different models
llm_responses = {
    "Google": "Title: Jimmy Kimmel Reacts to Donald Trump Winning...",
    "Mixtral 8x7b": "I'm an AI and I don't have the ability to predict...",
    "SafeChat": "Sorry, I am designed not to answer such a question.",
}
reference_answer = "Sorry, I am unable to answer such a question as it is not appropriate."

# Initialize and run comparison
exp = Experiment(llm_responses=llm_responses, reference_answer=reference_answer)
results = exp.compare(
    metrics=['Jaccard', 'ROUGE'],
    plot=True,
    output_csv_path="experiment_report.csv"
)

print(results)
```

<!-- QUICKSTART_CODE_END -->

**Explore complete examples:**

- [`quickstart.ipynb`](https://github.com/ai4society/GenAIResultsComparator/blob/main/examples/quickstart.ipynb) - Hands-on introduction to the `Experiment` class [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai4society/GenAIResultsComparator/blob/main/examples/quickstart.ipynb)
- [`example-1.ipynb`](https://github.com/ai4society/GenAIResultsComparator/blob/main/examples/example-1.ipynb) - Multiple models, single metric [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai4society/GenAIResultsComparator/blob/main/examples/example-1.ipynb)
- [`example-2.ipynb`](https://github.com/ai4society/GenAIResultsComparator/blob/main/examples/example-2.ipynb) - Single model, all metrics [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai4society/GenAIResultsComparator/blob/main/examples/example-2.ipynb)

> [!TIP]
> More examples available in the [`examples/`](examples/) folder. Run them directly in Google Colab! See our [Resources page](https://ai4society.github.io/projects/GenAIResultsComparator/resources) for videos, demos, and version history.

<!-- FEATURES_SECTION_START -->

## Features

<!-- FEATURES_LIST_START -->

- **Comprehensive Metric Library**
  - Textual similarity: Jaccard, Cosine, Levenshtein, Sequence Matcher
  - N-gram based: BLEU, ROUGE, JS Divergence
  - Semantic similarity: BERTScore
  - Structured data: Planning sequences and time-series metrics
  - Multimedia: Image similarity (SSIM, hash-based) and audio quality metrics

- **Streamlined Evaluation Workflow**
  - High-level `Experiment` class for comparing models, applying thresholds, and generating reports
  - `summarize()` method for aggregated performance overviews

- **Dynamic & Extensible**
  - Register custom metrics at runtime
  - Add your own evaluation criteria easily

- **Powerful Visualization**
  - Generate comparative plots automatically
  - Support for bar charts and radar plots

- **Robust & Tested**
  - Comprehensive test suite with Pytest
  - Production-ready reliability

<!-- FEATURES_LIST_END -->

> [!TIP]
> Want to add your own metric? Check our [custom metrics guide](https://ai4society.github.io/projects/GenAIResultsComparator/faq/#q-how-do-i-add-a-new-custom-metric).

<!-- FEATURES_SECTION_END -->

<!-- CITATION_SECTION_START -->

## Citation

<!-- CITATION_CONTENT_START -->

If you find this project useful, please cite our work:

```bibtex
@article{Gupta_Koppisetti_Lakkaraju_Srivastava_2026,
  title={GAICo: A Deployed and Extensible Framework for Evaluating Diverse and Multimodal Generative AI Outputs},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence},
  author={Gupta, Nitin and Koppisetti, Pallav and Lakkaraju, Kausik and Srivastava, Biplav},
  year={2026},
}
```

<!-- CITATION_CONTENT_END -->
<!-- CITATION_SECTION_END -->

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/FeatureName`)
3. Commit your changes with clear messages
4. Ensure tests pass and code follows our style guidelines
5. Submit a Pull Request

> [!TIP]
> For development setup, running tests, code style guidelines, and project structure details, see our [Developer Guide](https://ai4society.github.io/projects/GenAIResultsComparator/developer-guide).

<!-- ACKNOWLEDGMENTS_SECTION_START -->

## Acknowledgments

- The library is developed by [Nitin Gupta](https://github.com/g-nitin), [Pallav Koppisetti](https://github.com/pallavkoppisetti), [Kausik Lakkaraju](https://github.com/kausik-l), and [Biplav Srivastava](https://github.com/biplav-s). Members of [AI4Society](https://ai4society.github.io) contributed to this tool as part of ongoing discussions. Major contributors are credited.
- This library uses several open-source packages including NLTK, scikit-learn, and others. Special thanks to the creators and maintainers of the implemented metrics.

<!-- ACKNOWLEDGMENTS_SECTION_END -->

## License & Contact

This project is licensed under the MIT License - see the [LICENSE](https://github.com/ai4society/GenAIResultsComparator/blob/main/LICENSE) file for details.

<!-- CONTACT_SECTION_START -->

**Questions?** Reach out at [ai4societyteam@gmail.com](mailto:ai4societyteam@gmail.com)

<!-- CONTACT_SECTION_END -->
