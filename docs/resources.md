# Resources

This page collects all GAICo resources including videos, demos, examples, and version history. _Use the table of contents in the left sidebar to navigate._

## **Video Demo**

<div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.15); margin: 20px 0;">
  <iframe src="https://www.youtube.com/embed/ixxqTT-eV48?rel=0"
          title="GAICo YouTube Demo"
          style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; border: 0;"
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
          allowfullscreen>
  </iframe>
</div>

Watch this comprehensive demonstration of GAICo's capabilities, including:

- Setting up and running evaluations
- Comparing multiple LLM outputs
- Generating visualizations
- Working with different metric types
- Interpreting results

[View on YouTube →](https://youtu.be/ixxqTT-eV48){ .md-button }

## **Interactive Demo**

Try GAICo without installing anything:

[🚀 Launch Streamlit Demo](https://gaico-demo.streamlit.app/){ .md-button .md-button--primary }

The interactive demo allows you to:

- Upload your own LLM outputs
- Select metrics to apply
- Generate comparison visualizations
- Download results as CSV

## **Version History & News**

### Recent Releases

This section summarizes the major releases of the GAICo library, highlighting key features and providing quick start examples.

| Release | Date | Summary | Details |
|---------|------|---------|---------|
| v0.3.0 | August 2025 | Added multimedia metrics (image and audio) and enhancements for the `Experiment` class | [Full changelog →](news.md#v030-august-2025) |
| v0.2.0 | July 2025 | Added specialized text metrics: time-series & automated planning | [Full changelog →](news.md#v020-july-2025) |
| v0.1.5 | June 2025 | Initial release: generic text metrics, `Experiment` class, & visualizations | [Full changelog →](news.md#v015-june-2025) |

[View all release notes →](news.md){ .md-button }

## **Example Notebooks**

All examples are available as Jupyter notebooks that can be run locally or in Google Colab.

### Quick Start Examples

| Notebook | Description | Open in Colab |
|----------|-------------|---------------|
| [quickstart.ipynb](https://github.com/ai4society/GenAIResultsComparator/blob/main/examples/quickstart.ipynb) | Rapid hands-on introduction to the `Experiment` class | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai4society/GenAIResultsComparator/blob/main/examples/quickstart.ipynb) |
| [example-1.ipynb](https://github.com/ai4society/GenAIResultsComparator/blob/main/examples/example-1.ipynb) | Compare **multiple model outputs** with a **single metric** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai4society/GenAIResultsComparator/blob/main/examples/example-1.ipynb) |
| [example-2.ipynb](https://github.com/ai4society/GenAIResultsComparator/blob/main/examples/example-2.ipynb) | Evaluate a **single model output** across **all available metrics** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai4society/GenAIResultsComparator/blob/main/examples/example-2.ipynb) |

### Advanced Examples

Browse the full [examples directory](https://github.com/ai4society/GenAIResultsComparator/tree/main/examples) for more specialized use cases:

- Working with multimedia metrics (images, audio)
- Batch processing large datasets
- Custom metric implementation
- Advanced visualization techniques
- Integration with popular LLM frameworks

## **Learning Resources**

### Documentation

- 📖 [Installation Guide](installation-guide.md) - Detailed setup instructions
- 🔧 [Developer Guide](developer-guide.md) - Contributing and development
- 🤔 [FAQ](faq.md) - Frequently asked questions
- 📊 [API Reference](api/metrics/index.md) - Complete API documentation

### External Resources

- [Microsoft's Guide to LLM Evaluation](https://learn.microsoft.com/en-us/ai/playbook/technology-guidance/generative-ai/working-with-llms/evaluation/list-of-eval-metrics) - Inspiration for GAICo's metrics
- [Hugging Face Evaluate Library](https://huggingface.co/docs/evaluate/index) - Complementary evaluation tools
- [HELM Benchmark](https://crfm.stanford.edu/helm/) - Holistic evaluation framework

## **Community & Support**

### Get Help

- 💬 [GitHub Discussions](https://github.com/ai4society/GenAIResultsComparator/discussions) - Ask questions and share ideas
- 🐛 [Issue Tracker](https://github.com/ai4society/GenAIResultsComparator/issues) - Report bugs or request features
- 📧 [Email Support](mailto:ai4societyteam@gmail.com) - Direct contact with the team

### Contributing

We welcome contributions! See our [Developer Guide](developer-guide.md) for:

- Setting up your development environment
- Code style guidelines
- Testing requirements
- Pull request process

## **Publications**

1. 📄 [GAICo: A Deployed and Extensible Framework for Evaluating Diverse and Multimodal Generative AI Outputs](https://arxiv.org/abs/2508.16753)
2. 📑 [GAICo: Demonstrating a Unified Framework for Multi-Modal GenAI Evaluation (Demo)](https://ai4society.github.io/publications/papers_local/GAICO-Demo-AAAI2026.pdf)

**Citation:**
```bibtex
@article{gupta2025gaico,
  title={GAICo: A Deployed and Extensible Framework for Evaluating Diverse and Multimodal Generative AI Outputs},
  author={Gupta, Nitin and Koppisetti, Pallav and Lakkaraju, Kausik and Srivastava, Biplav},
  journal={arXiv preprint arXiv:2508.16753},
  year={2025}
}
```

### Cite GAICo

If you use GAICo in your research or projects, please cite our work. See the [Citation section](#publications) above.
