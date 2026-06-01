import os
from pathlib import Path
import subprocess
import sys
import textwrap

from gaico import BaseMetric, Experiment
from gaico.metrics import BLEU, ROUGE


def test_existing_public_imports_still_work():
    assert Experiment.__name__ == "Experiment"
    assert BaseMetric.__name__ == "BaseMetric"
    assert BLEU.__name__ == "BLEU"
    assert ROUGE.__name__ == "ROUGE"


def test_plugin_package_can_extend_gaico_metrics_namespace(tmp_path):
    plugin_package = tmp_path / "gaico" / "metrics" / "health"
    plugin_package.mkdir(parents=True)
    plugin_package.joinpath("__init__.py").write_text(
        "class HealthMetric:\n"
        "    name = 'health'\n",
        encoding="utf-8",
    )

    repo_root = Path(__file__).resolve().parents[1]
    pythonpath = os.pathsep.join(
        path
        for path in [str(tmp_path), str(repo_root), os.environ.get("PYTHONPATH", "")]
        if path
    )
    env = {**os.environ, "PYTHONPATH": pythonpath}

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                from gaico import BaseMetric, Experiment
                from gaico.metrics import BLEU, ROUGE
                from gaico.metrics.health import HealthMetric

                assert Experiment.__name__ == "Experiment"
                assert BaseMetric.__name__ == "BaseMetric"
                assert BLEU.__name__ == "BLEU"
                assert ROUGE.__name__ == "ROUGE"
                assert HealthMetric.name == "health"
                """
            ),
        ],
        cwd=repo_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
