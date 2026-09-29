"""Test that verifies expected success rates for pretrained evaluation runs.

Each case is skipped unless its final models exist under ``runs/<env name>/<alg>/<run>/models``.
"""

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

from jaxolotl.utils.artifact_utils import discover_seed_models

PROJECT_ROOT = Path(__file__).parents[1]
SUCCESS_RATE_PATTERN = re.compile(
    r"Overall SR/AV:\s*(?P<success_rate>[0-9]+(?:\.[0-9]+)?)\s*.*"
)


@dataclass(frozen=True)
class EvaluationCase:
    alg: str
    expected_success_rate: float
    env: str = "zone_env"
    task_set: str = "finite"
    run: str = "pretrained"
    tolerance: float = 0.05
    extra_overrides: tuple[str, ...] = ("+eval.num_seeds=1",)

    @property
    def hydra_overrides(self) -> tuple[str, ...]:
        return (
            f"alg={self.alg}",
            f"env={self.env}",
            f"task_set={self.task_set}",
            f"run={self.run}",
            *self.extra_overrides,
        )

    @property
    def run_dir(self) -> Path:
        env_to_name = {
            "zone_env": "ZoneEnv",
            "letter_world": "LetterWorld",
            "warehouse": "WarehouseEnv",
        }
        return PROJECT_ROOT / "runs" / env_to_name[self.env] / self.alg / self.run

    @property
    def name(self) -> str:
        return f"{self.alg}/{self.env}-{self.task_set}"


EVALUATION_CASES = (
    EvaluationCase(alg="struct_ltl", expected_success_rate=0.95),
    EvaluationCase(alg="deep_ltl", expected_success_rate=0.92),
    EvaluationCase(alg="genz_ltl", expected_success_rate=1.0),
    EvaluationCase(alg="sem_ltl", expected_success_rate=0.83),
    EvaluationCase(alg="gcrl_ltl", expected_success_rate=0.93),
    EvaluationCase(alg="ltl2action", expected_success_rate=0.43),
)


def _format_process_output(result: subprocess.CompletedProcess[str]) -> str:
    return f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


@pytest.fixture(scope="session")
def require_cuda_gpu() -> None:
    if shutil.which("nvidia-smi") is None:
        pytest.skip("nvidia-smi is unavailable; a CUDA GPU is required")

    result = subprocess.run(
        ["nvidia-smi", "-L"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        pytest.skip(
            f"nvidia-smi could not access a CUDA GPU.\n{_format_process_output(result)}"
        )


@pytest.mark.e2e
@pytest.mark.usefixtures("require_cuda_gpu")
@pytest.mark.parametrize(
    "case",
    EVALUATION_CASES,
    ids=lambda case: case.name,
)
def test_evaluation_success_rate(case: EvaluationCase) -> None:
    if not discover_seed_models(case.run_dir):
        pytest.skip(f"no pretrained models found in {case.run_dir}")

    result = subprocess.run(
        [
            "pixi",
            "run",
            "-e",
            "gpu",
            "python",
            "scripts/eval/eval.py",
            *case.hydra_overrides,
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    process_output = _format_process_output(result)

    assert result.returncode == 0, (
        f"Evaluation command exited with status {result.returncode}.\n{process_output}"
    )

    match = SUCCESS_RATE_PATTERN.search(f"{result.stdout}\n{result.stderr}")
    assert match is not None, (
        f"Evaluation output did not contain 'Overall SR/AV:'.\n{process_output}"
    )

    success_rate = float(match.group("success_rate"))
    assert success_rate == pytest.approx(
        case.expected_success_rate,
        abs=case.tolerance,
    )
