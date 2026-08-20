"""Test that verifies expected success rates for pretrained evaluation runs."""

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parents[1]
SUCCESS_RATE_PATTERN = re.compile(
    r"Overall SR/AV:\s*(?P<success_rate>[0-9]+(?:\.[0-9]+)?)\s*\+-"
)


@dataclass(frozen=True)
class EvaluationCase:
    name: str
    experiment: str
    formulas: str
    run: str
    expected_success_rate: float
    tolerance: float = 0.02
    extra_overrides: tuple[str, ...] = ("+eval.num_seeds=1",)

    @property
    def hydra_overrides(self) -> tuple[str, ...]:
        return (
            f"experiment={self.experiment}",
            f"formulas={self.formulas}",
            f"run={self.run}",
            *self.extra_overrides,
        )


EVALUATION_CASES = (
    EvaluationCase(
        name="StructLTL/Zones-NM",
        experiment="struct_ltl/zones_nm",
        formulas="zones_nm/finite",
        run="pretrained",
        expected_success_rate=0.946,
    ),
    EvaluationCase(
        name="DeepLTL/Zones-NM",
        experiment="deep_ltl/zones_nm",
        formulas="zones_nm/finite",
        run="pretrained",
        expected_success_rate=0.917,
    ),
    EvaluationCase(
        name="GenZ-LTL/Zones-NM",
        experiment="genz_ltl/zones_nm",
        formulas="zones_nm/finite",
        run="pretrained",
        expected_success_rate=0.812,
    ),
    EvaluationCase(
        name="LTL2Action/Zones-NM",
        experiment="ltl2action/zones_nm",
        formulas="zones_nm/finite",
        run="pretrained",
        expected_success_rate=0.621,
    ),
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
