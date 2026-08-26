"""Utility functions for calling SemML to convert LTL formulas to LDBAs."""

import subprocess
import tempfile
from collections.abc import Iterable
from pathlib import Path

from jaxolotl import DATA_DIR, DEPENDENCIES_DIR
from jaxolotl.ltl.logic.assignment import Assignment

SEMML_PATH = DEPENDENCIES_DIR / "semml/scripts_semml/embedd_ldba.py"


def run_semml(
    formula: str,
    propositions: Iterable[str],
    assignments: Iterable[Assignment],
    use_attention_feature: bool = True,
) -> str:
    """Convert an LTL formula to a LDBA using SemML."""

    if not SEMML_PATH.exists():
        raise RuntimeError(
            f"SemML script not found at {SEMML_PATH}. Please ensure that SemML is installed and the path is correct. Refer to the Readme for installation instructions."
        )

    output_dir = DATA_DIR / "tmp"
    output_dir.mkdir(parents=True, exist_ok=True)
    assignments_str = f"[{','.join([_assignment_to_str(a) for a in assignments])}]"
    with tempfile.TemporaryDirectory(prefix="semml-", dir=output_dir) as temp_dir:
        output_path = Path(temp_dir) / "ldba.hoa"
        command = [
            "pixi",
            "run",
            "python",
            SEMML_PATH.as_posix(),
            "--formula",
            formula,
            "--aps",
            f"{','.join(propositions)}",
            "--eligibleLetters",
            assignments_str,
            "--outputPath",
            output_path.as_posix(),
        ]
        if not use_attention_feature:
            command += ["--featureList", "260112"]

        run = subprocess.run(command, capture_output=True, text=True, check=False)
        if run.returncode != 0:
            raise RuntimeError(
                f"SemML exited with status {run.returncode}."
                f"\nStandard output:\n{run.stdout}"
                f"\nStandard error:\n{run.stderr}"
            )
        if not output_path.exists():
            raise RuntimeError(
                "SemML did not produce an output file."
                f"\nStandard output:\n{run.stdout}"
                f"\nStandard error:\n{run.stderr}"
            )
        return output_path.read_text()


def _assignment_to_str(assignment: Assignment) -> str:
    return "[" + ",".join(sorted(assignment.true_propositions)) + "]"


if __name__ == "__main__":
    f = "FG a"
    ldba = run_semml(f, ["a", "b"], Assignment.zero_or_one_propositions({"a", "b"}))
    print(ldba)
