import subprocess
from pathlib import Path

import pytest

from jaxolotl.ltl.automata import semml
from jaxolotl.ltl.logic.assignment import Assignment


def _output_path(command: list[str]) -> Path:
    return Path(command[command.index("--outputPath") + 1])


def test_run_semml_uses_unique_output_paths_and_allows_stderr(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output_paths: list[Path] = []

    def run(command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        del kwargs
        output_path = _output_path(command)
        assert not output_path.exists()
        output_path.write_text("HOA: v1\n--BODY--\n--END--\n")
        output_paths.append(output_path)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="warning")

    monkeypatch.setattr(semml, "DATA_DIR", tmp_path)
    monkeypatch.setattr(semml.subprocess, "run", run)

    assignments = [Assignment(frozenset())]
    first = semml.run_semml("F a", ["a"], assignments)
    second = semml.run_semml("F a", ["a"], assignments)

    assert first == second == "HOA: v1\n--BODY--\n--END--\n"
    assert output_paths[0] != output_paths[1]
    assert all(not path.exists() for path in output_paths)


def test_run_semml_reports_nonzero_exit_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output_paths: list[Path] = []

    def run(command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        del kwargs
        output_path = _output_path(command)
        output_path.write_text("partial output")
        output_paths.append(output_path)
        return subprocess.CompletedProcess(command, 7, stdout="details", stderr="error")

    monkeypatch.setattr(semml, "DATA_DIR", tmp_path)
    monkeypatch.setattr(semml.subprocess, "run", run)

    with pytest.raises(RuntimeError, match="status 7"):
        semml.run_semml("F a", ["a"], [Assignment(frozenset())])

    assert all(not path.exists() for path in output_paths)


def test_run_semml_rejects_missing_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def run(command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        del kwargs
        return subprocess.CompletedProcess(command, 0, stdout="details", stderr="error")

    monkeypatch.setattr(semml, "DATA_DIR", tmp_path)
    monkeypatch.setattr(semml.subprocess, "run", run)

    with pytest.raises(RuntimeError, match="did not produce"):
        semml.run_semml("F a", ["a"], [Assignment(frozenset())])
