"""Install bundled pretrained models into the run layout used by evaluation scripts."""

import argparse
import filecmp
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRETRAINED_DIR = PROJECT_ROOT / "pretrained_models"


def source(environment: str, algorithm: str) -> Path:
    return PRETRAINED_DIR / environment / algorithm / "pretrained" / "models"


def destination(environment: str, algorithm: str) -> Path:
    return PROJECT_ROOT / "runs" / environment / algorithm / "pretrained" / "models"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Copy bundled pretrained models to their evaluation run paths."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite destination models that differ from the bundled checkpoints",
    )
    args = parser.parse_args()

    checkpoints: list[tuple[str, str]] = []
    for env_folder in PRETRAINED_DIR.iterdir():
        if not env_folder.is_dir():
            continue
        for alg_folder in env_folder.iterdir():
            if not alg_folder.is_dir():
                continue
            if (alg_folder / "pretrained" / "models").exists():
                checkpoints.append((env_folder.name, alg_folder.name))

    conflicts = []
    for environment, algorithm in checkpoints:
        bundled_model = source(environment, algorithm)
        target = destination(environment, algorithm)
        if target.exists() and not filecmp.cmp(bundled_model, target, shallow=False):
            conflicts.append(target.relative_to(PROJECT_ROOT))

    if conflicts and not args.force:
        formatted = "\n  ".join(str(path) for path in conflicts)
        parser.error(
            "refusing to overwrite differing model(s):\n"
            f"  {formatted}\n"
            "rerun with --force to replace them"
        )

    installed = 0
    unchanged = 0
    for environment, algorithm in checkpoints:
        bundled_model = source(environment, algorithm)
        target = destination(environment, algorithm)
        if target.exists() and filecmp.cmp(bundled_model, target, shallow=False):
            print(f"Up to date: {target.relative_to(PROJECT_ROOT)}")
            unchanged += 1
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(bundled_model, target)
        print(f"Installed:  {target.relative_to(PROJECT_ROOT)}")
        installed += 1

    print(f"Installed {installed} model(s); {unchanged} already up to date.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
