"""Install bundled pretrained models into the run layout used by evaluation scripts."""

import argparse
import filecmp
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRETRAINED_DIR = PROJECT_ROOT / "pretrained_models"

CHECKPOINTS = (
    ("WarehouseEnv", "struct_ltl"),
    ("WarehouseEnv", "ltl2action"),
    ("WarehouseEnv", "genz_ltl"),
    ("WarehouseEnv", "deep_ltl"),
    ("ZoneEnv-NM", "struct_ltl"),
    ("ZoneEnv-NM", "ltl2action"),
    ("ZoneEnv-NM", "genz_ltl"),
    ("ZoneEnv-NM", "deep_ltl"),
)


def source(environment: str, algorithm: str) -> Path:
    return PRETRAINED_DIR / environment / f"{algorithm}.eqx"


def destination(environment: str, algorithm: str) -> Path:
    return (
        PROJECT_ROOT
        / "runs"
        / environment
        / algorithm
        / "pretrained"
        / "models.eqx"
    )


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

    missing = [
        source(environment, algorithm).relative_to(PROJECT_ROOT)
        for environment, algorithm in CHECKPOINTS
        if not source(environment, algorithm).is_file()
    ]
    if missing:
        parser.error(
            f"missing pretrained model(s): {', '.join(map(str, missing))}"
        )

    conflicts = []
    for environment, algorithm in CHECKPOINTS:
        bundled_model = source(environment, algorithm)
        target = destination(environment, algorithm)
        if target.exists() and not filecmp.cmp(
            bundled_model, target, shallow=False
        ):
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
    for environment, algorithm in CHECKPOINTS:
        bundled_model = source(environment, algorithm)
        target = destination(environment, algorithm)
        if target.exists() and filecmp.cmp(
            bundled_model, target, shallow=False
        ):
            print(f"Up to date: {target.relative_to(PROJECT_ROOT)}")
            unchanged += 1
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp")
        shutil.copy2(bundled_model, temporary)
        temporary.replace(target)
        print(f"Installed:  {target.relative_to(PROJECT_ROOT)}")
        installed += 1

    print(f"Installed {installed} model(s); {unchanged} already up to date.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
