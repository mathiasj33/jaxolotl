"""Plot evaluation or training curves for one environment."""

import logging
from functools import partial
from pathlib import Path

import hydra
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import ticker
from omegaconf import DictConfig

from jaxolotl.utils.plot_utils import smooth
from jaxolotl.utils.stats import student_t_ci

logger = logging.getLogger(__name__)


def _load_eval_data(path: Path, smooth_radius: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Eval data not found: {path}")
    df = pd.read_csv(path).sort_values(by=["seed", "timestep"])
    df["success"] = df.groupby("seed")["metric"].transform(
        lambda values: smooth(values, radius=smooth_radius)
    )
    df["length"] = df.groupby("seed")["length"].transform(
        lambda values: smooth(values, radius=smooth_radius)
    )
    return df


def _load_training_data(path: Path, bin_size: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Training data not found: {path}")
    df = pd.read_csv(path, engine="pyarrow")
    df["bin"] = df["timestep"] // bin_size
    df = (
        df.groupby(["seed", "bin"])[["return", "length", "curriculum_stage", "success"]]
        .mean()
        .reset_index()
    )
    df["timestep"] = df["bin"] * bin_size
    return df


def _eval_path(env: str, algorithm: str, run: str) -> Path:
    path = Path("runs") / env / algorithm / run / "eval" / "checkpoints.csv"
    return path


def _load_data(cfg: DictConfig) -> pd.DataFrame:
    dfs = []
    for algorithm in cfg.algorithms:
        for run in cfg.runs:
            run_dir = Path("runs") / cfg.env.name / algorithm / run
            if cfg.type == "eval":
                df = _load_eval_data(
                    _eval_path(cfg.env.name, algorithm, run), cfg.smooth_radius
                )
            elif cfg.type == "training":
                df = _load_training_data(run_dir / "logs.csv", cfg.bin_size)
            else:
                raise ValueError("type must be 'eval' or 'training'")
            name = f"{algorithm}/{run}" if len(cfg.algorithms) > 1 else run
            df["name"] = name
            dfs.append(df)

    if not dfs:
        raise ValueError("algorithms and runs must both contain at least one value")
    return pd.concat(dfs, ignore_index=True)


def _student_t_errorbar(
    values: pd.Series,
    *,
    confidence_level: float,
    bounds: tuple[float, float] | None = None,
) -> tuple[float, float]:
    interval = student_t_ci(
        values,
        confidence_level=confidence_level,
        bounds=bounds,
    )
    return interval.lower, interval.upper


def _millions_formatter(value: float, _: float) -> str:
    return f"{value / 1e6:g}"


def _plot_metric(  # noqa: PLR0913
    df: pd.DataFrame,
    *,
    metric: str,
    title: str,
    ylabel: str,
    ax,
    ci_method: str,
    confidence_level: float,
    bounds: tuple[float, float] | None = None,
    legend: bool = False,
) -> None:
    if ci_method == "student_t":
        errorbar = partial(
            _student_t_errorbar, confidence_level=confidence_level, bounds=bounds
        )
    elif ci_method == "bootstrap":
        errorbar = ("ci", confidence_level * 100)
    else:
        raise ValueError(f"Unknown confidence interval method: {ci_method}")
    sns.lineplot(
        data=df,
        x="timestep",
        y=metric,
        hue="name",
        errorbar=errorbar,  # type: ignore
        ax=ax,
        legend=legend,
    )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(_millions_formatter))


def _plot(cfg: DictConfig, df: pd.DataFrame):
    if cfg.type == "eval":
        metrics = [
            ("success", "Average Success Rate", "Success Rate", (0.0, 1.0)),
            ("length", "Average Episode Length", "Length", None),
        ]
    else:
        metrics = [
            ("success", "Average Success Rate", "Success Rate", (0.0, 1.0)),
            ("length", "Average Episode Length", "Length", None),
            ("curriculum_stage", "Curriculum Stage", "Curriculum Stage", None),
        ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    for index, (metric, title, ylabel, bounds) in enumerate(metrics):
        _plot_metric(
            df,
            metric=metric,
            title=title,
            ylabel=ylabel,
            ax=axes[index],
            ci_method=cfg.confidence_interval.method,
            confidence_level=cfg.confidence_interval.confidence_level,
            bounds=bounds,
            legend=index == len(metrics) - 1,
        )

    sns.move_legend(axes[-1], "upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    return fig


@hydra.main(version_base="1.3", config_path="../../conf", config_name="plot_curves")
def main(cfg: DictConfig) -> None:
    sns.set_theme(style="whitegrid")
    fig = _plot(cfg, _load_data(cfg))

    if cfg.save:
        output_path = Path("plots") / f"{cfg.type}_curves.pdf"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        logger.info("Saved plot to %s", output_path)
    else:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
