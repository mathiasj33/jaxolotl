import time
from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import ticker

sns.set_theme(style="darkgrid")


def load_df(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, engine="pyarrow")
    bin_size = 4096 * 16
    df["bin"] = df["timestep"] // bin_size
    if "positive_return" not in df.columns:
        df["positive_return"] = df["return"]

    avg_data = (
        df.groupby(["seed", "bin"])[
            ["return", "length", "curriculum_stage", "positive_return"]
        ]
        .mean()
        .reset_index()
    )

    avg_data["timestep"] = avg_data["bin"] * bin_size

    avg_data["name"] = Path(path).parent.parent.name
    return avg_data


log_files = ["runs/ZoneEnv-NM/struct_ltl/main/logs.csv"]

dfs = [load_df(path) for path in log_files]
df = pd.concat(dfs, ignore_index=True)


fig, axes = plt.subplots(1, 3, figsize=(20, 5))

start = time.time()
sns.lineplot(
    data=df,
    x="timestep",
    y="positive_return",
    hue="name",
    ax=axes[0],
    legend=False,
    errorbar="sd",
)

axes[0].set_title("Average SR")
axes[0].set_ylabel("SR")


def millions_formatter(x, pos):
    return f"{x / 1e6:g}"  # :g removes unnecessary trailing zeros


# 2. Apply the formatter to the Y axis
axes[0].xaxis.set_major_formatter(ticker.FuncFormatter(millions_formatter))

sns.lineplot(
    data=df,
    x="timestep",
    y="length",
    hue="name",
    ax=axes[1],
    legend=False,
    errorbar="sd",
)
axes[1].set_title("Average Episode Length")
axes[1].set_ylabel("Length")

sns.lineplot(
    data=df,
    x="timestep",
    y="curriculum_stage",
    hue="name",
    ax=axes[2],
    errorbar="sd",
)
axes[2].set_title("Curriculum Stage")
axes[2].set_ylabel("Curriculum Stage")

# Move the legend outside the plot area
sns.move_legend(axes[2], "upper left", bbox_to_anchor=(1, 1))

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()
