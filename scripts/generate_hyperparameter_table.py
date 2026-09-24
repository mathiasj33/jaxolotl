"""Generate LaTeX tables of the training hyperparameters of each method, one table per environment.

Hyperparameters are read from the composed Hydra training config (`train` with `env=<env>` and
`alg=<alg>`, i.e. `conf/experiment/<env>/<alg>.yaml`), and the maximum episode length from the
environment's default parameters. An environment family (e.g. Conveyor-1 to -10 and
ConveyorSimple-1 to -32) gets a single table, in which values that differ between its
environments are shown as a range. Adjacent methods sharing a value are merged into one `\\spanval` cell, and
settings that do not apply to a method are marked N/A. Rows and sections that apply to none of
the tabulated methods are dropped.

The tables need the `booktabs` and `graphicx` packages and the `\\spanval` and `\\linefill`
macros (see `MACROS`; pass `--standalone` for a compilable document).

Example:
    pixi run python scripts/generate_hyperparameter_table.py --output-dir ../results_final/tables
"""

import argparse
import functools
import math
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import jaxolotl

CONF_DIR = Path(__file__).resolve().parents[1] / "conf"

# (environment label, experiment config directories of its variants)
Group = tuple[str, list[str]]

# family -> environments sharing one table
FAMILIES: dict[str, list[Group]] = {
    "letter_world": [("LetterWorld", ["letter_world"])],
    "warehouse": [("Warehouse", ["warehouse"])],
    "zone_env": [("ZoneEnv", ["zone_env"]), ("ZoneEnv-NM", ["zone_env_nm"])],
    "franka_zone_env": [
        ("FrankaZoneEnv", [f"franka_zone_env_{k}" for k in range(8, 13)])
    ],
    "conveyor": [
        ("Conveyor", [f"conveyor_world_{k}" for k in range(1, 11)]),
        (
            "ConveyorSimple",
            [f"conveyor_world_simple_{k}" for k in (1, 2, 4, 8, 16, 32)],
        ),
    ],
}
DEFAULT_ENVS = list(FAMILIES)

# column label -> alg configs, in order of preference; the first one configured for all variants
# of a family is tabulated. The GenZ-LTL variants without observation reduction
# (`genz_ltl_no_obs_red`, `genz_ltl_flat`) stand in for GenZ-LTL where it is not configured.
# A method without configs (e.g. GCRL-LTL on Warehouse, which it does not support) is left out.
METHODS = {
    "LTL2Action": ["ltl2action"],
    "SemLTL": ["sem_ltl"],
    "DeepLTL": ["deep_ltl"],
    "StructLTL": ["struct_ltl"],
    "GCRL-LTL": ["gcrl_ltl"],
    "GenZ-LTL": ["genz_ltl", "genz_ltl_no_obs_red", "genz_ltl_flat"],
}

MACROS = r"""\providecommand{\linefill}{\leavevmode\leaders\hrule height \dimexpr0.5ex+0.2pt\relax depth \dimexpr-0.5ex+0.2pt\relax\hfill\kern0pt}
\providecommand{\spanval}[1]{\linefill\enspace #1\enspace\linefill}"""

ACTIVATIONS = {"relu": "ReLU", "tanh": "Tanh", "softplus": "Softplus"}

Config = dict[str, Any]
Getter = str | tuple[str, ...] | Callable[[Config], Any]


def _lookup(cfg: Config, key: str) -> Any:
    node: Any = cfg
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _sequence_model(cfg: Config) -> str | None:
    if cfg["alg"]["name"] == "deep_ltl":
        return "GRU"  # DeepLTLModel hard-codes a GRU sequence encoder
    encoder = _lookup(cfg, "model.sequence.encoder")
    if encoder == "attention":
        target = _lookup(cfg, "model.sequence.attention._target_")
        return {"ALiBiAttention": "ALiBi attention"}.get(
            target.rsplit(".", 1)[-1], "Attention"
        )
    return encoder.upper() if encoder else None


TASK_SETS = ("finite", "infinite")
TRAIN_LENGTH = "Max episode length"
EVAL_LENGTH = "Max episode length (eval, {})"

# (category, [(hyperparameter, getter)]). A getter is a config key, alternative keys (the first
# present one is used) or a function of the config; a missing value means N/A.
SECTIONS: list[tuple[str, list[tuple[str, Getter]]]] = [
    (
        "PPO",
        [
            ("Total environment steps", "rl_alg.total_timesteps"),
            ("Environments", "rl_alg.num_envs"),
            ("Steps/update", "rl_alg.num_steps"),
            ("Minibatches", "rl_alg.num_minibatches"),
            ("Update epochs", "rl_alg.update_epochs"),
            (r"Discount ($\gamma$)", "rl_alg.gamma"),
            (r"GAE lambda ($\lambda$)", "rl_alg.gae_lambda"),
            ("Clip epsilon", "rl_alg.clip_eps"),
            ("Entropy coef.", "rl_alg.ent_coef"),
            (r"Value func.\ coef.", "rl_alg.vf_coef"),
            ("Learning rate", "rl_alg.lr"),
            ("Max grad norm", "rl_alg.max_grad_norm"),
            ("Adam epsilon", "rl_alg.adam_eps"),
        ],
    ),
    (
        "Safe PPO",
        [
            (r"Cost discount ($\gamma_c$)", "rl_alg.cost_gamma"),
            (r"Cost value func.\ coef.", "rl_alg.cost_vf_coef"),
            ("Lagrangian coef.", "rl_alg.lag_coef"),
            ("Target cost", "rl_alg.target_cost"),
            ("Min Lagrangian", "rl_alg.min_lag"),
            ("Max Lagrangian", "rl_alg.max_lag"),
            ("Target KL", "rl_alg.target_kl"),
        ],
    ),
    (
        "Curriculum",
        [
            ("Episode window", "curriculum.window"),
            ("Adoption prob.", "curriculum.adopt_prob"),
            ("Min coverage", "curriculum.min_coverage"),
        ],
    ),
    (
        "Env Net",
        [
            ("Hidden sizes", "model.env_net.hidden_sizes"),
            ("Channels", "model.env_net.channels"),
            ("Kernel size", "model.env_net.kernel_size"),
            ("Output size", "model.env_net.out_size"),
            ("Activation", "model.env_net.activation"),
        ],
    ),
    (
        "Actor",
        [
            ("Hidden sizes", "model.actor.hidden_sizes"),
            ("Activation", ("model.actor.hidden_activation", "model.actor.activation")),
            ("Output activation", "model.actor.output_activation"),
        ],
    ),
    (
        "Critic",
        [
            ("Hidden sizes", "model.critic.hidden_sizes"),
            ("Activation", "model.critic.activation"),
        ],
    ),
    (
        "Cost Critic",
        [
            ("Hidden sizes", "model.cost_critic.hidden_sizes"),
            ("Activation", "model.cost_critic.activation"),
        ],
    ),
    (
        "Lagrangian Net",
        [
            ("Hidden sizes", "model.lagrangian.hidden_sizes"),
            ("Activation", "model.lagrangian.activation"),
        ],
    ),
    (
        "LTL Encoder",
        [
            (
                "Embedding dim.",
                (
                    "model.embedding_dim",
                    "model.sequence.embedding_dim",
                    "model.rgcn.embedding_dim",
                ),
            ),
            ("RGCN layers", "model.rgcn.num_layers"),
            ("RGCN activation", "model.rgcn.activation"),
            ("Semantic input size", "model.semantic.embedding_size"),
            ("Sequence model", _sequence_model),
            ("Deep sets hidden sizes", "model.sequence.deep_sets.hidden_sizes"),
            ("Deep sets output size", "model.sequence.deep_sets.out_size"),
            ("Deep sets activation", "model.sequence.deep_sets.activation"),
            ("Clause net output size", "model.sequence.clause_mlp.out_size"),
            ("Disjunct net output size", "model.sequence.disjunct_mlp.out_size"),
            ("Attention hidden dim.", "model.sequence.attention.hidden_dim"),
            ("ALiBi slope", "model.sequence.attention.alibi_slope"),
        ],
    ),
    (
        "GCVF",
        [
            ("Samples", "gcvf_training.num_samples"),
            ("Batch size", "gcvf_training.batch_size"),
            ("Learning rate", "gcvf_training.lr"),
            ("Epochs", "gcvf_training.epochs"),
            ("Environments", "gcvf_training.num_envs"),
            ("Steps/env", "gcvf_training.steps_per_env"),
        ],
    ),
    (
        "Environment",
        [
            (TRAIN_LENGTH, "env.max_steps_in_episode"),
            *(
                (
                    EVAL_LENGTH.format(task_set),
                    f"env.eval_max_steps_in_episode.{task_set}",
                )
                for task_set in TASK_SETS
            ),
        ],
    ),
]


def _get(cfg: Config, getter: Getter) -> Any:
    if callable(getter):
        return getter(cfg)
    for key in (getter,) if isinstance(getter, str) else getter:
        value = _lookup(cfg, key)
        if value is not None:
            return value
    return None


def _as_number(value: Any) -> int | float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return value
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _is_small_power_of_ten(value: float) -> bool:
    exponent = math.log10(abs(value))
    return exponent <= -3 and math.isclose(exponent, round(exponent))


def _needs_scientific(value: float) -> bool:
    if value == 0 or (float(value).is_integer() and abs(value) < 1e5):
        return False
    return abs(value) >= 1e5 or abs(value) < 1e-3 or _is_small_power_of_ten(value)


def format_number(value: float, scientific: bool | None = None) -> str:
    """Format a number as in the paper: `1,000`, `0.998`, `$3 \\times 10^{-4}$`, `$-0.1$`.
    `scientific` forces the notation (so that a row uses one notation throughout); by default it
    is chosen from the value."""
    if scientific is None:
        scientific = _needs_scientific(value)
    if value == 0:
        return "0"
    if scientific:
        mantissa, exponent = f"{value:e}".split("e")
        mantissa = mantissa.rstrip("0").rstrip(".")
        power = f"10^{{{int(exponent)}}}"
        if mantissa in ("1", "-1"):
            text = power if mantissa == "1" else f"-{power}"
        else:
            text = rf"{mantissa} \times {power}"
    elif float(value).is_integer() and abs(value) < 1e5:
        text = f"{int(value):,}"
    else:
        text = repr(float(value))
    if isinstance(value, float) and "." not in text and "^" not in text:
        text += ".0"  # keep coefficients such as `1.0` recognisable as floats
    return f"${text}$" if "^" in text or value < 0 else text


def format_value(value: Any, scientific: bool | None = None) -> str:
    """Render a single config value as table cell content."""
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, list):
        return "[" + ", ".join(format_value(v, scientific) for v in value) + "]"
    if isinstance(value, str):
        match = re.fullmatch(r"\$\{act:(\w+)\}", value)
        if match:
            return ACTIVATIONS.get(match[1], match[1].capitalize())
    number = _as_number(value)
    if number is not None:
        return format_number(number, scientific)
    return str(value)


def _variant_ks(variants: list[str]) -> list[int] | None:
    """The k of each variant (`<env>_<k>`), or None if the variants are not indexed by k."""
    matches = [re.search(r"_(\d+)$", env) for env in variants]
    if len(variants) < 2 or not all(matches):
        return None
    return [int(m[1]) for m in matches]  # type: ignore[index]


def _evaluate(coeffs: list[int], k: int) -> int:
    return sum(c * k ** (len(coeffs) - 1 - i) for i, c in enumerate(coeffs))


def _fit_polynomial(
    ks: list[int], values: list[int], max_degree: int | None = None
) -> list[int] | None:
    """Integer coefficients (highest degree first) of the lowest-degree polynomial through all
    points. By default the degree is capped so that at least one point validates the fit."""
    max_degree = len(ks) - 2 if max_degree is None else max_degree
    for degree in range(max_degree + 1):
        vander = np.vander(ks[: degree + 1], degree + 1).astype(float)
        coeffs = np.linalg.solve(vander, values[: degree + 1])
        rounded = [round(c) for c in coeffs]
        if all(_evaluate(rounded, k) == v for k, v in zip(ks, values, strict=True)):
            return rounded
    return None


def _polynomial_tex(coeffs: list[int]) -> str:
    terms = []
    for i, c in enumerate(coeffs):
        power = len(coeffs) - 1 - i
        if c == 0:
            continue
        monomial = {0: "", 1: "k"}.get(power, f"k^{{{power}}}")
        magnitude = "" if abs(c) == 1 and power > 0 else str(abs(c))
        terms.append(("-" if c < 0 else "+", magnitude + monomial))
    if not terms:
        return "0"
    text = ("-" if terms[0][0] == "-" else "") + terms[0][1]
    return text + "".join(f" {sign} {term}" for sign, term in terms[1:])


def fit_expression(ks: list[int], values: list[float]) -> str | None:
    """LaTeX expression in k that matches the values exactly: a polynomial with integer
    coefficients, or such a polynomial clamped from below (`\\max(c, p(k))`)."""
    if not all(float(v).is_integer() for v in values):
        return None
    ints = [int(v) for v in values]
    if (coeffs := _fit_polynomial(ks, ints)) is not None:
        return _polynomial_tex(coeffs)
    floor = min(ints)
    above = [(k, v) for k, v in zip(ks, ints, strict=True) if v > floor]
    clamped = [k for k, v in zip(ks, ints, strict=True) if v == floor]
    if len(above) < 2 or len(clamped) < 2:
        return None
    above_ks, above_values = map(list, zip(*above, strict=True))
    coeffs = _fit_polynomial(above_ks, above_values, max_degree=len(above) - 1)
    # The clamped points validate the fit: the polynomial must not exceed the floor there.
    if coeffs is None or any(_evaluate(coeffs, k) > floor for k in clamped):
        return None
    return rf"\max({floor}, {_polynomial_tex(coeffs)})"


def _describe(
    values: list[Any], ks: list[int] | None, scientific: bool | None
) -> tuple[str, str]:
    """Render one hyperparameter across the variants of one environment, with the kind of
    rendering used (value, function, range or list)."""
    distinct = {repr(v): v for v in values}
    if len(distinct) == 1:
        return format_value(values[0], scientific), "value"
    numbers = [_as_number(v) for v in values]
    if any(n is None for n in numbers):
        return ", ".join(format_value(v, scientific) for v in distinct.values()), "list"
    if ks is not None and (expression := fit_expression(ks, numbers)) is not None:
        return f"${expression}$", "function"
    low, high = (format_number(n, scientific) for n in (min(numbers), max(numbers)))
    return f"{low}--{high}", "range"


Cell = tuple[str | None, set[str]]
# One hyperparameter of one method: (environment label, variant ks, values) per environment.
Raw = list[tuple[str, list[int] | None, list[Any]]]


def _row_notation(raws: list[Raw]) -> bool | None:
    """Scientific notation for every number in a row if any of them needs it, so that a row
    does not mix notations; otherwise each number picks its own."""
    numbers = [
        n
        for groups in raws
        for _, _, values in groups
        for v in values
        if (n := _as_number(v)) is not None
    ]
    return True if any(_needs_scientific(n) for n in numbers) else None


def aggregate(groups: Raw, scientific: bool | None = None) -> Cell:
    """Combine one hyperparameter across the environments (label, variant ks, values) of a
    family. Environments that disagree are listed as `value (label)`."""
    values = [v for _, _, group_values in groups for v in group_values]
    if all(v is None for v in values):
        return None, set()
    if any(v is None for v in values):
        raise ValueError(f"Hyperparameter set for only some variants: {values}")
    described = [(label, *_describe(vs, ks, scientific)) for label, ks, vs in groups]
    kinds = {kind for _, _, kind in described}
    if len({text for _, text, _ in described}) == 1:
        return described[0][1], kinds
    return ", ".join(f"{text} ({label})" for label, text, _ in described), kinds


def load_config(env: str, alg: str) -> Config | None:
    if not (CONF_DIR / "experiment" / env / f"{alg}.yaml").exists():
        return None
    cfg = compose("train", overrides=[f"env={env}", f"alg={alg}"])
    container = OmegaConf.to_container(cfg, resolve=False)
    assert isinstance(container, dict)
    _, env_params = jaxolotl.make(cfg.env.name)
    container["env"]["max_steps_in_episode"] = env_params.max_steps_in_episode
    container["env"]["eval_max_steps_in_episode"] = {
        task_set: eval_max_steps(env, task_set) for task_set in TASK_SETS
    }
    return container


@functools.cache
def eval_max_steps(env: str, task_set: str) -> int | None:
    """Episode cap during evaluation on a task set: the environment's default, overridden by the
    `env_params` of the task set's formula config (as in `scripts/eval/eval.py`)."""
    if not (CONF_DIR / "formulas" / env / f"{task_set}.yaml").exists():
        return None
    cfg = compose(overrides=[f"+env={env}", f"+formulas={env}/{task_set}"])
    env_params = OmegaConf.to_container(cfg.env_params) if "env_params" in cfg else {}
    assert isinstance(env_params, dict)
    _, params = jaxolotl.make(cfg.env.name, **env_params)
    return params.max_steps_in_episode


def _episode_length_rows(
    rows: list[tuple[str, list[Cell]]],
) -> list[tuple[str, list[Cell]]]:
    """Keep evaluation episode caps only where they differ from the training cap, merging the
    task sets if their caps agree."""
    evals = {EVAL_LENGTH.format(task_set) for task_set in TASK_SETS}
    texts = {label: [text for text, _ in cells] for label, cells in rows}
    train = texts.get(TRAIN_LENGTH)
    shown = [label for label in texts if label in evals and texts[label] != train]
    merged = len(shown) > 1 and all(texts[label] == texts[shown[0]] for label in shown)
    result = []
    for label, cells in rows:
        if label in evals and (label not in shown or (merged and label != shown[0])):
            continue
        if label == TRAIN_LENGTH and shown:
            result.append((f"{TRAIN_LENGTH} (train)", cells))
        elif label in evals and merged:
            result.append(("Max episode length (eval)", cells))
        else:
            result.append((label, cells))
    return result


def method_values(groups: list[Group], algs: list[str]) -> list[Raw] | None:
    """Tabulated values of the first alg with configs for all variants, or None if there is none."""
    for alg in algs:
        cfgs = [[load_config(env, alg) for env in variants] for _, variants in groups]
        if all(cfg is not None for group_cfgs in cfgs for cfg in group_cfgs):
            return [
                [
                    (
                        label,
                        _variant_ks(variants),
                        [_get(c, getter) for c in group_cfgs],
                    )
                    for (label, variants), group_cfgs in zip(groups, cfgs, strict=True)
                ]
                for _, rows in SECTIONS
                for _, getter in rows
            ]
    return None


def _cells(values: list[str]) -> str:
    """Join a row's values, merging runs of equal values into one `\\spanval` cell."""
    cells = []
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[j + 1] == values[i]:
            j += 1
        if j > i:
            cells.append(
                rf"\multicolumn{{{j - i + 1}}}{{c}}{{\spanval{{{values[i]}}}}}"
            )
        else:
            cells.append(values[i])
        i = j + 1
    return " & ".join(cells)


def _variant_note(groups: list[Group], kinds: set[str]) -> str:
    covered = [
        rf"{label}-$k$ for $k \in \{{{', '.join(map(str, ks))}\}}$"
        for label, variants in groups
        if (ks := _variant_ks(variants)) is not None
    ]
    note = f" The table covers {' and '.join(covered)}." if covered else ""
    if "function" in kinds:
        note += " Hyperparameters that vary with $k$ are given as functions of $k$."
    if "range" in kinds:
        note += (
            " Ranges (a--b) give hyperparameters that vary across these environments."
        )
    return note


def build_table(family: str) -> str:
    groups = FAMILIES.get(family)
    if groups is None:
        with initialize_config_dir(str(CONF_DIR), version_base=None):
            name = compose("train", overrides=[f"env={family}"]).env.name
        groups = [(name, [family])]
    label = " and ".join(group_label for group_label, _ in groups)

    with initialize_config_dir(str(CONF_DIR), version_base=None):
        columns = {
            name: values
            for name, algs in METHODS.items()
            if (values := method_values(groups, algs)) is not None
        }
    if not columns:
        raise ValueError(f"No method configs found for {family}.")
    names = list(columns)

    body = []
    kinds: set[str] = set()
    index = 0
    for category, rows in SECTIONS:
        shown = []
        for hyperparameter, _ in rows:
            raws = [columns[name][index] for name in names]
            index += 1
            scientific = _row_notation(raws)
            cells = [aggregate(raw, scientific) for raw in raws]
            if any(text is not None for text, _ in cells):
                shown.append((hyperparameter, cells))
        if category == "Environment":
            shown = _episode_length_rows(shown)
        if not shown:
            continue
        section = []
        for hyperparameter, cells in shown:
            kinds.update(*(cell_kinds for _, cell_kinds in cells))
            row = _cells(["N/A" if text is None else text for text, _ in cells])
            section.append(f"& {hyperparameter} & {row} \\\\")
        head = (
            category
            if len(section) == 1
            else rf"\multirow{{{len(section)}}}{{*}}{{{category}}}"
        )
        body += [r"\midrule", head, *section]

    caption = (
        f"Training hyperparameters for {label}. Values spanning multiple columns are shared by"
        " those methods; N/A marks settings that do not apply to a method."
        + _variant_note(groups, kinds)
    )
    lines = [
        "% Generated by scripts/generate_hyperparameter_table.py; requires booktabs, multirow,",
        "% graphicx and the \\spanval and \\linefill macros.",
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{tab:hyperparameters_{family}}}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{ll{'c' * len(names)}}}",
        r"\toprule",
        "Category & Hyperparameter & " + " & ".join(names) + r" \\",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        "}",
        r"\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def wrap_standalone(table: str) -> str:
    return "\n".join(
        [
            r"\documentclass{article}",
            r"\usepackage[margin=2cm]{geometry}",
            r"\usepackage{amsmath,booktabs,multirow,graphicx}",
            MACROS,
            r"\begin{document}",
            table,
            r"\end{document}",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--envs",
        nargs="+",
        default=DEFAULT_ENVS,
        help=(
            f"Environment families ({', '.join(FAMILIES)}) or single experiment config"
            " directories under conf/experiment; one table each."
        ),
    )
    parser.add_argument(
        "--standalone",
        action="store_true",
        help="Wrap each table in a compilable document with the required packages and macros.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for family in args.envs:
        table = build_table(family)
        if args.standalone:
            table = wrap_standalone(table)
        output = args.output_dir / f"hyperparameters_{family}.tex"
        output.write_text(table)
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
