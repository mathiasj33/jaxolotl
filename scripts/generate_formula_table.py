"""Generate LaTeX tables of the finite and infinite evaluation LTL specifications.

Formulas are read from the Hydra `formulas` config group (`conf/formulas/<env>/<task_set>.yaml`)
and translated token by token, so the table keeps the parenthesisation of the source formulas.
One table is written per task set (`ltl_specs_<task_set>.tex`, label `tab:ltl_specs_<task_set>`),
or a single sectioned table with `--combined`. Environment names are placed once per group,
rotated by 90 degrees, as `\\multirow` entries.

The tables need the `booktabs`, `multirow` and `graphicx` packages and the `\\event`, `\\always`
and `\\until` macros (see `MACROS`; pass `--standalone` for a compilable document).

Example:
    pixi run python scripts/generate_formula_table.py --output-dir tables
"""

import argparse
import math
import re
from pathlib import Path

from hydra import compose, initialize_config_dir

from jaxolotl.ltl.progression.ltl_lexer import LTLLexer, LTLToken, LTLTokenType
from jaxolotl.ltl.progression.ltl_parser import parse

CONF_DIR = Path(__file__).resolve().parents[1] / "conf"

DEFAULT_ENVS = [
    "letter_world",
    "zone_env",
    "franka_zone_env",
    "warehouse",
    "zone_env_nm",
]
ENV_LABELS = {
    "letter_world": "LetterWorld",
    "zone_env": "ZoneEnv",
    "zone_env_nm": "ZoneEnv-NM",
    "franka_zone_env": "FrankaZoneEnv",
    "warehouse": "Warehouse",
}
# task set -> (section title, formula ID symbol, caption of the standalone table)
SECTIONS = {
    "finite": (
        "Finite Horizon",
        r"\varphi",
        "Finite-horizon evaluation specifications.",
    ),
    "infinite": (
        "Infinite Horizon",
        r"\psi",
        "Infinite-horizon evaluation specifications.",
    ),
}

MACROS = r"""\providecommand{\event}{\mathsf{F}\,}
\providecommand{\always}{\mathsf{G}\,}
\providecommand{\until}{\mathbin{\mathsf{U}}}"""

# Rough geometry of the rotated labels: width of one character and height of one table row
# (both in pt, at \small). Used to make sure a group spans enough rows to fit its label.
CHAR_WIDTH_PT = 5.0
ROW_HEIGHT_PT = 11.0

_TOKEN_TO_TEX = {
    LTLTokenType.NOT: r"\neg",
    LTLTokenType.AND: r"\wedge",
    LTLTokenType.OR: r"\vee",
    LTLTokenType.IMPLIES: r"\rightarrow",
    LTLTokenType.EVENTUALLY: r"\event",
    LTLTokenType.ALWAYS: r"\always",
    LTLTokenType.UNTIL: r"\until",
    LTLTokenType.TRUE: r"\top",
    LTLTokenType.FALSE: r"\bot",
}


def _strip_atom_parens(tokens: list[LTLToken]) -> list[LTLToken]:
    """Drop redundant parentheses around single propositions, e.g. `F (red)` -> `F red`."""
    out: list[LTLToken] = []
    i = 0
    while i < len(tokens):
        if (
            i + 2 < len(tokens)
            and tokens[i].type == LTLTokenType.LPAREN
            and tokens[i + 1].type == LTLTokenType.VAR
            and tokens[i + 2].type == LTLTokenType.RPAREN
        ):
            out.append(tokens[i + 1])
            i += 3
        else:
            out.append(tokens[i])
            i += 1
    return out


def formula_to_tex(formula: str) -> str:
    """Translate a formula in the repo's LTL syntax to LaTeX math (without `$` delimiters)."""
    parse(formula)  # validate syntax
    parts: list[str] = []
    for token in _strip_atom_parens(LTLLexer(formula).lex()):
        if token.type == LTLTokenType.VAR:
            parts.append(r"\mathsf{" + token.value.replace("_", r"\_") + "}")
        elif token.type in (LTLTokenType.LPAREN, LTLTokenType.RPAREN):
            parts.append(token.value)
        else:
            parts.append(_TOKEN_TO_TEX[token.type])
    tex = " ".join(parts)
    return tex.replace("( ", "(").replace(" )", ")")


def load_formulas(env: str, task_set: str) -> list[str]:
    if not (CONF_DIR / "formulas" / env / f"{task_set}.yaml").exists():
        return []
    with initialize_config_dir(str(CONF_DIR), version_base=None):
        cfg = compose(overrides=[f"+formulas={env}/{task_set}"])
    return list(cfg.formulas)


def _rows_needed(text: str) -> int:
    return math.ceil(len(text) * CHAR_WIDTH_PT / ROW_HEIGHT_PT)


def _split_label(label: str) -> list[str]:
    """Split a label into two lines at the camel-case/hyphen boundary closest to its middle."""
    cuts = [m.start() for m in re.finditer(r"(?<=[a-z])(?=[A-Z])|(?<=-)", label)]
    if not cuts:
        return [label]
    cut = min(cuts, key=lambda c: abs(c - len(label) / 2))
    return [label[:cut], label[cut:]]


def env_label_cell(label: str, num_rows: int) -> tuple[str, int]:
    """Return the rotated multirow cell for a group and the number of rows it must span.

    Labels that are too tall for the group are split over two lines and, if still too tall,
    the group is padded with empty rows.
    """
    lines = [label]
    if _rows_needed(label) > num_rows:
        lines = _split_label(label)
    span = max(num_rows, *(_rows_needed(line) for line in lines))
    text = (
        lines[0]
        if len(lines) == 1
        else r"\begin{tabular}{@{}c@{}}" + r"\\".join(lines) + r"\end{tabular}"
    )
    return rf"\multirow{{{span}}}{{*}}{{\rotatebox[origin=c]{{90}}{{{text}}}}}", span


def build_table(
    envs: list[str], task_sets: list[str], caption: str, label: str, formula_width: str
) -> str:
    lines = [
        "% Generated by scripts/generate_formula_table.py; requires booktabs, multirow, graphicx",
        "% and the \\event, \\always and \\until macros.",
        r"\begin{table}[ht]",
        rf"\caption{{{caption}}}",
        r"\centering",
        r"\small",
        rf"\label{{{label}}}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{c l p{{{formula_width}}}}}",
        r"\toprule",
        r" & {ID} & {LTL Formula} \\",
    ]
    for task_set in task_sets:
        title, symbol, _ = SECTIONS[task_set]
        lines.append(r"\midrule")
        if len(task_sets) > 1:
            lines += [rf"\multicolumn{{3}}{{c}}{{\textit{{{title}}}}} \\", r"\midrule"]
        index = 1
        groups = [(env, load_formulas(env, task_set)) for env in envs]
        groups = [(env, formulas) for env, formulas in groups if formulas]
        for g, (env, formulas) in enumerate(groups):
            if g > 0:
                lines.append(r"\midrule")
            cell, span = env_label_cell(ENV_LABELS.get(env, env), len(formulas))
            for i, formula in enumerate(formulas):
                first = cell if i == 0 else ""
                lines.append(
                    rf"{first} & ${symbol}_{{{index}}}$ & ${formula_to_tex(formula)}$ \\"
                )
                index += 1
            lines += [r" & & \\"] * (span - len(formulas))
    lines += [r"\bottomrule", r"\end{tabular}", "}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def wrap_standalone(table: str) -> str:
    return "\n".join(
        [
            r"\documentclass{article}",
            r"\usepackage[margin=2cm]{geometry}",
            r"\usepackage{amsmath,amssymb,booktabs,multirow,graphicx}",
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
        help="Formula config directories under conf/formulas, in table order.",
    )
    parser.add_argument(
        "--task-sets",
        nargs="+",
        choices=list(SECTIONS),
        default=list(SECTIONS),
        help="Task sets to tabulate.",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Write all task sets as sections of a single table instead of one table each.",
    )
    parser.add_argument("--formula-width", default="12cm")
    parser.add_argument(
        "--standalone",
        action="store_true",
        help="Wrap each table in a compilable document with the required packages and macros.",
    )
    args = parser.parse_args()

    if args.combined:
        tables = [
            (
                "ltl_specs",
                args.task_sets,
                "Complete list of evaluation specifications.",
            )
        ]
    else:
        tables = [
            (f"ltl_specs_{task_set}", [task_set], SECTIONS[task_set][2])
            for task_set in args.task_sets
        ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, task_sets, caption in tables:
        table = build_table(
            args.envs, task_sets, caption, f"tab:{name}", args.formula_width
        )
        if args.standalone:
            table = wrap_standalone(table)
        output = args.output_dir / f"{name}.tex"
        output.write_text(table)
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
