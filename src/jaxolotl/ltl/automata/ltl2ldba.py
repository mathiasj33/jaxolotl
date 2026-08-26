from collections.abc import Iterable
from typing import Literal

from jaxolotl.ltl.automata.ldba import LDBA
from jaxolotl.ltl.automata.rabinizer import run_rabinizer
from jaxolotl.ltl.automata.semml import run_semml
from jaxolotl.ltl.logic.assignment import Assignment
from jaxolotl.utils.utils import memory

LDBABackend = Literal["rabinizer", "semml"]


@memory.cache
def ltl2ldba(
    formula: str,
    propositions: Iterable[str],
    assignments: Iterable[Assignment],
    backend: LDBABackend = "rabinizer",
) -> LDBA:
    """Converts an LTL formula to an LDBA using the rabinizer tool."""
    from jaxolotl.ltl.hoa import HOAParser  # noqa: PLC0415

    if backend == "rabinizer":
        hoa = run_rabinizer(formula)
    elif backend == "semml":
        hoa = run_semml(formula, propositions, assignments)
    return HOAParser(formula, hoa, propositions).parse_hoa()
