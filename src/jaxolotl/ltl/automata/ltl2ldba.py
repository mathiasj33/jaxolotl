from collections.abc import Iterable

from jaxolotl.ltl.automata.ldba import LDBA
from jaxolotl.ltl.automata.rabinizer import run_rabinizer
from jaxolotl.utils import memory


@memory.cache
def ltl2ldba(
    formula: str,
    propositions: Iterable[str] | None = None,
) -> LDBA:
    """Converts an LTL formula to an LDBA using the rabinizer tool."""
    from jaxolotl.ltl.hoa import HOAParser  # noqa: PLC0415

    hoa = run_rabinizer(formula)
    return HOAParser(formula, hoa, propositions).parse_hoa()
