import functools
from dataclasses import dataclass
from typing import overload

from jaxolotl.ltl.logic.boolean_parser import parse


@dataclass(frozen=True)
class Assignment:
    """An assignment of propositions to truth values. Represented as a set of true propositions."""

    true_propositions: frozenset[str]

    @overload
    def __init__(self, *propositions: str) -> None: ...

    @overload
    def __init__(self, propositions: frozenset[str], /) -> None: ...

    def __init__(self, *args: str | frozenset[str]) -> None:
        if len(args) == 1 and isinstance(args[0], frozenset):
            object.__setattr__(self, "true_propositions", args[0])
        else:
            object.__setattr__(self, "true_propositions", frozenset(args))

    @staticmethod
    @functools.cache
    def all_possible_assignments(propositions: tuple[str, ...]) -> list["Assignment"]:
        """Returns all possible assignments for a given set of propositions. Guarantees a deterministic order."""
        p = propositions[0]
        rest = propositions[1:]
        if not rest:
            return [Assignment(frozenset({p})), Assignment(frozenset())]
        result = Assignment.all_possible_assignments(rest)
        result += [assignment | Assignment(frozenset({p})) for assignment in result]
        return result

    @staticmethod
    def zero_or_one_propositions(propositions: set[str]) -> list["Assignment"]:
        assignments = []
        for p in propositions:
            assignments.append(Assignment(frozenset({p})))
        assignments.append(Assignment(frozenset()))
        return assignments

    def satisfies(self, label: str | None) -> bool:
        if label is None:
            return False
        if label == "t":
            return True
        ast = parse(label)
        return ast.eval(self)

    def __repr__(self) -> str:
        return "{" + ", ".join(sorted(self.true_propositions)) + "}"

    def __len__(self):
        return len(self.true_propositions)

    def __iter__(self):
        return iter(self.true_propositions)

    def __or__(self, other: "Assignment") -> "Assignment":
        return Assignment(self.true_propositions | other.true_propositions)

    def __contains__(self, item: str) -> bool:
        return item in self.true_propositions
