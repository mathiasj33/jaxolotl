import random

from jaxolotl.alg.curriculum.curriculum import Sampler


class SimpleGFSampler(Sampler[str]):
    """Samples GF formulas."""

    def __init__(
        self,
        reach: int | tuple[int, int],
        avoid: int | tuple[int, int],
        propositions: list[str],
    ):
        if isinstance(reach, int):
            reach = (reach, reach)
        if isinstance(avoid, int):
            avoid = (avoid, avoid)
        self.reach = reach
        self.avoid = avoid
        self.propositions = propositions

    def sample(self, rng: random.Random) -> str:
        reach = rng.randint(self.reach[0], self.reach[1])
        avoid = rng.randint(self.avoid[0], self.avoid[1])
        reach_props = rng.sample(self.propositions, reach)
        available_props = [p for p in self.propositions if p not in reach_props]
        avoid_props = rng.sample(available_props, min(avoid, len(available_props)))
        formula = " & ".join(f"GF {p}" for p in reach_props)
        if avoid_props:
            formula += " & G(!(" + " | ".join(avoid_props) + "))"
        return formula

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, SimpleGFSampler):
            return False
        return (
            self.reach == value.reach
            and self.avoid == value.avoid
            and set(self.propositions) == set(value.propositions)
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.reach,
                self.avoid,
                tuple(sorted(self.propositions)),
            )
        )


class SimpleFGSampler(Sampler[str]):
    """Samples FG formulas."""

    def __init__(
        self,
        avoid: int | tuple[int, int],
        propositions: list[str],
    ):
        if isinstance(avoid, int):
            avoid = (avoid, avoid)
        self.avoid = avoid
        self.propositions = propositions

    def sample(self, rng: random.Random) -> str:
        reach_prop = rng.choice(self.propositions)
        avoid = rng.randint(self.avoid[0], self.avoid[1])
        available_props = [p for p in self.propositions if p != reach_prop]
        avoid_props = rng.sample(available_props, min(avoid, len(available_props)))
        formula = f"FG {reach_prop}"
        if avoid_props:
            formula += " & G(!(" + " | ".join(avoid_props) + "))"
        return formula

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, SimpleFGSampler):
            return False
        return self.avoid == value.avoid and set(self.propositions) == set(
            value.propositions
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.avoid,
                tuple(sorted(self.propositions)),
            )
        )
