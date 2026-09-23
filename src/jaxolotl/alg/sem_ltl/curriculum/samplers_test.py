import random

import pytest

from jaxolotl.alg.sem_ltl.curriculum.samplers import (
    BooleanReachStayFormulaSampler,
)
from jaxolotl.environments.warehouse_env.warehouse_env import WarehouseEnv
from jaxolotl.ltl.logic.boolean_parser import VarNode


def test_boolean_reach_stay_without_avoid_samples_fg_formula():
    sampler = BooleanReachStayFormulaSampler(
        reach_formulas=[VarNode("region_a")],
        avoid_formulas=[],
        avoid_prob=1.0,
        assignments=WarehouseEnv.assignments(),
    )

    assert sampler.sample(random.Random(0)) == "FG (region_a)"


def test_boolean_reach_stay_samples_feasible_avoid_formula():
    sampler = BooleanReachStayFormulaSampler(
        reach_formulas=[VarNode("region_a")],
        avoid_formulas=[VarNode("region_a"), VarNode("door")],
        avoid_prob=1.0,
        assignments=WarehouseEnv.assignments(),
    )

    assert sampler.sample(random.Random(0)) == "(!(door) U G(region_a))"


def test_boolean_reach_stay_requires_reach_formula():
    with pytest.raises(ValueError, match="At least one reach formula"):
        BooleanReachStayFormulaSampler(
            reach_formulas=[],
            avoid_formulas=[],
            assignments=WarehouseEnv.assignments(),
        )
