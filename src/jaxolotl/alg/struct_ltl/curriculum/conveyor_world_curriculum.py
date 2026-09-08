import random

from jaxolotl.alg.curriculum import (
    CurriculumStage,
)
from jaxolotl.alg.curriculum.curriculum import RandomCurriculumStage, Sampler
from jaxolotl.alg.struct_ltl.reach_avoid.boolean_reach_avoid_sequence import (
    BooleanReachAvoidSequence,
)
from jaxolotl.environments.conveyor_world.conveyor_world import ConveyorWorld
from jaxolotl.ltl.logic.boolean_parser import VarNode


def make_stages(_: ConveyorWorld) -> list[CurriculumStage]:
    return [RandomCurriculumStage(sampler=ConveyorSequenceSampler(), threshold=None)]


class ConveyorSequenceSampler(Sampler[BooleanReachAvoidSequence]):
    """Samples formulas specific to the conveyor world."""

    def sample(self, rng: random.Random) -> BooleanReachAvoidSequence:
        parcel = VarNode("parcel")
        wrench = VarNode("wrench")
        hammer = VarNode("hammer")
        if rng.random() < 0.5:
            seq = [(parcel, None), (wrench, None)]
        else:
            seq = [(parcel, None), (hammer, None)]
        return BooleanReachAvoidSequence(seq, assignments=ConveyorWorld.assignments())
