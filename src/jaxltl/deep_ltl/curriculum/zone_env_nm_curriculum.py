from pathlib import Path

from jaxltl.deep_ltl.curriculum.simple_samplers import (
    SimpleReachAvoidSampler,
    SimpleReachStaySampler,
)
from jaxltl.deep_ltl.utils.batching import ReachAvoidSequenceBatcher
from jaxltl.environments.environment import Environment
from jaxltl.environments.wrappers.wrapper import EnvWrapper
from jaxltl.ltl2action.curriculum.curriculum import (
    Curriculum,
    MultiRandomStage,
    RandomCurriculumStage,
)


def make(env: Environment | EnvWrapper, load_path: Path | None = None) -> Curriculum:
    return Curriculum(
        [
            # 1. Simple reach tasks
            RandomCurriculumStage(
                sampler=SimpleReachAvoidSampler(
                    depth=1,
                    reach=1,
                    avoid=0,
                    assignments=env.assignments(),
                ),
                threshold=0.9,
            ),
            # 2. Reach tasks of depth 2
            RandomCurriculumStage(
                sampler=SimpleReachAvoidSampler(
                    depth=2, reach=1, avoid=0, assignments=env.assignments()
                ),
                threshold=0.8,
            ),
            # 3. Simple reach-avoid tasks
            RandomCurriculumStage(
                sampler=SimpleReachAvoidSampler(
                    depth=1, reach=1, avoid=1, assignments=env.assignments()
                ),
                threshold=0.95,
            ),
            # 4. Reach-avoid tasks of depth 2
            RandomCurriculumStage(
                sampler=SimpleReachAvoidSampler(
                    depth=2, reach=1, avoid=1, assignments=env.assignments()
                ),
                threshold=0.8,
            ),
            # 5. Reach-avoid / reach-stay tasks
            MultiRandomStage(
                stages=[
                    RandomCurriculumStage(
                        sampler=SimpleReachAvoidSampler(
                            depth=(1, 2),
                            reach=(1, 2),
                            avoid=(0, 2),
                            assignments=env.assignments(),
                        ),
                        threshold=None,
                    ),
                    RandomCurriculumStage(
                        sampler=SimpleReachStaySampler(
                            num_stay=30, avoid=(0, 1), assignments=env.assignments()
                        ),
                        threshold=None,
                    ),
                ],
                probs=[0.4, 0.6],
                threshold=0.85,
            ),
            # 6. More complex reach-avoid / reach-stay tasks
            MultiRandomStage(
                stages=[
                    RandomCurriculumStage(
                        sampler=SimpleReachAvoidSampler(
                            depth=(1, 2),
                            reach=(1, 2),
                            avoid=(0, 2),
                            assignments=env.assignments(),
                        ),
                        threshold=None,
                    ),
                    RandomCurriculumStage(
                        sampler=SimpleReachStaySampler(
                            num_stay=60, avoid=(0, 1), assignments=env.assignments()
                        ),
                        threshold=None,
                    ),
                ],
                probs=[0.8, 0.2],
                threshold=0.85,
            ),
            # 7. Final mixture of complex tasks
            MultiRandomStage(
                stages=[
                    RandomCurriculumStage(
                        sampler=SimpleReachAvoidSampler(
                            depth=(1, 2),
                            reach=(1, 2),
                            avoid=(0, 2),
                            assignments=env.assignments(),
                        ),
                        threshold=None,
                    ),
                    RandomCurriculumStage(
                        sampler=SimpleReachStaySampler(
                            num_stay=60, avoid=(0, 2), assignments=env.assignments()
                        ),
                        threshold=None,
                    ),
                ],
                probs=[0.8, 0.2],
                threshold=None,
            ),
        ],
        num_samples=int(1e5),
        batcher=ReachAvoidSequenceBatcher(),
        env=env,
        load_path=load_path,
    )
