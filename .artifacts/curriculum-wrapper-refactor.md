# Curriculum and wrapper refactor log — 2026-08-20 11:37 BST

## Scope

Finish refactoring points 6 and 7:

1. Move shared curriculum abstractions and the curriculum environment wrapper
   out of the LTL2Action algorithm package.
2. Consolidate duplicated StructLTL curricula and algorithm wrapper factories.

## Starting state

- Worktree clean at `226962e`; the user's cleanup commits are preserved.
- `Sampler`, `SampleBatcher`, stages, and `Curriculum` are used by every
  algorithm but owned by `alg.ltl2action`.
- `CurriculumWrapper` and `CurriculumResetOptions` are likewise shared.
- Warehouse clause, graph, and token curricula contain the same seven stages
  and differ only in their batcher.
- DeepLTL, GenZ-LTL, and LTL2Action repeat curriculum loading/wrapping logic.

## Design decisions

- Create `jaxolotl.alg.curriculum` as the neutral owner of curriculum types and
  wrapper state.
- Keep algorithm-specific samplers and task wrappers in their algorithm
  packages.
- Make the warehouse and zone curriculum builders accept a batcher and an
  `ablation` flag; retain thin compatibility modules for old Python imports.
- Point Hydra graph/token configurations directly at the central warehouse
  factory with an instantiated batcher argument.
- Centralize curriculum loading and training wrapping in an algorithm-neutral
  wrapper factory while keeping algorithm `wrap_env` entry points stable.

## Progress

- Inspected current imports, Hydra targets, curricula, and wrapper flows.
- Moved the shared curriculum model and environment wrapper into
  `jaxolotl.alg.curriculum`; updated all consumers.
- Verified the extraction with Ruff and the complete test suite (95 passed).
