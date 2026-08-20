# Preprocessing refactor log — 2026-08-20 08:56 BST

## Scope

Implement refactoring step 5: consolidate formula/LDBA preprocessing and PyTree
padding without changing the public Hydra preprocessing entry points.

## Starting state

- The worktree is clean at `65b0a25`.
- User changes renamed the shared reach-avoid encoder to `batching.py` and its
  functions to `batch_assignments` / `batch_state_sequences`.
- DeepLTL and GenZ-LTL duplicate LDBA construction, JAX conversion, batching,
  and reach-avoid path search.
- StructLTL imports private DeepLTL helpers and contains three separate sequence
  batchers with field-specific padding behavior.

## Design decisions

- Put LDBA construction and JAX LDBA batching beside the automata types under
  `jaxolotl.ltl.automata`.
- Put the shared formula→LDBA→reach-avoid pipeline under
  `jaxolotl.alg.reach_avoid`; algorithm modules remain thin public adapters.
- Introduce one generic PyTree `pad_and_stack` helper with an explicit parallel
  PyTree of padding values. Do not infer semantic padding from dtype.
- Preserve the configured preprocessing entry points and formula ordering.
- Add unit tests for LDBA batching, generic pipeline delegation, heterogeneous
  PyTree shapes, and semantic sequence/graph padding.

## Progress

- Inspected the rebased user changes and current duplicated implementations.
- Added shared LDBA construction/batching and a generic reach-avoid formula
  preprocessing pipeline.
- Converted DeepLTL, GenZ-LTL, and StructLTL preprocessing modules into thin
  adapters while preserving their public entry points.
- Added focused tests for LDBA padding, construction ordering, pipeline
  delegation, transforms, and empty input validation.
