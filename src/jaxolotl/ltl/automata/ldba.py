from collections import Counter
from dataclasses import dataclass, field

from jaxolotl.ltl.logic import Assignment


class LDBA:
    """Represents a limit-deterministic Büchi automaton (LDBA)."""

    def __init__(self, propositions: set[str], formula: str | None = None):
        self.formula = formula
        self.propositions: tuple[str, ...] = tuple(sorted(propositions))
        self.num_states = 0
        self.num_transitions = 0
        self.initial_state = None
        self.state_to_transitions: dict[int, list[LDBATransition]] = {}
        self.state_to_incoming_transitions: dict[int, list[LDBATransition]] = {}
        self.sink_state: int | None = None
        self.complete = False
        self.possible_assignments: list[Assignment] | None = None
        self.state_to_scc = {}
        self.state_to_info = {}

    @property
    def states(self) -> list[int]:
        return list(range(self.num_states))

    def add_state(self, state: int, initial=False, info: dict | None = None) -> None:
        if state < 0:
            raise ValueError("State must be a positive integer.")
        if initial:
            if self.initial_state is not None:
                raise ValueError("Initial state already set.")
            self.initial_state = state
        self.num_states = max(self.num_states, state + 1)
        if state not in self.state_to_transitions:
            self.state_to_transitions[state] = []
        if state not in self.state_to_incoming_transitions:
            self.state_to_incoming_transitions[state] = []
        if info is not None:
            self.state_to_info[state] = info

    def get_next_state(
        self, state: int, propositions: set[str], take_epsilon=False
    ) -> tuple[int, bool]:
        """Returns the next state and whether the taken transition is accepting,
        given the current state and the propositions that are true."""
        if take_epsilon:
            eps_transitions = [
                t for t in self.state_to_transitions[state] if t.is_epsilon()
            ]
            if len(eps_transitions) > 1:
                raise ValueError(
                    "More than one epsilon transition from a state is currently not supported."
                )
            assert eps_transitions
            t = eps_transitions[0]
            return t.target, t.accepting
        assignment = Assignment(frozenset(propositions))
        for transition in self.state_to_transitions[state]:
            if assignment in transition.valid_assignments:
                return transition.target, transition.accepting
        raise ValueError("Invalid transition.")

    def get_ordered_epsilon_transitions(self, state: int) -> list[int]:
        """Returns the list of target states for epsilon transitions from the given state.
        Ordered by target state index."""
        eps_transitions = [
            t for t in self.state_to_transitions[state] if t.is_epsilon()
        ]
        eps_transitions.sort(key=lambda t: t.target)
        return [t.target for t in eps_transitions]

    def contains_state(self, state: int) -> bool:
        return state <= self.num_states

    def is_finite_specification(self) -> bool:
        """
        Checks if the LDBA represents a finite specification.
        Note: this is not an exhaustive check, but a sufficient heuristic for LDBAs constructed by rabinizer.
        """
        if not self.state_to_scc:
            self.compute_sccs()
        accepting_sccs = [scc for scc in self.state_to_scc.values() if scc.accepting]
        if len(accepting_sccs) > 1:
            return False
        if len(accepting_sccs) == 0:
            raise ValueError(f"LDBA for formula {self.formula} has no accepting SCCs.")
        scc = accepting_sccs[0]
        return scc.bottom and len(scc.states) == 1

    def add_transition(
        self, source: int, target: int, label: str | None, accepting: bool
    ) -> "LDBATransition":
        if source < 0 or source >= self.num_states:
            raise ValueError("Source state must be a valid state index.")
        if target < 0 or target >= self.num_states:
            raise ValueError("Target state must be a valid state index.")
        transition = LDBATransition(source, target, label, accepting, self.propositions)
        self.num_transitions += 1
        for t in self.state_to_transitions[source]:
            if t == transition:
                raise ValueError(
                    "There can only be a single transition between two states. Consider merging labels."
                )
        self.state_to_transitions[source].append(transition)
        self.state_to_incoming_transitions[target].append(transition)
        return transition

    def check_deterministic_transitions(self, state: int) -> bool:
        """Checks that the transitions from a state are deterministic."""
        num_assignment_transitions = Counter(
            [
                a
                for transition in self.state_to_transitions[state]
                for a in transition.valid_assignments
            ]
        )
        return all(c <= 1 for c in num_assignment_transitions.values())

    def complete_sink_state(self):
        """Adds a sink state and transitions to the LDBA if any transitions are missing."""
        if self.complete:
            return
        sink_state = self.num_states
        if self.possible_assignments:
            all_assignments = set(self.possible_assignments)
        else:
            all_assignments = set(
                Assignment.all_possible_assignments(tuple(self.propositions))
            )
        for state in range(self.num_states):
            covered_assignments = (
                set()
                if not self.state_to_transitions[state]
                else set.union(
                    *[t.valid_assignments for t in self.state_to_transitions[state]]
                )
            )
            if len(covered_assignments) != len(all_assignments):
                # missing transitions - need to add sink state
                if not self.has_sink_state():
                    self.sink_state = sink_state
                    self.add_state(sink_state)
                    t = self.add_transition(sink_state, sink_state, "t", False)
                    t._valid_assignments = all_assignments
                    assert self.has_sink_state()
                sink_assignments = all_assignments - covered_assignments
                sink_label = self.valid_assignments_to_label(sink_assignments)
                t = self.add_transition(state, sink_state, sink_label, False)
                t._valid_assignments = sink_assignments
        self.complete = True

    def has_sink_state(self) -> bool:
        return self.sink_state is not None

    def valid_assignments_to_label(self, valid_assignments: set[Assignment]) -> str:
        assert len(valid_assignments) > 0
        formula = " | ".join(str(a) for a in valid_assignments)
        return formula

    def prune(self, possible_assignments: list[Assignment]):
        """Prunes transitions that involve impossible assignments. Impossible assignments may be derived from knowledge
        of the underlying MDP."""
        self.possible_assignments = possible_assignments
        to_remove = set()
        for transitions in self.state_to_transitions.values():
            for t in transitions:
                if t.is_epsilon():
                    continue
                t._valid_assignments = {
                    a for a in possible_assignments if a.satisfies(t.label)
                }
                if t.valid_assignments:
                    t.label = self.valid_assignments_to_label(t.valid_assignments)
                else:
                    to_remove.add(t)
        self.num_transitions -= len(to_remove)
        for state in range(self.num_states):
            self.state_to_transitions[state] = [
                t for t in self.state_to_transitions[state] if t not in to_remove
            ]
            self.state_to_incoming_transitions[state] = [
                t
                for t in self.state_to_incoming_transitions[state]
                if t not in to_remove
            ]

    def eliminate_forced_epsilons(self):
        """Removes states whose only outgoing edge is an epsilon transition by retargeting
        their incoming transitions to the epsilon's target.
        """
        if self.complete:
            raise ValueError(
                "eliminate_forced_epsilons must be called before complete_sink_state."
            )
        self.state_to_scc = {}
        removed: set[int] = set()
        while True:
            forced = next(
                (
                    s
                    for s in range(self.num_states)
                    if s not in removed
                    and len(self.state_to_transitions[s]) == 1
                    and self.state_to_transitions[s][0].is_epsilon()
                ),
                None,
            )
            if forced is None:
                break
            eps = self.state_to_transitions[forced][0]
            if eps.target == forced:
                raise ValueError(
                    "Cannot eliminate a state whose only edge is an epsilon self-loop."
                )
            self.state_to_transitions[forced] = []
            self.state_to_incoming_transitions[eps.target].remove(eps)
            self.num_transitions -= 1
            for t in self.state_to_incoming_transitions[forced]:
                accepting = t.accepting or eps.accepting
                merge_into = next(
                    (
                        e
                        for e in self.state_to_transitions[t.source]
                        if e is not t
                        and e.target == eps.target
                        and e.is_epsilon() == t.is_epsilon()
                        and e.accepting == accepting
                    ),
                    None,
                )
                if merge_into is None:
                    t.target = eps.target
                    t.accepting = accepting
                    self.state_to_incoming_transitions[eps.target].append(t)
                else:
                    if not t.is_epsilon():
                        merge_into.label = f"({merge_into.label}) | ({t.label})"
                        merge_into._valid_assignments = (
                            merge_into.valid_assignments | t.valid_assignments
                        )
                    self.state_to_transitions[t.source].remove(t)
                    self.num_transitions -= 1
            self.state_to_incoming_transitions[forced] = []
            if self.initial_state == forced:
                self.initial_state = eps.target
            removed.add(forced)
        if removed:
            self._compact_states(removed)

    def _compact_states(self, removed: set[int]):
        """Renumbers the remaining states densely, dropping `removed`. The removed states
        must have no transitions left."""
        remap: dict[int, int] = {}
        for s in range(self.num_states):
            if s not in removed:
                remap[s] = len(remap)
        transitions = {
            id(t): t for s in remap for t in self.state_to_transitions[s]
        }.values()
        for t in transitions:
            t.source = remap[t.source]
            t.target = remap[t.target]
        self.state_to_transitions = {
            remap[s]: ts for s, ts in self.state_to_transitions.items() if s in remap
        }
        self.state_to_incoming_transitions = {
            remap[s]: ts
            for s, ts in self.state_to_incoming_transitions.items()
            if s in remap
        }
        self.state_to_info = {
            remap[s]: info for s, info in self.state_to_info.items() if s in remap
        }
        self.num_states = len(remap)
        assert self.initial_state is not None
        self.initial_state = remap[self.initial_state]
        if self.sink_state is not None:
            self.sink_state = remap[self.sink_state]

    def compute_sccs(self) -> None:
        """Computes the strongly connected components of the LDBA using Tarjan's algorithm."""
        if self.initial_state is None:
            raise ValueError("Initial state is not set.")
        if self.state_to_scc:
            return
        num = 0
        nums: dict[int, int] = {}
        visited: set[int] = set()
        stack: list[tuple[int, set[int]]] = []
        active: set[int] = set()

        def tarjan(s: int):
            nonlocal num
            nonlocal nums
            nonlocal visited
            nonlocal stack
            nonlocal active
            visited.add(s)
            active.add(s)
            num += 1
            nums[s] = num
            stack.append((s, {s}))
            for t in self.state_to_transitions[s]:
                if t.target not in visited:
                    tarjan(t.target)
                elif t.target in active:
                    scc = set()
                    while True:
                        u, current = stack.pop()
                        scc |= current
                        if nums[u] <= nums[t.target]:
                            break
                    stack.append((u, scc))
            if stack[-1][0] == s:
                _, states = stack.pop()
                active -= states
                transitions = [
                    t for state in states for t in self.state_to_transitions[state]
                ]
                accepting = any(t.accepting and t.target in states for t in transitions)
                bottom = all(t.target in states for t in transitions)
                scc = SCC(frozenset(states), accepting, bottom)
                for state in states:
                    assert state not in self.state_to_scc
                    self.state_to_scc[state] = scc

        tarjan(self.initial_state)


@dataclass
class LDBATransition:
    source: int
    target: int
    label: str | None  # None for epsilon transitions
    accepting: bool
    propositions: tuple[str, ...]
    _valid_assignments: set[Assignment] = field(default_factory=set)

    def is_epsilon(self) -> bool:
        return self.label is None

    @property
    def valid_assignments(self) -> set[Assignment]:
        if self.is_epsilon():
            return set()
        if self._valid_assignments is None:
            self._valid_assignments = {
                a
                for a in Assignment.all_possible_assignments(self.propositions)
                if a.satisfies(self.label)
            }
        return self._valid_assignments

    @property
    def positive_label(self) -> str:
        if self.is_epsilon():
            return "ε"
        return " | ".join(str(a) for a in self.valid_assignments)

    def __eq__(self, other):
        if not isinstance(other, LDBATransition):
            return False
        return (
            self.source == other.source
            and self.target == other.target
            and self.is_epsilon() == other.is_epsilon()
            and self.accepting == other.accepting
        )

    def __hash__(self):
        return hash((self.source, self.target, self.is_epsilon(), self.accepting))


@dataclass(eq=True, frozen=True)
class SCC:
    states: frozenset[int]
    accepting: bool
    bottom: bool
