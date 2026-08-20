"""Script to convert LTL formulas to LDBAs and plot them using Graphviz."""

import enum

from graphviz import Source

from jaxolotl.environments.warehouse_env.warehouse_env import WarehouseEnv
from jaxolotl.ltl.automata import LDBA, ltl2ldba
from jaxolotl.ltl.logic import Assignment
from jaxolotl.ltl.logic.utils import synthesize_formula
from jaxolotl.ltl.reach_avoid import EPSILON, path_search


class Color(enum.Enum):
    SINK = "tomato"
    ACCEPTING = "lightskyblue"
    ROOT = "#ffcc00"

    def __str__(self):
        return self.value


def draw_ldba(
    ldba: LDBA,
    filename="ldba",
    fmt="pdf",
    view=True,
    graph_label=False,
    assignments: list[Assignment] | None = None,
    self_loops=True,
) -> None:
    """Draw an LDBA as a graph using Graphviz."""

    dot = 'digraph "" {\n'
    dot += "rankdir=LR\n"
    dot += 'labelloc="t"\n'
    dot += 'node [shape="circle"]\n'
    dot += 'I [label="", style=invis, width=0]\n'
    dot += f"I -> {ldba.initial_state}\n"
    for state, transitions in ldba.state_to_transitions.items():
        dot += f'{state} [label="{state}" fontname="helvetica"'
        if state == ldba.sink_state:
            dot += f' color="{Color.SINK}" style="filled"'
        elif state == ldba.initial_state:
            dot += f' color="{Color.ROOT}" style="filled"'
        dot += "]\n"
        for transition in transitions:
            if not self_loops and transition.target == state:
                continue
            label = transition.positive_label
            if graph_label and not transition.is_epsilon():
                assert assignments is not None, (
                    "Assignments must be provided for graph labeling."
                )
                label = synthesize_formula(
                    frozenset(transition.valid_assignments),
                    tuple(assignments),
                    tuple(ldba.propositions),
                )
            dot += f'{state} -> {transition.target} [label="{label}"'
            if transition.accepting:
                dot += f' color="{Color.ACCEPTING}"'
            dot += ' fontname="helvetica"'
            dot += "]\n"
    dot += "}"
    s = Source(dot, filename=filename, format=fmt)
    s.render(view=view, cleanup=True)


def construct_ldba(
    formula: str, prune: bool = True, assignments: list[Assignment] | None = None
) -> LDBA:
    ldba = ltl2ldba(formula)
    print("Constructed LDBA.")
    assert ldba.check_valid()
    print("Checked valid.")
    if prune:
        assignments = assignments or Assignment.zero_or_one_propositions(
            set(ldba.propositions)
        )
        ldba.prune(assignments)
        print("Pruned impossible transitions.")
    ldba.complete_sink_state()
    print("Added sink state.")
    ldba.compute_sccs()
    return ldba


if __name__ == "__main__":
    f = "FG region_a"
    assignments = WarehouseEnv.assignments()
    # for i, a in enumerate(assignments):
    #     print(f"Assignment {i}: {a}")
    props = WarehouseEnv.propositions
    print_paths = False

    ldba = construct_ldba(f, prune=True, assignments=assignments)

    for transitions in ldba.state_to_transitions.values():
        num_eps = sum(t.is_epsilon() for t in transitions)
        if num_eps > 1:
            print(f"State has {num_eps} epsilon transitions.")

    print(f"Finite: {ldba.is_finite_specification()}")
    print(f"Num states: {ldba.num_states}")
    draw_ldba(
        ldba, fmt="png", graph_label=True, assignments=assignments, self_loops=True
    )

    if print_paths:
        paths = path_search.compute_sequences(ldba)
        for state in range(ldba.num_states):
            print(f"State {state}:")
            for path in paths[state]:  # type: ignore
                path_str = []
                for reach, avoid in path:
                    if reach == EPSILON:
                        reach_graph = "ε"
                    else:
                        reach_graph = synthesize_formula(
                            reach, tuple(assignments), props
                        )
                    avoid_graph = synthesize_formula(avoid, tuple(assignments), props)
                    path_str.append(f"[{reach_graph}, {avoid_graph}]")
                print(" -> ".join(path_str))
                for reach, avoid in path:
                    print((reach, avoid))
        print()
