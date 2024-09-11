from pyformlang.regular_expression import Regex
from pyformlang.finite_automaton import (
    DeterministicFiniteAutomaton,
    NondeterministicFiniteAutomaton,
    EpsilonNFA,
    State,
)
from networkx.classes import MultiDiGraph
from typing import Set


def regex_to_dfa(regex: str) -> DeterministicFiniteAutomaton:
    reg_expr = Regex(regex)
    nfa: EpsilonNFA = reg_expr.to_epsilon_nfa()
    dfa: DeterministicFiniteAutomaton = nfa.to_deterministic()
    return dfa.minimize()


def graph_to_nfa(
    graph: MultiDiGraph, start_states: Set[int], final_states: Set[int]
) -> NondeterministicFiniteAutomaton:
    nfa: NondeterministicFiniteAutomaton = (
        NondeterministicFiniteAutomaton.from_networkx(
            graph
        ).remove_epsilon_transitions()
    )

    nodes = set(int(i) for i in graph.nodes)
    start_nodes = nodes if not start_states else start_states
    final_nodes = nodes if not final_states else final_states

    for state in start_nodes:
        nfa.add_start_state(State(state))

    for state in final_nodes:
        nfa.add_final_state(State(state))

    return nfa
