"""
FSM tools
"""

from typing import Set

from networkx.classes import MultiDiGraph
from pyformlang.finite_automaton import (
    DeterministicFiniteAutomaton,
    EpsilonNFA,
    NondeterministicFiniteAutomaton,
    State,
)
from pyformlang.regular_expression import Regex


def regex_to_dfa(regex: str) -> DeterministicFiniteAutomaton:
    """Convert regex to DFA

    Args:
        regex (str): regular expression

    Returns:
        DeterministicFiniteAutomaton: DFA
    """
    reg_expr = Regex(regex)
    nfa: EpsilonNFA = reg_expr.to_epsilon_nfa()
    dfa: DeterministicFiniteAutomaton = nfa.to_deterministic()
    return dfa.minimize()


def graph_to_nfa(
    graph: MultiDiGraph, start_states: Set[int], final_states: Set[int]
) -> NondeterministicFiniteAutomaton:
    """Convert graph to NFA

    Args:
        graph (MultiDiGraph)
        start_states (Set[int])
        final_states (Set[int])

    Returns:
        NondeterministicFiniteAutomaton
    """
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
