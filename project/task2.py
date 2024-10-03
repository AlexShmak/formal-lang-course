"""
FSM tools
"""

from typing import Set

from networkx import MultiDiGraph
from pyformlang.finite_automaton import (
    DeterministicFiniteAutomaton,
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
    pf_regex = Regex(regex)
    nfa: NondeterministicFiniteAutomaton = (
        pf_regex.to_epsilon_nfa().remove_epsilon_transitions()
    )
    dfa = nfa.to_deterministic()
    return dfa


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
            graph=graph
        ).remove_epsilon_transitions()
    )
    fa_states = set(int(i) for i in graph.nodes)
    fa_start_states = fa_states if not start_states else start_states
    fa_final_states = fa_states if not final_states else final_states

    for state in fa_start_states:
        nfa.add_start_state(State(state))

    for state in fa_final_states:
        nfa.add_final_state(State(state))

    return nfa
