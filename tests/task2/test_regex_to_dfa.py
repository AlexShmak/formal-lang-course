"""Tests for FSM tools: regex_to_dfa"""

from pyformlang.finite_automaton import DeterministicFiniteAutomaton

from project.task2 import regex_to_dfa


def test_regex_to_dfa():
    """Test regex to DFA conversion"""
    regex = "abc|d"
    dfa: DeterministicFiniteAutomaton = regex_to_dfa(regex=regex)
    assert not dfa.accepts(["a", "b", "c"])
    assert dfa.accepts(["d"])
    assert dfa.accepts(["abc"])
