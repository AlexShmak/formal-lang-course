"""Tests for FSM tools: regex_to_dfa"""

import pytest
from pyformlang.finite_automaton import DeterministicFiniteAutomaton

try:
    from project.task2 import regex_to_dfa
except ImportError:
    pytestmark = pytest.mark.skip("Task 2 is not ready to test!")


def test_regex_to_dfa():
    """Test regex to DFA conversion"""
    regex = "abc|d"
    dfa: DeterministicFiniteAutomaton = regex_to_dfa(regex=regex)
    assert not dfa.accepts(["a", "b", "c"])
    assert dfa.accepts(["d"])
    assert dfa.accepts(["abc"])
