from typing import Optional

from pyformlang.finite_automaton import EpsilonNFA, Symbol
from pyformlang.regular_expression import Regex
from pyformlang.rsa import RecursiveAutomaton, Box


def nfa_from_char(char: str) -> EpsilonNFA:
    if len(char) != 1:
        raise ValueError(f"Incorrect char: '{char}'.")
    return Regex(char).to_epsilon_nfa()


def nfa_from_var(var_name: str) -> EpsilonNFA:
    return Regex(var_name.upper()).to_epsilon_nfa()


def create_empty_nfa() -> EpsilonNFA:
    return Regex("$").to_epsilon_nfa()


def intersect(nfa1: EpsilonNFA, nfa2: EpsilonNFA) -> EpsilonNFA:
    """regex & regex"""
    return nfa1.get_intersection(nfa2).minimize()


def concatenate(nfa1: EpsilonNFA, nfa2: EpsilonNFA) -> EpsilonNFA:
    """regex . regex"""
    return nfa1.concatenate(nfa2).minimize()


def union(nfa1: EpsilonNFA, nfa2: EpsilonNFA) -> EpsilonNFA:
    """regex | regex"""
    return nfa1.union(nfa2).minimize()


def repeat(nfa: EpsilonNFA, times: int) -> EpsilonNFA:
    """regex ^ [range]"""
    if times == 0:
        return create_empty_nfa()

    result_nfa = nfa
    for _ in range(times - 1):
        result_nfa = concatenate(result_nfa, nfa)

    return result_nfa.minimize()


def kleene(nfa: EpsilonNFA) -> EpsilonNFA:
    """regex *"""
    return nfa.kleene_star().minimize()


def repeat_range(
    nfa: EpsilonNFA, left_border: int, right_border: Optional[int]
) -> EpsilonNFA:
    """regex ^ [left_border..right_border]"""
    if left_border == 0 and right_border is None:  # ^ [0..]
        return kleene(nfa)

    if right_border is None:  # ^ [left_border..]
        return concatenate(repeat(nfa, left_border), kleene(nfa))

    if left_border == right_border:  # ^ [x..x]
        return repeat(nfa, left_border)

    result_nfa = repeat(nfa, left_border)
    for times in range(left_border + 1, right_border + 1):
        result_nfa = union(result_nfa, repeat(nfa, times))

    return result_nfa.minimize()


def group(nfa: EpsilonNFA) -> EpsilonNFA:
    """( regex )"""
    minimized_regex = nfa.minimize().to_regex()
    return Regex(f"({minimized_regex})").to_epsilon_nfa()


def build_rsm(nfa: EpsilonNFA, subs_dict: dict[str, EpsilonNFA]) -> RecursiveAutomaton:
    start_terminal_name = "S"
    boxes = [
        Box(var_nfa, Symbol(var_name.upper()))
        for var_name, var_nfa in subs_dict.items()
    ]
    boxes.append(Box(nfa, Symbol(start_terminal_name)))

    return RecursiveAutomaton(initial_label=Symbol(start_terminal_name), boxes=boxes)
