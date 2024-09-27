"""FSM tools: Matrices"""

from collections import defaultdict
from itertools import product
from typing import Iterable

import numpy as np
from networkx import MultiDiGraph
from numpy import bool_
from numpy.typing import NDArray
from pyformlang.finite_automaton import (
    DeterministicFiniteAutomaton,
    NondeterministicFiniteAutomaton,
    Symbol,
    State,
)
from scipy.sparse import csr_array, kron

from project.task2 import graph_to_nfa, regex_to_dfa


class AdjacencyMatrixFA:
    """Adjacency Matrix Finite Automata class"""

    def __init__(
        self, fa: NondeterministicFiniteAutomaton | DeterministicFiniteAutomaton | None
    ):
        self.start_states: set[int] = set()
        self.final_states: set[int] = set()
        self.states_count = 0
        self.states: list[State] = []
        self.adjacency_matrices: dict[Symbol, csr_array] = {}

        if not fa:
            return

        self.states_count = len(fa.states)
        self.states = list(fa.states)
        for i in range(self.states_count):
            state: State = self.states[i]

            if state in fa.start_states:
                self.start_states.add(i)
            if state in fa.final_states:
                self.final_states.add(i)

        # get transitions
        self.number_transitions = fa.get_number_transitions()
        self.transitions = list(fa.to_dict().items())

        # get adjacency sparce matrices
        transition_matrices: dict[Symbol, NDArray[bool_]] = defaultdict(
            lambda: np.zeros(shape=(self.states_count, self.states_count), dtype=bool_)
        )

        for source_state, dest_state, label in fa.to_networkx().edges(data="label"):
            if label:
                transition_matrices[Symbol(label)][
                    self.states.index(source_state), self.states.index(dest_state)
                ] = True

        self.adjacency_matrices = {
            label: csr_array(transition_matrix)
            for (label, transition_matrix) in transition_matrices.items()
        }

    def accepts(self, word: Iterable[Symbol]) -> bool:
        """Interpreter function

        Args:
            word (Iterable[Symbol])

        Returns:
            bool
        """
        # current word and all the possible states to start from
        # [(`word`, 0), ('word`, 1) ... ] -> [('ord', 2), ('ord`, 3) ... ] -> ...
        configurations: list[(list[Symbol], int)] = []
        for state in self.start_states:
            configurations.append((list(word), state))

        for configuration in configurations:
            if not configurations:
                return False

            word = configuration[0]
            state = configuration[1]
            if (not word) and (state in self.final_states):
                return True
            elif not word:
                continue
            else:
                adjacency_matrices = self.adjacency_matrices
                sym = word[0]
                if sym not in adjacency_matrices.keys():
                    continue
                matrix_row = adjacency_matrices[sym].toarray()[state]

                for state, state_bool_value in enumerate(matrix_row):
                    if state_bool_value:
                        configurations.append((word[1:], state))
                    else:
                        continue
        return False

    def transitive_closure(self) -> NDArray[bool_]:
        """Transitive closure for the states of the automata

        Returns:
            NDArray[bool_]
        """
        transitive_closure: NDArray[bool_] = np.diag(
            np.ones(self.states_count, dtype=bool_)
        )
        for adjacency_matrix in self.adjacency_matrices.values():
            transitive_closure |= adjacency_matrix.toarray()

        for m in range(self.states_count):
            for x in range(self.states_count):
                for y in range(self.states_count):
                    transitive_closure[x, y] = transitive_closure[x, y] or (
                        transitive_closure[x, m] and transitive_closure[m, y]
                    )
        return transitive_closure

    def is_empty(self) -> bool:
        """Check whether the automata-generated language is empty or not

        Returns:
            bool
        """
        for i in self.start_states:
            for j in self.final_states:
                if self.transitive_closure()[i, j]:
                    return False
        return True


def intersect_automata(
    automaton1: AdjacencyMatrixFA, automaton2: AdjacencyMatrixFA
) -> AdjacencyMatrixFA:
    """Automaton intersection function

    Args:
        automaton1 (AdjacencyMatrixFA)
        automaton2 (AdjacencyMatrixFA)

    Returns:
        AdjacencyMatrixFA
    """
    intersection = AdjacencyMatrixFA(None)
    intersection.states_count = automaton1.states_count * automaton2.states_count
    for st1, st2 in product(automaton1.states, automaton2.states):
        st1_ind, st2_ind = automaton1.states.index(st1), automaton2.states.index(st2)
        intersection_ind = automaton2.states_count * st1_ind + st2_ind

        # add the new state
        intersection.states.append(State((st1, st2)))

        # add the new state to start or final states
        if st1_ind in automaton1.start_states and st2_ind in automaton2.start_states:
            intersection.start_states.add(intersection_ind)
        if st1_ind in automaton1.final_states and st2_ind in automaton2.final_states:
            intersection.final_states.add(intersection_ind)

    for sym, adj_matrix1 in automaton1.adjacency_matrices.items():
        if sym in automaton2.adjacency_matrices.keys():
            adj_matrix2 = automaton2.adjacency_matrices[sym]

            intersection.adjacency_matrices[sym] = kron(
                adj_matrix1, adj_matrix2, format="csr"
            )
        else:
            continue
    return intersection


def tensor_based_rpq(
    graph: MultiDiGraph, start_nodes: set[int], final_nodes: set[int], regex: str
) -> list[tuple[int, int]]:
    """Regular Path Queries

    Args:
        graph (MultiDiGraph)
        start_nodes (set[int])
        final_nodes (set[int])
        regex (str)

    Returns:
        list[tuple[int, int]]
    """
    regex_amfa = AdjacencyMatrixFA(regex_to_dfa(regex))
    graph_amfa = AdjacencyMatrixFA(
        graph_to_nfa(graph=graph, start_states=start_nodes, final_states=final_nodes)
    )
    intersection = intersect_automata(regex_amfa, graph_amfa)
    transitive_closure = intersection.transitive_closure()

    return_list: list[tuple[int, int]] = []

    for graph_start, graph_final in product(start_nodes, final_nodes):
        for regex_start_index, regex_final_index in product(
            regex_amfa.start_states, regex_amfa.final_states
        ):
            graph_start_i = graph_amfa.states.index(graph_start)
            graph_final_i = graph_amfa.states.index(graph_final)

            start_index = regex_amfa.states_count * graph_start_i + regex_start_index
            final_index = regex_amfa.states_count * graph_final_i + regex_final_index

            if transitive_closure[start_index, final_index]:
                return_list.append((graph_start, graph_final))

    return return_list
