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
)
from scipy.sparse import csr_array, kron

from project.task2 import graph_to_nfa, regex_to_dfa


class AdjacencyMatrixFA:
    """Adjacency Matrix Finite Automata class"""

    def __init__(
        self, fa: NondeterministicFiniteAutomaton | DeterministicFiniteAutomaton | None
    ):
        # get start and final states

        if not fa:
            self.states_count = 0
            self.states = {}
            self.start_states = set()
            self.final_states = set()
            self.adjacency_matrices = {}
            return

        self.states_count = len(fa.states)
        self.states = {st: ind for (ind, st) in enumerate(fa.states)}
        self.start_states = set(self.states[int(x.value)] for x in fa.start_states)
        self.final_states = set(self.states[int(x.value)] for x in fa.final_states)

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
                    self.states[source_state], self.states[dest_state]
                ] = True

        self.adjacency_matrices: dict[Symbol, csr_array] = {
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
            _word = configuration[0]
            _state = configuration[1]
            if not _word and _state in self.final_states:
                print("Accepts!")
                return True
            elif not _word:
                print("Don't accept!")
                return False
            else:
                adjacency_matrices = self.adjacency_matrices
                sym = _word[0]
                if sym not in adjacency_matrices.keys():
                    print("Don't accept!")
                    return False
                matrix_row = adjacency_matrices[sym].toarray()[_state]

                for state, state_bool_value in enumerate(matrix_row):
                    if state_bool_value:
                        configurations.append((_word[1:], state))
                    else:
                        continue

        print("Don't accept!")
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
        st1_ind, st2_ind = automaton1.states[st1], automaton2.states[st2]
        intersection_ind = automaton2.states_count * st1_ind + st2_ind

        # add the new state
        intersection.states[(st1, st2)] = intersection_ind

        # add the new state to start or final states
        if st1_ind in automaton1.start_states and st2_ind in automaton2.start_states:
            intersection.start_states.add(intersection_ind)
        elif st1_ind in automaton1.final_states and st2_ind in automaton2.final_states:
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
