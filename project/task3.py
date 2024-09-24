"""FSM tools: Matrices"""

from collections import defaultdict
from typing import Iterable

import numpy as np
from numpy import bool_
from numpy.typing import NDArray
from pyformlang.finite_automaton import (
    NondeterministicFiniteAutomaton,
    Symbol,
)
from scipy.sparse import csr_array


class AdjacencyMatrixFA:
    """Adjacency Matrix Finite Automata class"""

    def __init__(self, fa: NondeterministicFiniteAutomaton):
        # get start and final states
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
                    print(
                        state,
                        state_bool_value,
                    )
                    if state_bool_value:
                        configurations.append((_word[1:], state))
                    else:
                        continue

        print("Don't accept!")
        return False

    def is_empty(self) -> bool:
        """Check whether the automata-generated language is empty or not

        Returns:
            bool
        """
        # TODO
        pass
