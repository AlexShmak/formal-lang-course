"""
Altered Functions for Experimental Purposes (Generics added)
"""

from collections import defaultdict
from functools import reduce
from itertools import product
from typing import Iterable, List, Set, Tuple, TypeVar

import numpy as np
from networkx import MultiDiGraph
from numpy import bool_
from numpy.typing import NDArray
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton, Symbol
from scipy.sparse import dok_matrix, kron, lil_matrix

from project.task2 import graph_to_nfa, regex_to_dfa

T = TypeVar("T")
F = TypeVar("F")


class ExperimentalAdjacencyMatrixFA:
    """Adjacency Matrix Finite Automata class"""

    def __init__(self, fa: NondeterministicFiniteAutomaton, matrix_type: T):
        self.start_states: Set[int] = set()
        self.final_states: Set[int] = set()

        if not fa:
            self.states_count = 0
            self.states: dict[any, int] = {}
            self.adj_matrices = {}
            return

        graph = fa.to_networkx()
        self.states_count = graph.number_of_nodes()
        self.states: dict[any, int] = {st: i for (i, st) in enumerate(graph.nodes)}

        for state in fa.states:
            if state in fa.start_states:
                self.start_states.add(self.states[state])
            if state in fa.final_states:
                self.final_states.add(self.states[state])

        matrices: dict[Symbol, NDArray[bool_]] = defaultdict(
            lambda: np.zeros((self.states_count, self.states_count), dtype=bool_)
        )

        for source_st, dest_st, label in graph.edges(data="label"):
            if label:
                matrices[Symbol(label)][
                    self.states[source_st], self.states[dest_st]
                ] = True

        self.adj_matrices: dict[Symbol, T] = {
            symbol: matrix_type(matrix) for symbol, matrix in matrices.items()
        }

    def accepts(self, word: Iterable[Symbol]) -> bool:
        """Interpreter function

        Args:
            word (Iterable[Symbol])

        Returns:
            bool
        """
        fa_word = list(word)
        confs: List[Tuple[List, int]] = [(fa_word, st) for st in self.start_states]

        while len(confs) != 0:
            conf = confs.pop()
            word = conf[0]
            st = conf[1]

            if not word:
                if st in self.final_states:
                    return True
                continue

            adj_matrice = self.adj_matrices[word[0]]
            if adj_matrice is None:
                continue

            for next_st in range(self.states_count):
                if adj_matrice[st, next_st]:
                    confs.append((word[1:], next_st))
        return False

    def transitive_closure(self):
        """Transitive closure for the states of the automata

        Returns:
            NDArray[bool_]
        """
        if not self.adj_matrices:
            return np.eye(self.states_count, self.states_count, dtype=bool_)

        combined = sum(self.adj_matrices.values())
        combined.setdiag(True)

        transitive_closure = combined.toarray()
        for power in range(2, self.states_count + 1):
            prev = transitive_closure
            transitive_closure = np.linalg.matrix_power(prev, power)
            if np.array_equal(prev, transitive_closure):
                break
        return transitive_closure

    def is_empty(self) -> bool:
        """Check whether the automata-generated language is empty or not

        Returns:
            bool
        """
        transitive_closure = self.transitive_closure()

        for start_st, final_st in product(self.start_states, self.final_states):
            return not transitive_closure[start_st, final_st]


def intersect_automata(
    automaton1: ExperimentalAdjacencyMatrixFA,
    automaton2: ExperimentalAdjacencyMatrixFA,
    matrix_type: T,
    out_matrix_format: F,
) -> ExperimentalAdjacencyMatrixFA:
    """Automaton intersection function

    Args:
        automaton1 (ExperimentalAdjacencyMatrixFA)
        automaton2 (ExperimentalAdjacencyMatrixFA)

    Returns:
        AdjacencyMatrixFA
    """
    intersection = ExperimentalAdjacencyMatrixFA(None, matrix_type)
    intersection.states_count = automaton1.states_count * automaton2.states_count

    for st1, st2 in product(automaton1.states.keys(), automaton2.states.keys()):
        st1_ind = automaton1.states[st1]
        st2_ind = automaton2.states[st2]
        intersection_ind = automaton2.states_count * st1_ind + st2_ind
        if st1_ind in automaton1.start_states and st2_ind in automaton2.start_states:
            intersection.start_states.add(intersection_ind)
        if st1_ind in automaton1.final_states and st2_ind in automaton2.final_states:
            intersection.final_states.add(intersection_ind)
        intersection.states[(st1, st2)] = intersection_ind

    for symbol, adj1 in automaton1.adj_matrices.items():
        if symbol not in automaton2.adj_matrices:
            continue

        adj2 = automaton2.adj_matrices[symbol]
        intersection.adj_matrices[symbol] = kron(adj1, adj2, format=out_matrix_format)

    return intersection


# * TENSOR_BASED_RPQ


def experimental_tensor_based_rpq(
    regex: str,
    graph: MultiDiGraph,
    start_nodes: set[int],
    final_nodes: set[int],
    matrix_type: T,
    out_matrix_format: F,
) -> set[tuple[int, int]]:
    """Regular Path Queries

    Args:
        graph (MultiDiGraph)
        start_nodes (set[int])
        final_nodes (set[int])
        regex (str)

    Returns:
        list[tuple[int, int]]
    """

    nodes = {int(node) for node in graph.nodes}
    start_nodes = start_nodes or nodes
    final_nodes = final_nodes or nodes

    graph_amfa = ExperimentalAdjacencyMatrixFA(
        graph_to_nfa(graph=graph, start_states=start_nodes, final_states=final_nodes),
        matrix_type,
    )
    regex_dfa = regex_to_dfa(regex=regex)
    regex_amfa = ExperimentalAdjacencyMatrixFA(regex_to_dfa(regex=regex), matrix_type)

    intersection_amfa = intersect_automata(
        graph_amfa, regex_amfa, matrix_type, out_matrix_format
    )
    intersection_tc = intersection_amfa.transitive_closure()
    result: set[tuple[int, int]] = set()
    if not intersection_tc.any():
        return result

    for start, final in product(start_nodes, final_nodes):
        for regex_start, regex_final in product(
            regex_dfa.start_states, regex_dfa.final_states
        ):
            if intersection_tc[
                intersection_amfa.states[(start, regex_start)],
                intersection_amfa.states[(final, regex_final)],
            ]:
                result.add((start, final))

    return result


# * MS_BFS_BASED_RPQ


def experimental_start_front(
    dfa: ExperimentalAdjacencyMatrixFA,
    nfa: ExperimentalAdjacencyMatrixFA,
    matrix_type: T,
):
    """Initialize the start front

    Args:
        dfa (ExperimentalAdjacencyMatrixFA): Regex DFA
        nfa (ExperimentalAdjacencyMatrixFA): Graph NFA

    Returns:
        matrix: front
    """
    dfa_start_state = list(dfa.start_states)[0]
    data = np.ones(len(nfa.start_states), dtype=bool)
    rows = [
        dfa_start_state + dfa.states_count * i for i in range(len(nfa.start_states))
    ]
    cols = list(nfa.start_states)

    if matrix_type in [dok_matrix, lil_matrix]:
        matrix = dok_matrix(
            (
                dfa.states_count * len(nfa.start_states),
                nfa.states_count,
            ),
            dtype=bool,
        )

        for i, r in enumerate(rows):
            matrix[r, cols[i]] = data[i]
        return matrix

    return matrix_type(
        (data, (rows, cols)),
        shape=(dfa.states_count * len(nfa.start_states), nfa.states_count),
        dtype=bool,
    )


def experimental_ms_bfs_based_rpq(
    regex: str,
    graph: MultiDiGraph,
    start_nodes: set[int],
    final_nodes: set[int],
    matrix_type: T,
    out_matrix_format: F,
) -> set[tuple[int, int]]:
    """Multi Source BFS Based Regular Path Queries

    Args:
        regex (str): Regex constraints
        graph (MultiDiGraph): Given graph
        start_nodes (set[int]): Start graph nodes
        final_nodes (set[int]): Final graph nodes

    Returns:
        set[tuple[int, int]]
    """
    regex_dfa = regex_to_dfa(regex)
    graph_nfa = graph_to_nfa(
        graph=graph, start_states=start_nodes, final_states=final_nodes
    )

    regex_amfa = ExperimentalAdjacencyMatrixFA(regex_dfa, matrix_type)
    graph_amfa = ExperimentalAdjacencyMatrixFA(graph_nfa, matrix_type)

    init_front = experimental_start_front(regex_amfa, graph_amfa, matrix_type)
    result: set[tuple[int, int]] = set()
    reg_amfa_transpose_matrices: dict[Symbol, T] = {}
    for symbol, matrix in regex_amfa.adj_matrices.items():
        reg_amfa_transpose_matrices[symbol] = matrix.transpose()

    visited = init_front
    symbols = [
        sym
        for sym in regex_amfa.adj_matrices.keys()
        if sym in graph_amfa.adj_matrices.keys()
    ]

    while init_front.toarray().any():
        next_fronts: dict[Symbol, T] = {}

        for symbol in symbols:
            next_fronts[symbol] = init_front @ graph_amfa.adj_matrices[symbol]

            for i in range(len(graph_amfa.start_states)):
                dfa_states_cnt = regex_amfa.states_count
                start_ind, end_ind = i * dfa_states_cnt, (i + 1) * dfa_states_cnt
                next_fronts[symbol][start_ind:end_ind] = (
                    reg_amfa_transpose_matrices[symbol]
                    @ next_fronts[symbol][start_ind:end_ind]
                )

        init_front = reduce(lambda x, y: x + y, next_fronts.values(), init_front)
        init_front = init_front > visited
        visited += init_front

    reversed_nfa_states = {value: key for key, value in graph_amfa.states.items()}

    for dfa_fn_state in regex_amfa.final_states:
        for i, nfa_start_state in enumerate(graph_amfa.start_states):
            for nfa_reached in visited.getrow(
                regex_amfa.states_count * i + dfa_fn_state
            ).indices:
                if nfa_reached in graph_amfa.final_states:
                    result.add(
                        (
                            reversed_nfa_states[nfa_start_state],
                            reversed_nfa_states[nfa_reached],
                        )
                    )

    return result
