"""
Hellings Algorithm with Matrix Based Approach for Context-Free Path Quering
"""

from typing import Set

from networkx import DiGraph
from pyformlang.cfg import CFG, Terminal, Variable
from scipy.sparse import csr_matrix

from project.task6 import cfg_to_weak_normal_form


def matrix_based_cfpq(
    cfg: CFG,
    graph: DiGraph,
    start_nodes: Set[int] = None,
    final_nodes: Set[int] = None,
) -> set[tuple[int, int]]:
    """Matrix based approach for context-free path quering.

    Args:
        cfg (CFG): Context-Free Grammar.
        graph (DiGraph): Directed graph.

        start_nodes (Set[int], optional): Starting nodes. Defaults to None.
        final_nodes (Set[int], optional): Final nodes. Defaults to None.

    Returns:
        set[tuple[int, int]]: A set of tuples representing valid paths.
    """

    wcnf = cfg_to_weak_normal_form(cfg)
    total_nodes = graph.number_of_nodes()
    node2index = {node: ind for ind, node in enumerate(graph.nodes)}
    index2node = {ind: node for node, ind in node2index.items()}

    matrices: dict[Variable, csr_matrix] = {}
    for u, v, label in graph.edges(data="label"):
        if label:
            heads = [
                prod.head
                for prod in wcnf.productions
                if Terminal(label) in prod.body and len(prod.body) == 1
            ]

            for head in heads:
                if head not in matrices:
                    matrices[head] = csr_matrix((total_nodes, total_nodes), dtype=bool)
                matrices[head][node2index[u], node2index[v]] = True

    nullable_vars = wcnf.get_nullable_symbols()
    for node in graph.nodes:
        for var in nullable_vars:
            nullable_var = Variable(var.value)
            if nullable_var not in matrices:
                matrices[nullable_var] = csr_matrix(
                    (total_nodes, total_nodes), dtype=bool
                )
            matrices[nullable_var][node2index[node], node2index[node]] = True

    has_changes = True
    while has_changes:
        has_changes = False
        for prod in wcnf.productions:
            if len(prod.body) == 2:
                var1 = Variable(prod.body[0].value)
                var2 = Variable(prod.body[1].value)
                head = prod.head

                if var1 in matrices and var2 in matrices:
                    if head not in matrices:
                        matrices[head] = csr_matrix(
                            (total_nodes, total_nodes), dtype=bool
                        )

                    new_matrix = matrices[var1] @ matrices[var2]
                    coords = new_matrix.tocoo()

                    for row, col, value in zip(coords.row, coords.col, coords.data):
                        if value and not matrices[head][row, col]:
                            matrices[head][row, col] = True
                            has_changes = True

    result_pairs = set()
    if wcnf.start_symbol in matrices:
        final_matrix = matrices[wcnf.start_symbol].tocoo()
        for start_idx, end_idx in zip(final_matrix.row, final_matrix.col):
            start_node = index2node[start_idx]
            end_node = index2node[end_idx]
            if (not start_nodes or start_node in start_nodes) and (
                not final_nodes or end_node in final_nodes
            ):
                result_pairs.add((start_node, end_node))

    return result_pairs
