"""
Chomskiy Weak Normal Form and Hellings ALgorithm
"""

from networkx import DiGraph
from pyformlang.cfg import CFG, Production, Terminal


def cfg_to_weak_normal_form(cfg: CFG) -> CFG:
    """Context Free Grammar to Chomsky Normal Form

    Args:
        cfg (CFG)

    Returns:
        CFG
    """
    nullable_symbols = cfg.get_nullable_symbols()
    productions: set[Production] = set(cfg.to_normal_form().productions)
    for sym in nullable_symbols:
        productions.add(Production(sym, []))
    new_cfg = CFG(cfg.variables, cfg.terminals, cfg.start_symbol, productions)
    return new_cfg


def hellings_based_cfpq(
    cfg: CFG,
    graph: DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    """Hellings Algorithm Based Context Free Grammar Path Quering

    Args:
        cfg (CFG)
        graph (DiGraph)
        start_nodes (set[int], optional):  Defaults to None.
        final_nodes (set[int], optional):  Defaults to None.

    Returns:
        set[tuple[int, int]]
    """
    wcnf = cfg_to_weak_normal_form(cfg)
    m = set()

    for u, v, label in graph.edges(data="label"):
        if not label:
            continue

        variables = [
            prod.head
            for prod in wcnf.productions
            if Terminal(label) in prod.body and len(prod.body) == 1
        ]
        for var in variables:
            m.add((var, u, v))

    nullable = wcnf.get_nullable_symbols()
    for var in nullable:
        for node in graph.nodes:
            m.add((var, node, node))

    r = m.copy()

    while m:
        (var1, u1, v1) = m.pop()
        new = set()
        for var2, u2, v2 in r:
            if v2 == u1:
                body = [var2, var1]
                heads = {prod.head for prod in wcnf.productions if prod.body == body}
                for head in heads:
                    new_path = (head, u2, v1)
                    if new_path not in r:
                        new.add(new_path)
                        m.add(new_path)

            if v1 == u2:
                body = [var1, var2]
                heads = {prod.head for prod in wcnf.productions if prod.body == body}
                for head in heads:
                    new_path = (head, u1, v2)
                    if new_path not in r:
                        new.add(new_path)
                        m.add(new_path)

        r.update(new)

    result_pairs = set()
    for var, u, v in r:
        if var == wcnf.start_symbol:
            if (not start_nodes or u in start_nodes) and (
                not final_nodes or v in final_nodes
            ):
                result_pairs.add((u, v))

    return result_pairs
