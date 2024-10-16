"""
Graph class to get information about `cfpq_data` graphs
"""

from typing import Tuple
from pathlib import Path

import cfpq_data as cd
import networkx as nx


class GraphInfo:
    """Graph class to manage data of a given graph (by name)"""

    def __init__(self, name):
        self._graph = cd.graph_from_csv(cd.download(name))
        self._nodes = self._graph.nodes
        self._edges = self._graph.edges
        self._labels = cd.get_sorted_labels(self._graph)

    @property
    def nodes(self):
        """Get the number of nodes of the graph"""
        return len(self._nodes)

    @property
    def edges(self):
        """Get the number of edges of the graph"""
        return len(self._edges)

    @property
    def labels(self):
        """Get the ordered labels of the graph"""
        return self._labels


def save_graph_to_file(graph: nx.MultiDiGraph, output_path: Path) -> None:
    """Save the graph to a DOT file (`pydot`)"""
    pydot_gr = nx.drawing.nx_pydot.to_pydot(graph)
    pydot_gr.write_raw(output_path)


def create_and_save_two_cycles_graph(
    n: int, m: int, labels: Tuple[str], output_path: Path
):
    """Create two cycles graph

    Args:
        n (int): The number of nodes in the first cycle
        m (int): The number of nodes in the second cycle
        labels (Tuple[str,str]): Labels that will be used to mark the edges of the grapgh
    """
    graph = cd.labeled_two_cycles_graph(n=n, m=m, labels=labels)
    save_graph_to_file(graph=graph, output_path=output_path)
