"""
Graph class to get information about `cfpq_data` grapghs
"""

import cfpq_data
import networkx as nx
from typing import Tuple


class Graph:
    """Graph class to manage data of a given graph (by name)"""

    def __init__(self, name):
        csv_path = cfpq_data.download(name)
        self.graph = cfpq_data.graph_from_csv(csv_path)

    @property
    def nodes(self):
        """Get the number of nodes of the graph"""
        return self.graph.number_of_nodes()

    @property
    def edges(self):
        """Get the number of edges of the graph"""
        return self.graph.number_of_edges()

    @property
    def labels(self):
        """Get the ordered labels of the graph"""
        return cfpq_data.get_sorted_labels(self.graph)


def save_graph_to_file(graph, output_path):
    """Save the graph to a DOT file (`pydot`)"""
    pydot_gr = nx.drawing.nx_pydot.to_pydot(graph)
    pydot_gr.write_raw(output_path)


def create_and_save_two_cycles_graph(n, m, labels: Tuple[str, str], output_path):
    """Create two cycles graph

    Args:
        n (int): The number of nodes in the first cycle
        m (int): The number of nodes in the second cycle
        labels (Tuple[str,str]): Labels that will be used to mark the edges of the grapgh
    """
    graph = cfpq_data.labeled_two_cycles_graph(n, m, labels=labels)
    save_graph_to_file(graph=graph, output_path=output_path)
