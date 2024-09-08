import networkx as nx
import cfpq_data
from project.graphs import create_and_save_two_cycles_graph
from pydot import graph_from_dot_file
import os


def test_create_and_save_two_cycles_graphs():
    n, m = 13, 31
    labels = ("biba", "boba")
    nodes_count = n + m + 1
    output_path = "output.dot"

    create_and_save_two_cycles_graph(n, m, labels=labels, output_path=output_path)
    graph = graph_from_dot_file(output_path)[0]
    os.remove(output_path)

    g = nx.drawing.nx_pydot.from_pydot(graph)
    cycle = nx.find_cycle(g, orientation="original")
    expected_cycle_length = 14

    assert len(g.nodes) == nodes_count
    assert set(labels) == set(cfpq_data.get_sorted_labels(g))
    assert len(cycle) == expected_cycle_length
