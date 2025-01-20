from __future__ import annotations
from pyformlang.finite_automaton import Symbol
from pyformlang import rsa
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Set, Tuple
import networkx as nx

LABEL_NAME = "label"


@dataclass(frozen=True)
class SymbolState:
    symbol: Symbol
    state: str


@dataclass(frozen=True)
class NodeConnection:
    gss_node: GraphStateNode
    state: SymbolState
    origin: int


class GraphStateNode:
    state: SymbolState
    node_id: int
    transitions: Dict[SymbolState, Set[GraphStateNode]]
    visited: Set[int]

    def __init__(self, state: SymbolState, node_id: int):
        self.state = state
        self.node_id = node_id
        self.transitions = defaultdict(set)
        self.visited = set()

    def visit(self, node_id: int) -> Set[NodeConnection]:
        connections = set()
        if node_id not in self.visited:
            for state in self.transitions:
                for next_node in self.transitions[state]:
                    connections.add(NodeConnection(next_node, state, node_id))
            self.visited.add(node_id)
        return connections

    def add_transition(
        self, state: SymbolState, target_node: GraphStateNode
    ) -> Set[NodeConnection]:
        connections = set()
        current_transitions = self.transitions.get(state, set())
        if target_node not in current_transitions:
            current_transitions.add(target_node)
            for visited_node in self.visited:
                connections.add(NodeConnection(target_node, state, visited_node))
        self.transitions[state] = current_transitions
        return connections


class StateStack:
    nodes: Dict[Tuple[SymbolState, int], GraphStateNode]

    def __init__(self):
        self.nodes = {}

    def retrieve(self, state: SymbolState, node_id: int):
        if (state, node_id) not in self.nodes:
            self.nodes[(state, node_id)] = GraphStateNode(state, node_id)
        return self.nodes[(state, node_id)]


@dataclass
class TransitionData:
    terminal_transitions: Dict[Symbol, SymbolState]
    variable_transitions: Dict[Symbol, Tuple[SymbolState, SymbolState]]
    is_terminal_state: bool


class CFParser:
    def is_terminal(self, state: str) -> bool:
        return Symbol(state) not in self.symbol_state_data

    def initialize_graph(self, graph: nx.DiGraph):
        edges = graph.edges(data=LABEL_NAME)

        for node in graph.nodes():
            self.node_edges[node] = {}

        for source, target, label in edges:
            if label is not None:
                node_edges = self.node_edges[source]
                node_set = node_edges.get(label, set())
                node_set.add(target)
                node_edges[label] = node_set

    def initialize_rsm(self, rsm: rsa.RecursiveAutomaton):
        for var in rsm.boxes:
            self.symbol_state_data[var] = {}

        for var in rsm.boxes:
            box = rsm.boxes[var]
            fa = box.dfa
            graph = fa.to_networkx()
            state_data = self.symbol_state_data[var]

            for substate in graph.nodes:
                is_final = substate in fa.final_states
                state_data[substate] = TransitionData({}, {}, is_final)

            edges = graph.edges(data=LABEL_NAME)
            for from_state, to_state, label in edges:
                if label is not None:
                    state_edges = state_data[from_state]
                    if self.is_terminal(label):
                        state_edges.terminal_transitions[label] = SymbolState(
                            var, to_state
                        )
                    else:
                        inner_fa = rsm.boxes[Symbol(label)].dfa
                        inner_start = inner_fa.start_state.value
                        state_edges.variable_transitions[label] = (
                            SymbolState(Symbol(label), inner_start),
                            SymbolState(var, to_state),
                        )

        start_symbol = rsm.initial_label
        fa_start = rsm.boxes[start_symbol].dfa
        self.start_state = SymbolState(start_symbol, fa_start.start_state.value)

    def __init__(self, rsm: rsa.RecursiveAutomaton, graph: nx.DiGraph):
        self.start_state = None
        self.node_edges: Dict[int, Dict[Symbol, Set[int]]] = {}
        self.symbol_state_data: Dict[Symbol, Dict[str, TransitionData]] = {}
        self.start_state: SymbolState

        self.rsm = rsm
        self.graph = graph

        self.initialize_graph(graph)
        self.initialize_rsm(rsm)

        self.state_stack = StateStack()
        self.final_acceptance_node = self.state_stack.retrieve(
            SymbolState(Symbol("$"), "fin"), -1
        )

        self.pending_nodes: Set[NodeConnection] = set()
        self.processed_nodes: Set[NodeConnection] = set()

    def add_nodes(self, new_nodes: Set[NodeConnection]):
        new_nodes.difference_update(self.processed_nodes)
        self.processed_nodes.update(new_nodes)
        self.pending_nodes.update(new_nodes)

    def filter_visited_nodes(
        self, nodes: Set[NodeConnection], prev_node: NodeConnection
    ) -> Tuple[Set[NodeConnection], Set[Tuple[int, int]]]:
        node_connections = set()
        start_end_connections = set()

        for node in nodes:
            if node.gss_node == self.final_acceptance_node:
                start_node = prev_node.gss_node.node_id
                end_node = node.origin
                start_end_connections.add((start_node, end_node))
            else:
                node_connections.add(node)

        return node_connections, start_end_connections

    def execute_step(self, node: NodeConnection) -> Set[Tuple[int, int]]:
        symbol_state = node.state
        state_data = self.symbol_state_data[symbol_state.symbol][symbol_state.state]

        def terminal_step():
            terminal_transitions = state_data.terminal_transitions
            graph_edges = self.node_edges[node.origin]
            for terminal in terminal_transitions:
                if terminal in graph_edges:
                    new_nodes = set()
                    new_state = terminal_transitions[terminal]
                    graph_targets = graph_edges[terminal]
                    for target in graph_targets:
                        new_nodes.add(NodeConnection(node.gss_node, new_state, target))
                    self.add_nodes(new_nodes)

        def variable_step() -> Set[Tuple[int, int]]:
            start_end_connections = set()
            for var in state_data.variable_transitions:
                start_state, return_state = state_data.variable_transitions[var]
                inner_node = self.state_stack.retrieve(start_state, node.origin)
                new_nodes = inner_node.add_transition(return_state, node.gss_node)

                new_nodes, sub_start_end = self.filter_visited_nodes(new_nodes, node)
                self.add_nodes(new_nodes)
                self.add_nodes({NodeConnection(inner_node, start_state, node.origin)})
                start_end_connections.update(sub_start_end)

            return start_end_connections

        def pop_step() -> Set[Tuple[int, int]]:
            new_nodes = node.gss_node.visit(node.origin)
            new_nodes, start_end = self.filter_visited_nodes(new_nodes, node)
            self.add_nodes(new_nodes)
            return start_end

        terminal_step()
        result_set = variable_step()

        if state_data.is_terminal_state:
            result_set.update(pop_step())

        return result_set

    def resolve_reachability(
        self, from_nodes: Set[int], to_nodes: Set[int]
    ) -> Set[Tuple[int, int]]:
        reachable_nodes = set()
        for node in from_nodes:
            gss_node = self.state_stack.retrieve(self.start_state, node)
            gss_node.add_transition(
                SymbolState(Symbol("$"), "fin"), self.final_acceptance_node
            )

            self.add_nodes({NodeConnection(gss_node, self.start_state, node)})

        while self.pending_nodes:
            reachable_nodes.update(self.execute_step(self.pending_nodes.pop()))

        return {(start, end) for start, end in reachable_nodes if end in to_nodes}


def gll_based_cfpq(
    rsm: rsa.RecursiveAutomaton,
    graph: nx.DiGraph,
    start_nodes: Set[int] = None,
    final_nodes: Set[int] = None,
) -> Set[Tuple[int, int]]:
    if not start_nodes:
        start_nodes = set(graph.nodes())
    if not final_nodes:
        final_nodes = set(graph.nodes())

    solver = CFParser(rsm, graph)
    return solver.resolve_reachability(start_nodes, final_nodes)
