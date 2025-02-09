from typing import Tuple, Set, Optional, Any

from networkx import MultiDiGraph
from pyformlang.finite_automaton import EpsilonNFA

from project.queryLanguage.interpreter.utils import (
    nfa_from_char,
    nfa_from_var,
    group,
    intersect,
    concatenate,
    union,
    repeat_range,
    build_rsm,
)
from project.queryLanguage.utils.QueryLanguageParser import QueryLanguageParser
from project.queryLanguage.utils.QueryLanguageVisitor import QueryLanguageVisitor
from project.task8 import tensor_based_cfpq


class QueryLanguageInterpreterVisitor(QueryLanguageVisitor):
    def __init__(self):
        super(QueryLanguageVisitor, self).__init__()
        self.__variables = {}
        self.__results = {}
        self.__query_completed = False

    def get_results(self):
        return self.__results.copy()

    def visitDeclare(self, ctx: QueryLanguageParser.DeclareContext):
        var_name = self.__get_var_name(ctx.var())

        self.__variables[var_name] = MultiDiGraph()

    def visitBind(self, ctx: QueryLanguageParser.BindContext):
        var_name: str = self.__get_var_name(ctx.var())
        expr_value = self.visitExpr(ctx.expr())

        self.__variables[var_name] = expr_value

        if self.__query_completed:
            self.__query_completed = False

            self.__results[var_name] = expr_value

    def visitRegexp(self, ctx: QueryLanguageParser.RegexpContext):
        if ctx.char():
            return nfa_from_char(self.visitChar(ctx.char()))

        if ctx.var():
            return nfa_from_var(ctx.var().getText())

        if ctx.L_PARENTHESIS() and ctx.R_PARENTHESIS():
            return group(self.visitRegexp(ctx.regexp(0)))

        if ctx.CIRCUMFLEX():
            left_regexp = self.visitRegexp(ctx.regexp(0))
            _range = self.visitRange(ctx.range_())
            return repeat_range(
                left_regexp, self.visitNum(_range[0]), self.visitNum(_range[1])
            )

        left_regexp = self.visitRegexp(ctx.regexp(0))
        right_regexp = self.visitRegexp(ctx.regexp(1))

        if ctx.PIPE():
            return union(left_regexp, right_regexp)

        if ctx.DOT():
            return concatenate(left_regexp, right_regexp)

        if ctx.AMPERSAND():
            return intersect(left_regexp, right_regexp)

    def visitSelect(self, ctx: QueryLanguageParser.SelectContext):
        # Extract information from variable filters
        for_var_1, for_nodes_1 = self.visitVar_filter(ctx.var_filter(0))
        for_var_2, for_nodes_2 = self.visitVar_filter(ctx.var_filter(1))

        # Retrieve the graph and build the query
        graph: MultiDiGraph = self.visitVar(ctx.var()[-1])
        nfa_subs_dict = self.__build_nfa_substitutions()
        query = build_rsm(self.__expr_to_nfa(ctx.expr()), nfa_subs_dict)

        # Map start and final variable names to nodes
        start_var_name = self.__get_var_name(ctx.var()[-2])
        final_var_name = self.__get_var_name(ctx.var()[-3])
        start_nodes = self.__resolve_nodes(
            start_var_name, for_var_1, for_var_2, for_nodes_1, for_nodes_2
        )
        final_nodes = self.__resolve_nodes(
            final_var_name, for_var_1, for_var_2, for_nodes_1, for_nodes_2
        )

        # Perform context-free path querying
        cfpq_result = tensor_based_cfpq(query, graph, start_nodes, final_nodes)

        # Determine the selection result
        select_result = self.__get_select_result(ctx, cfpq_result)
        self.__query_completed = True
        return select_result

    def visitVar_filter(
        self, ctx: QueryLanguageParser.Var_filterContext
    ) -> tuple[None, None] | tuple[Any, Any]:
        if not ctx:
            return None, None

        var_name = ctx.var().getText()
        set_expr = self.visitExpr(ctx.expr())

        return var_name, set_expr

    def visitAdd(self, ctx: QueryLanguageParser.AddContext):
        graph: MultiDiGraph = self.visitVar(ctx.var())

        if ctx.EDGE():
            edge: Tuple[int, str, int] = self.visitExpr(ctx.expr())

            graph.add_edge(edge[0], edge[2], label=edge[1])
        else:
            vertex: int = self.visitExpr(ctx.expr())

            graph.add_node(vertex)

    def visitRemove(self, ctx: QueryLanguageParser.RemoveContext):
        graph: MultiDiGraph = self.visitVar(ctx.var())

        if ctx.EDGE():
            edge: Tuple[int, str, int] = self.visitExpr(ctx.expr())

            graph.remove_edge(edge[0], edge[2])
        elif ctx.VERTEX():
            vertex: int = self.visitExpr(ctx.expr())

            graph.remove_node(vertex)
        else:
            vertices: set[int] = self.visitExpr(ctx.expr())

            for vertex in vertices:
                graph.remove_node(vertex)

    def visitSet_expr(self, ctx: QueryLanguageParser.Set_exprContext) -> Set:
        expressions = ctx.expr()

        result_set: Set[int] = set()

        for expr in expressions:
            num = self.visitExpr(expr)
            result_set.add(num)

        return result_set

    def visitEdge_expr(
        self, ctx: QueryLanguageParser.Edge_exprContext
    ) -> Tuple[int, str, int]:
        edge_expressions = ctx.expr()

        left_num = self.visitExpr(edge_expressions[0])
        char = self.visitExpr(edge_expressions[1])
        right_num = self.visitExpr(edge_expressions[2])

        edge = (left_num, char, right_num)

        return edge

    def visitRange(
        self, ctx: QueryLanguageParser.RangeContext
    ) -> Tuple[int, Optional[int]]:
        return ctx.num(0), ctx.num(1)

    def visitNum(self, ctx: QueryLanguageParser.NumContext):
        return int(ctx.NUM().getText())

    def visitChar(self, ctx: QueryLanguageParser.CharContext):
        return str(ctx.CHAR().getText()[1])

    def visitVar(self, ctx: QueryLanguageParser.VarContext):
        var_name = self.__get_var_name(ctx)
        if var_name not in self.__variables:
            raise Exception(f"Variable '{var_name}' is not defined.")

        return self.__variables[var_name]

    def __get_var_name(self, ctx: QueryLanguageParser.VarContext) -> str:
        return str(ctx.VAR().getText())

    def __build_nfa_substitutions(self) -> dict[str, EpsilonNFA]:
        """Builds a dictionary of NFA substitutions from variable values."""
        nfa_subs_dict = {}
        for key, value in self.__variables.items():
            if isinstance(value, EpsilonNFA):
                nfa_subs_dict[key] = value
            elif isinstance(value, str):
                nfa_subs_dict[key] = nfa_from_char(value)
        return nfa_subs_dict

    def __resolve_nodes(
        self, target_var: str, var_1: str, var_2: str, nodes_1: set, nodes_2: set
    ) -> set:
        """Resolves the nodes corresponding to the target variable."""
        if target_var == var_1:
            return nodes_1
        elif target_var == var_2:
            return nodes_2
        raise ValueError(f"Variable '{target_var}' is not defined.")

    def __get_select_result(
        self, ctx: QueryLanguageParser.SelectContext, cfpq_result: set[tuple]
    ) -> set:
        """Determines the selection result based on the query context."""
        select_result = set()

        first_return_var = ctx.var(0).getText()
        second_return_var = ctx.var(1).getText() if len(ctx.var()) > 1 else None

        start_var_name = ctx.var()[-2].getText()
        final_var_name = ctx.var()[-3].getText()

        if first_return_var == start_var_name and not second_return_var:
            select_result = {res[0] for res in cfpq_result}
        elif first_return_var == final_var_name and not second_return_var:
            select_result = {res[1] for res in cfpq_result}
        else:
            select_result = cfpq_result

        return select_result

    def __expr_to_nfa(self, ctx: QueryLanguageParser.ExprContext) -> EpsilonNFA:
        expr_value = self.visitExpr(ctx)

        if isinstance(expr_value, EpsilonNFA):
            return expr_value
        elif isinstance(expr_value, str):
            return nfa_from_char(expr_value)
        else:
            raise Exception(
                f"illegal type '{expr_value}'. Can't convert to EpsilonNFA."
            )
