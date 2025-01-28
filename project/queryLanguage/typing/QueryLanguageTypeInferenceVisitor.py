from typing import Dict

from project.queryLanguage.typing.QueryLanguageType import QueryLanguageType
from project.queryLanguage.utils.QueryLanguageParser import QueryLanguageParser
from project.queryLanguage.utils.QueryLanguageVisitor import QueryLanguageVisitor


class QueryLanguageTypeInferenceVisitor(QueryLanguageVisitor):
    def __init__(self):
        super(QueryLanguageVisitor, self).__init__()
        self.__variables: Dict[str, QueryLanguageType] = {}

    def visitDeclare(self, ctx: QueryLanguageParser.DeclareContext) -> None:
        var_name = self.__get_var_name(ctx.var())

        self.__variables[var_name] = QueryLanguageType.GRAPH

    def visitAdd(self, ctx: QueryLanguageParser.AddContext) -> None:
        var_name = self.__get_var_name(ctx.var())
        self.__check_graph_variable(var_name)

        added_entity_type = self.visitExpr(ctx.expr())

        if ctx.EDGE() and added_entity_type != QueryLanguageType.EDGE:
            raise Exception(
                f"Illegal edge type ({added_entity_type}). Can't be added to '{var_name}'."
            )
        elif ctx.VERTEX() and added_entity_type != QueryLanguageType.NUM:
            raise Exception(
                f"Illegal vertex type ({added_entity_type}). Can't be added to '{var_name}'."
            )
        else:
            return

    def visitRemove(self, ctx: QueryLanguageParser.RemoveContext) -> None:
        var_name = self.__get_var_name(ctx.var())
        self.__check_graph_variable(var_name)

        removed_entity_type = self.visitExpr(ctx.expr())

        if ctx.EDGE() and removed_entity_type != QueryLanguageType.EDGE:
            raise Exception(
                f"Illegal edge type ({removed_entity_type}). Can't be removed from '{var_name}'."
            )
        elif ctx.VERTEX() and removed_entity_type != QueryLanguageType.NUM:
            raise Exception(
                f"Illegal vertex type ({removed_entity_type}). Can't be removed from '{var_name}'."
            )
        elif ctx.VERTICES() and removed_entity_type != QueryLanguageType.SET:
            raise Exception(
                f"Illegal vertices type ({removed_entity_type}). Can't be removed from '{var_name}'"
            )
        else:
            return

    def visitBind(self, ctx: QueryLanguageParser.BindContext) -> None:
        var_name: str = self.__get_var_name(ctx.var())
        bind_expr_type = self.visitExpr(ctx.expr())

        self.__variables[var_name] = bind_expr_type

    def visitRegexp(self, ctx: QueryLanguageParser.RegexpContext) -> QueryLanguageType:
        regexp_in_parens = ctx.L_PARENTHESIS() and ctx.R_PARENTHESIS()

        if ctx.char():
            return self.__regexp_char()
        elif ctx.var():
            return self.__regexp_var(ctx)

        elif regexp_in_parens:
            return self.__regexp_paren(ctx)
        else:
            if ctx.CIRCUMFLEX():
                return self.__regex_circumflex(ctx)

            else:
                left_regexp, right_regexp = ctx.regexp(0), ctx.regexp(1)

                left_regexp_type, right_regexp_type = (
                    self.visitRegexp(left_regexp),
                    self.visitRegexp(right_regexp),
                )

                if ctx.PIPE() or ctx.DOT():
                    return self.__automaton_type_infer(
                        left_regexp_type, right_regexp_type
                    )
                elif ctx.AMPERSAND():
                    if (
                        left_regexp_type == QueryLanguageType.RSM
                        and right_regexp_type == QueryLanguageType.RSM
                    ):
                        raise Exception(
                            f'Can\'t intersect two RSMs: "{ctx.getText()}".'
                        )

                    return self.__automaton_type_infer(left_regexp, right_regexp)

        return QueryLanguageType.UNKNOWN

    def visitSelect(self, ctx: QueryLanguageParser.SelectContext) -> QueryLanguageType:
        self.visitVar_filter(ctx.var_filter(0))
        self.visitVar_filter(ctx.var_filter(1))

        var_list: list = ctx.var()

        in_var = self.__get_var_name(var_list[-1])
        from_var = self.__get_var_name(var_list[-2])
        where_var = self.__get_var_name(var_list[-3])

        self.__check_graph_variable(in_var)

        result_var_1 = self.__get_var_name(var_list[0])
        result_var_2 = self.__get_var_name(var_list[1]) if ctx.COMMA() else None

        condition = result_var_1 not in [where_var, from_var]

        if condition:
            raise Exception(
                f"Result variable ('{result_var_1}') should be either '{from_var}' or '{where_var}'."
            )

        if result_var_2 and condition:
            raise Exception(
                f"Result variable ('{result_var_2}') should be either '{from_var}' or '{where_var}'."
            )

        expr_type = self.visitExpr(ctx.expr())

        if expr_type not in [
            QueryLanguageType.FA,
            QueryLanguageType.RSM,
            QueryLanguageType.CHAR,
        ]:
            raise Exception(f"Illegal expression type ('{expr_type}') in 'Select'.")

        return QueryLanguageType.SET if not result_var_2 else QueryLanguageType.PAIR_SET

    def visitVar_filter(self, ctx: QueryLanguageParser.Var_filterContext) -> str | None:
        if not ctx:
            return None

        var_name = self.__get_var_name(ctx.var())

        if self.__variables.__contains__(var_name):
            raise Exception(
                f"Variable '{var_name}' already exists in global context. It can't be used in 'For'."
            )

        expr_type = self.visitExpr(ctx.expr())

        if expr_type == QueryLanguageType.SET:
            return var_name
        else:
            raise Exception(
                f"Illegal expression type ('{ctx.getText()}' => '{expr_type}') in 'Filter'."
            )

    def visitSet_expr(
        self, ctx: QueryLanguageParser.Set_exprContext
    ) -> QueryLanguageType:
        expressions = ctx.expr()

        for expr in expressions:
            expr_type = self.visitExpr(expr)
            if expr_type != QueryLanguageType.NUM:
                raise Exception(
                    f"Illegal expression type ('{ctx.getText()}' => '{expr_type}') in 'Set'."
                )

        return QueryLanguageType.SET

    def visitEdge_expr(
        self, ctx: QueryLanguageParser.Edge_exprContext
    ) -> QueryLanguageType:
        edge_expressions = ctx.expr()

        left_num_check = self.visitExpr(edge_expressions[0]) == QueryLanguageType.NUM
        char_check = self.visitExpr(edge_expressions[1]) == QueryLanguageType.CHAR
        right_num_check = self.visitExpr(edge_expressions[2]) == QueryLanguageType.NUM

        edge_check = left_num_check and char_check and right_num_check

        if edge_check:
            return QueryLanguageType.EDGE
        else:
            raise Exception(f"Illegal edge construction ('{ctx.getText()}')")

    def visitRange(self, ctx: QueryLanguageParser.RangeContext) -> QueryLanguageType:
        return QueryLanguageType.RANGE

    def visitNum(self, ctx: QueryLanguageParser.NumContext) -> QueryLanguageType:
        return QueryLanguageType.NUM

    def visitChar(self, ctx: QueryLanguageParser.CharContext) -> QueryLanguageType:
        return QueryLanguageType.CHAR

    def visitVar(self, ctx: QueryLanguageParser.VarContext) -> QueryLanguageType:
        var_name = self.__get_var_name(ctx)
        if not self.__variables.__contains__(var_name):
            raise Exception(f"Variable '{var_name}' is not defined.")

        return self.__get_var_type(var_name)

    def __regexp_char(self) -> QueryLanguageType:
        return QueryLanguageType.FA

    def __regexp_var(self, ctx: QueryLanguageParser.RegexpContext) -> QueryLanguageType:
        var_name = self.__get_var_name(ctx.var())

        if not self.__variables.__contains__(var_name):
            return QueryLanguageType.RSM
        else:
            var_type = self.visitVar(ctx.var())

            if var_type in [QueryLanguageType.FA, QueryLanguageType.CHAR]:
                return QueryLanguageType.FA

            elif var_type == QueryLanguageType.RSM:
                return QueryLanguageType.RSM

            else:
                raise Exception(
                    f"Illegal variable type '{var_type}' occurred in regexp: '{var_name}'"
                )

    def __regexp_paren(
        self, ctx: QueryLanguageParser.RegexpContext
    ) -> QueryLanguageType:
        return self.visitRegexp(ctx.regexp(0))

    def __regex_circumflex(
        self, ctx: QueryLanguageParser.RegexpContext
    ) -> QueryLanguageType:
        left_regexp, _range = ctx.regexp(0), ctx.range_()

        left_regexp_type = self.visitRegexp(left_regexp)
        range_type = self.visitRange(_range)

        if range_type != QueryLanguageType.RANGE:
            raise Exception(
                f"Illegal type in regular expression: '{range_type}' instead of '{QueryLanguageType.RANGE}"
            )

        if left_regexp_type == QueryLanguageType.FA:
            return QueryLanguageType.FA
        elif left_regexp_type == QueryLanguageType.RSM:
            return QueryLanguageType.RSM
        else:
            raise Exception(
                f"Illegal type ('{left_regexp_type}')  in 'Repeat' (^) operation."
            )

    def __automaton_type_infer(
        self, left_regexp_type, right_regexp_type
    ) -> QueryLanguageType:
        return (
            QueryLanguageType.RSM
            if left_regexp_type == QueryLanguageType.RSM
            or right_regexp_type == QueryLanguageType.RSM
            else QueryLanguageType.FA
        )

    def __get_var_name(self, ctx: QueryLanguageParser.VarContext) -> str:
        return str(ctx.VAR().getText())

    def __get_var_type(self, var) -> QueryLanguageType:
        if var in self.__variables:
            return self.__variables[var]
        else:
            raise Exception(f"Variable {var} not found.")

    def __check_graph_variable(self, var_name) -> None:
        var_type = self.__get_var_type(var_name)

        if var_type != QueryLanguageType.GRAPH:
            raise Exception(
                f"Variable '{var_name}' must be '{QueryLanguageType.GRAPH}', not '{var_type}'!"
            )
