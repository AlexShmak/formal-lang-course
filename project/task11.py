from antlr4 import CommonTokenStream, ParserRuleContext, InputStream, ParseTreeWalker

from project.queryLanguage.QueryLanguageLexer import QueryLanguageLexer
from project.queryLanguage.QueryLanguageParser import QueryLanguageParser
from project.queryLanguage.listeners import CountListener, TokensListener


def program_to_tree(program: str) -> tuple[ParserRuleContext, bool]:
    input_stream = InputStream(program)
    lexer = QueryLanguageLexer(input_stream)
    token_stream = CommonTokenStream(lexer)
    parser = QueryLanguageParser(token_stream)
    tree = parser.prog()
    return tree, parser.getNumberOfSyntaxErrors() == 0


def nodes_count(tree: ParserRuleContext) -> int:
    listener = CountListener()
    ParseTreeWalker().walk(listener, tree)
    return listener.count


def tree_to_program(tree: ParserRuleContext) -> str:
    listener = TokensListener()
    ParseTreeWalker().walk(listener, tree)
    return " ".join(listener.tokens)
