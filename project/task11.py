from antlr4 import InputStream, CommonTokenStream, ParserRuleContext, TerminalNode
from project.queryLanguage.QueryLanguageLexer import QueryLanguageLexer
from project.queryLanguage.QueryLanguageParser import QueryLanguageParser


# Function to parse the program
def program_to_tree(program: str) -> tuple[None, bool] | tuple[ParserRuleContext, bool]:
    input_stream = InputStream(program)
    lexer = QueryLanguageLexer(input_stream)
    token_stream = CommonTokenStream(lexer)
    parser = QueryLanguageParser(token_stream)
    tree = parser.prog()  # Start from the prog rule

    num_errors = parser.getNumberOfSyntaxErrors()
    if num_errors > 0:
        return None, False
    return tree, True  # Return the tree and validity


# Function to count nodes in the tree
def nodes_count(tree: ParserRuleContext) -> int:
    count = 1
    for child in tree.children:
        if isinstance(child, ParserRuleContext):
            count += 1

    return count


# Function to convert the tree back to a string
def tree_to_program(tree: ParserRuleContext) -> str:
    program = ""
    for child in tree.children:
        if isinstance(child, ParserRuleContext):
            program += tree_to_program(child)
        if isinstance(child, TerminalNode):
            program += child.getText() + " "
    return program
