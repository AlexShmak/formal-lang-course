from project.queryLanguage.interpreter.QueryLanguageInterpreterVisitor import (
    QueryLanguageInterpreterVisitor,
)
from project.queryLanguage.parser.parser import program_to_tree


def exec_program(program: str) -> dict[str, set[tuple]]:
    tree, is_valid = program_to_tree(program)
    runner_visitor = QueryLanguageInterpreterVisitor()

    try:
        runner_visitor.visit(tree)

        return runner_visitor.get_results()
    except Exception:
        print("Something went wrong")
        return {}
