from project.queryLanguage.parser.parser import program_to_tree
from project.queryLanguage.typing.QueryLanguageTypeInferenceVisitor import (
    QueryLanguageTypeInferenceVisitor,
)


def typing_program(program: str) -> bool:
    tree, is_valid = program_to_tree(program)
    if not is_valid:
        return False
    else:
        visitor = QueryLanguageTypeInferenceVisitor()
        try:
            visitor.visit(tree)
            return True
        except Exception:
            return False
