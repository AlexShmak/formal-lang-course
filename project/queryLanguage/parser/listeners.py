from antlr4 import ParseTreeListener, ParserRuleContext, TerminalNode


class CountListener(ParseTreeListener):
    def __init__(self):
        super().__init__()
        self.count = 0

    def enterEveryRule(self, ctx: ParserRuleContext):
        self.count += 1


class TokensListener(ParseTreeListener):
    def __init__(self):
        super().__init__()
        self.tokens = []

    def visitTerminal(self, node: TerminalNode):
        self.tokens.append(node.getText())
