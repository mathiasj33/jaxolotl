import functools
from abc import ABC, abstractmethod
from typing import override

from jaxolotl.ltl.progression.ltl_lexer import LTLLexer, LTLToken, LTLTokenType


class LTLParser:
    def __init__(self, formula: str):
        lexer = LTLLexer(formula)
        self.tokens: list[LTLToken] = lexer.lex()
        self.pos = 0
        self.current_token: LTLToken | None = (
            self.tokens[self.pos] if self.tokens else None
        )
        self.formula = formula

    def parse(self) -> "LTLNode":
        result = self.parse_expression()
        if self.current_token is not None:
            raise SyntaxError(
                f"Unexpected token at the end: {self.current_token}. Formula: {self.formula}"
            )
        return result

    def parse_expression(self) -> "LTLNode":
        node = self.parse_implication()
        while self.match(LTLTokenType.OR):
            right = self.parse_implication()
            node = OrNode(node, right)
        return node

    def parse_implication(self) -> "LTLNode":
        node = self.parse_or()
        while self.match(LTLTokenType.IMPLIES):
            right = self.parse_or()
            node = ImplicationNode(node, right)
        return node

    def parse_or(self) -> "LTLNode":
        node = self.parse_and()
        while self.match(LTLTokenType.OR):
            right = self.parse_and()
            node = OrNode(node, right)
        return node

    def parse_and(self) -> "LTLNode":
        node = self.parse_until()
        while self.match(LTLTokenType.AND):
            right = self.parse_until()
            node = AndNode(node, right)
        return node

    def parse_until(self) -> "LTLNode":
        node = self.parse_primary()
        while self.match(LTLTokenType.UNTIL):
            right = self.parse_primary()
            node = UntilNode(node, right)
        return node

    def parse_primary(self) -> "LTLNode":  # noqa: PLR0911
        if self.match(LTLTokenType.NOT):
            return NotNode(self.parse_primary())
        elif self.match(LTLTokenType.EVENTUALLY):
            return EventuallyNode(self.parse_primary())
        elif self.match(LTLTokenType.ALWAYS):
            return AlwaysNode(self.parse_primary())
        elif self.match(LTLTokenType.LPAREN):
            node = self.parse_expression()
            if not self.match(LTLTokenType.RPAREN):
                raise SyntaxError(f"Expected ')'. Formula: {self.formula}")
            return node
        elif self.match(LTLTokenType.TRUE):
            return TrueNode()
        elif self.match(LTLTokenType.FALSE):
            return FalseNode()
        else:
            return self.parse_variable()

    def parse_variable(self) -> "LTLNode":
        if self.current_token and self.current_token.type == LTLTokenType.VAR:
            node = VarNode(self.current_token.value)
            self.next_token()
            return node
        else:
            raise SyntaxError(
                f"Unexpected token: {self.current_token}. Formula: {self.formula}"
            )

    def match(self, token_type: LTLTokenType) -> bool:
        if self.current_token and self.current_token.type == token_type:
            self.next_token()
            return True
        return False

    def next_token(self) -> None:
        self.pos += 1
        if self.pos < len(self.tokens):
            self.current_token = self.tokens[self.pos]
        else:
            self.current_token = None


class LTLNode(ABC):
    pass

    @abstractmethod
    def __eq__(self, other) -> bool:
        pass

    @abstractmethod
    def __hash__(self) -> int:
        pass

    @property
    @abstractmethod
    def children(self) -> list["LTLNode"]:
        pass

    @property
    def num_nodes(self) -> int:
        """Returns the number of nodes in the subtree rooted at this node."""
        return 1 + sum(child.num_nodes for child in self.children)

    @property
    def num_edges(self) -> int:
        """Returns the number of edges in the subtree rooted at this node."""
        return sum(child.num_edges for child in self.children) + len(self.children)


class AndNode(LTLNode):
    def __init__(self, left: LTLNode, right: LTLNode):
        self.left = left
        self.right = right

    def __repr__(self) -> str:
        return f"({self.left} & {self.right})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, AndNode):
            return False
        return self.left == other.left and self.right == other.right

    def __hash__(self) -> int:
        return hash((self.left, self.right, "and"))

    @property
    @override
    def children(self) -> list[LTLNode]:
        return [self.left, self.right]


class OrNode(LTLNode):
    def __init__(self, left: LTLNode, right: LTLNode):
        self.left = left
        self.right = right

    def __repr__(self) -> str:
        return f"({self.left} | {self.right})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, OrNode):
            return False
        return self.left == other.left and self.right == other.right

    def __hash__(self) -> int:
        return hash((self.left, self.right, "or"))

    @property
    @override
    def children(self) -> list[LTLNode]:
        return [self.left, self.right]


class NotNode(LTLNode):
    def __init__(self, operand: LTLNode):
        self.operand = operand

    def __repr__(self) -> str:
        return f"!({self.operand})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, NotNode):
            return False
        return self.operand == other.operand

    def __hash__(self) -> int:
        return hash((self.operand, "not"))

    @property
    @override
    def children(self) -> list[LTLNode]:
        return [self.operand]


class VarNode(LTLNode):
    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        return self.name

    def __eq__(self, other) -> bool:
        if not isinstance(other, VarNode):
            return False
        return self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)

    @property
    @override
    def children(self) -> list[LTLNode]:
        return []


class ImplicationNode(LTLNode):
    def __init__(self, left: LTLNode, right: LTLNode):
        self.left = left
        self.right = right

    def __repr__(self) -> str:
        return f"({self.left} -> {self.right})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, ImplicationNode):
            return False
        return self.left == other.left and self.right == other.right

    def __hash__(self) -> int:
        return hash((self.left, self.right, "implies"))

    @property
    @override
    def children(self) -> list[LTLNode]:
        return [self.left, self.right]


class EventuallyNode(LTLNode):
    def __init__(self, operand: LTLNode):
        self.operand = operand

    def __repr__(self) -> str:
        return f"F({self.operand})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, EventuallyNode):
            return False
        return self.operand == other.operand

    def __hash__(self) -> int:
        return hash((self.operand, "eventually"))

    @property
    @override
    def children(self) -> list[LTLNode]:
        return [self.operand]


class AlwaysNode(LTLNode):
    def __init__(self, operand: LTLNode):
        self.operand = operand

    def __repr__(self) -> str:
        return f"G({self.operand})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, AlwaysNode):
            return False
        return self.operand == other.operand

    def __hash__(self) -> int:
        return hash((self.operand, "always"))

    @property
    @override
    def children(self) -> list[LTLNode]:
        return [self.operand]


class UntilNode(LTLNode):
    def __init__(self, left: LTLNode, right: LTLNode):
        self.left = left
        self.right = right

    def __repr__(self) -> str:
        return f"({self.left} U {self.right})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, UntilNode):
            return False
        return self.left == other.left and self.right == other.right

    def __hash__(self) -> int:
        return hash((self.left, self.right, "until"))

    @property
    @override
    def children(self) -> list[LTLNode]:
        return [self.left, self.right]


class TrueNode(LTLNode):
    def __repr__(self) -> str:
        return "true"

    def __eq__(self, other) -> bool:
        return isinstance(other, TrueNode)

    def __hash__(self) -> int:
        return hash("true")

    @property
    @override
    def children(self) -> list[LTLNode]:
        return []


class FalseNode(LTLNode):
    def __repr__(self) -> str:
        return "false"

    def __eq__(self, other):
        return isinstance(other, FalseNode)

    def __hash__(self) -> int:
        return hash("false")

    @property
    @override
    def children(self) -> list[LTLNode]:
        return []


@functools.lru_cache(maxsize=500_000)
def parse(expression: str) -> LTLNode:
    return LTLParser(expression).parse()


if __name__ == "__main__":
    formula = "G(a => F b) & F c"
    parser = LTLParser(formula)
    ast = parser.parse()
    print(type(ast))
    print(str(formula))
