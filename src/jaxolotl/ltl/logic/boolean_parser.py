import functools
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

from jaxolotl.ltl.logic.boolean_lexer import Lexer, Token, TokenType

if TYPE_CHECKING:
    from jaxolotl.ltl.logic.assignment import Assignment


class Parser:
    def __init__(self, expression: str):
        lexer = Lexer(expression)
        self.tokens: list[Token] = lexer.lex()
        self.pos = 0
        self.current_token: Token | None = (
            self.tokens[self.pos] if self.tokens else None
        )

    def parse(self) -> "BooleanNode":
        result = self.parse_expression()
        if self.current_token is not None:
            raise SyntaxError(f"Unexpected token at the end: {self.current_token}")
        return result

    def parse_expression(self) -> "BooleanNode":
        node = self.parse_implication()
        while self.match(TokenType.OR):
            right = self.parse_implication()
            node = OrNode(node, right)
        return node

    def parse_implication(self) -> "BooleanNode":
        node = self.parse_or()
        while self.match(TokenType.IMPLIES):
            right = self.parse_or()
            node = ImplicationNode(node, right)
        return node

    def parse_or(self) -> "BooleanNode":
        node = self.parse_and()
        while self.match(TokenType.OR):
            right = self.parse_and()
            node = OrNode(node, right)
        return node

    def parse_and(self) -> "BooleanNode":
        node = self.parse_primary()
        while self.match(TokenType.AND):
            right = self.parse_primary()
            node = AndNode(node, right)
        return node

    def parse_primary(self) -> "BooleanNode":
        if self.match(TokenType.NOT):
            return NotNode(self.parse_primary())
        elif self.match(TokenType.LPAREN):
            node = self.parse_expression()
            if not self.match(TokenType.RPAREN):
                raise SyntaxError("Expected ')'")
            return node
        else:
            return self.parse_variable()

    def parse_variable(self) -> "BooleanNode":
        if self.current_token and self.current_token.type == TokenType.VAR:
            node = VarNode(self.current_token.value)
            self.next_token()
            return node
        else:
            raise SyntaxError(f"Unexpected token: {self.current_token}")

    def match(self, token_type: TokenType) -> bool:
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


class BooleanNode(ABC):
    @abstractmethod
    def eval(self, assignment: "Assignment") -> bool:
        pass

    @abstractmethod
    def __eq__(self, other) -> bool:
        pass

    @abstractmethod
    def __hash__(self) -> int:
        pass

    @abstractmethod
    def num_nodes(self) -> int:
        pass

    @abstractmethod
    def num_edges(self) -> int:
        pass


class AndNode(BooleanNode):
    def __init__(self, left: BooleanNode, right: BooleanNode):
        self.left = left
        self.right = right

    def __repr__(self) -> str:
        return f"({self.left} & {self.right})"

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, AndNode)
            and self.left == other.left
            and self.right == other.right
        )

    def __hash__(self) -> int:
        return hash((self.left, self.right))

    def eval(self, assignment: "Assignment") -> bool:
        return self.left.eval(assignment) and self.right.eval(assignment)

    def num_nodes(self) -> int:
        return 1 + self.left.num_nodes() + self.right.num_nodes()

    def num_edges(self) -> int:
        return 2 + self.left.num_edges() + self.right.num_edges()


class MultiAndNode(BooleanNode):
    def __init__(self, operands: Sequence[BooleanNode]):
        assert len(operands) > 1, "MultiAndNode requires at least two operands."
        self.operands = operands

    def __repr__(self) -> str:
        return f"({' & '.join(map(str, self.operands))})"

    def __eq__(self, other) -> bool:
        return isinstance(other, MultiAndNode) and self.operands == other.operands

    def __hash__(self) -> int:
        return hash(tuple(self.operands))

    def eval(self, assignment: "Assignment") -> bool:
        return all(operand.eval(assignment) for operand in self.operands)

    def num_nodes(self) -> int:
        return 1 + sum(op.num_nodes() for op in self.operands)

    def num_edges(self) -> int:
        return len(self.operands) + sum(op.num_edges() for op in self.operands)


class OrNode(BooleanNode):
    def __init__(self, left: BooleanNode, right: BooleanNode):
        self.left = left
        self.right = right

    def __repr__(self) -> str:
        return f"({self.left} | {self.right})"

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, OrNode)
            and self.left == other.left
            and self.right == other.right
        )

    def __hash__(self) -> int:
        return hash((self.left, self.right))

    def eval(self, assignment: "Assignment") -> bool:
        return self.left.eval(assignment) or self.right.eval(assignment)

    def num_nodes(self) -> int:
        return 1 + self.left.num_nodes() + self.right.num_nodes()

    def num_edges(self) -> int:
        return 2 + self.left.num_edges() + self.right.num_edges()


class MultiOrNode(BooleanNode):
    def __init__(self, operands: Sequence[BooleanNode]):
        assert len(operands) > 1, "MultiOrNode requires at least two operands."
        self.operands = operands

    def __repr__(self) -> str:
        return f"({' | '.join(map(str, self.operands))})"

    def __eq__(self, other) -> bool:
        return isinstance(other, MultiOrNode) and self.operands == other.operands

    def __hash__(self) -> int:
        return hash(tuple(self.operands))

    def eval(self, assignment: "Assignment") -> bool:
        return any(operand.eval(assignment) for operand in self.operands)

    def num_nodes(self) -> int:
        return 1 + sum(op.num_nodes() for op in self.operands)

    def num_edges(self) -> int:
        return len(self.operands) + sum(op.num_edges() for op in self.operands)


class NotNode(BooleanNode):
    def __init__(self, operand: BooleanNode):
        self.operand = operand

    def __repr__(self) -> str:
        return f"!({self.operand})"

    def __eq__(self, other) -> bool:
        return isinstance(other, NotNode) and self.operand == other.operand

    def __hash__(self) -> int:
        return hash(self.operand)

    def eval(self, assignment: "Assignment") -> bool:
        return not self.operand.eval(assignment)

    def num_nodes(self) -> int:
        return 1 + self.operand.num_nodes()

    def num_edges(self) -> int:
        return 1 + self.operand.num_edges()


class VarNode(BooleanNode):
    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        return self.name

    def __eq__(self, other) -> bool:
        return isinstance(other, VarNode) and self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)

    def eval(self, assignment: "Assignment") -> bool:
        return self.name in assignment.true_propositions

    def num_nodes(self) -> int:
        return 1

    def num_edges(self) -> int:
        return 0


class ImplicationNode(BooleanNode):
    def __init__(self, left: BooleanNode, right: BooleanNode):
        self.left = left
        self.right = right

    def __repr__(self) -> str:
        return f"({self.left} -> {self.right})"

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, ImplicationNode)
            and self.left == other.left
            and self.right == other.right
        )

    def __hash__(self) -> int:
        return hash((self.left, self.right))

    def eval(self, assignment: "Assignment") -> bool:
        return (not self.left.eval(assignment)) or self.right.eval(assignment)

    def num_nodes(self) -> int:
        return 1 + self.left.num_nodes() + self.right.num_nodes()

    def num_edges(self) -> int:
        return 2 + self.left.num_edges() + self.right.num_edges()


class FalseNode(BooleanNode):
    def eval(self, assignment: "Assignment") -> bool:
        return False

    def __eq__(self, other) -> bool:
        return isinstance(other, FalseNode)

    def __hash__(self) -> int:
        return hash(FalseNode)

    def __repr__(self) -> str:
        return "False"

    def num_nodes(self) -> int:
        return 1

    def num_edges(self) -> int:
        return 0


class EmptyNode(BooleanNode):
    def eval(self, assignment: "Assignment") -> bool:
        return len(assignment.true_propositions) == 0

    def __eq__(self, other) -> bool:
        return isinstance(other, EmptyNode)

    def __hash__(self) -> int:
        return hash(EmptyNode)

    def __repr__(self) -> str:
        return "{" + "}"

    def num_nodes(self) -> int:
        return 1

    def num_edges(self) -> int:
        return 0


@functools.lru_cache(maxsize=500_000)
def parse(expression: str) -> BooleanNode:
    return Parser(expression).parse()
