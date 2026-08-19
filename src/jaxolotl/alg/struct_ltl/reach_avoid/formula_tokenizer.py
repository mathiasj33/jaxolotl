"""Tokenizer for Boolean formulas.

This module provides functionality to convert Boolean formulas (BooleanNode trees)
into sequences of tokens suitable for processing by sequence models like GRUs.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from jaxolotl.alg.deep_ltl.reach_avoid.reach_avoid_sequence import EpsilonType
from jaxolotl.ltl.logic.boolean_parser import (
    AndNode,
    BooleanNode,
    EmptyNode,
    FalseNode,
    ImplicationNode,
    MultiAndNode,
    MultiOrNode,
    NotNode,
    OrNode,
    VarNode,
)

# Special tokens
EPSILON_TOKEN = "<EPS>"
AND_TOKEN = "&"
OR_TOKEN = "|"
NOT_TOKEN = "!"
TRUE_TOKEN = "True"
FALSE_TOKEN = "False"
EMPTY_TOKEN = "{}"


@dataclass(frozen=True)
class Vocabulary:
    """A vocabulary mapping tokens to indices and vice versa."""

    token_to_idx: dict[str, int]
    idx_to_token: dict[int, str]

    @classmethod
    def from_propositions(cls, propositions: Sequence[str]) -> "Vocabulary":
        """Create a vocabulary from a list of proposition names.

        The vocabulary includes:
        - Special tokens (EPSILON, AND, OR, NOT, TRUE, FALSE, EMPTY)
        - All proposition names
        """
        special_tokens = [
            EPSILON_TOKEN,
            AND_TOKEN,
            OR_TOKEN,
            NOT_TOKEN,
            TRUE_TOKEN,
            FALSE_TOKEN,
            EMPTY_TOKEN,
        ]
        all_tokens = special_tokens + list(propositions)
        token_to_idx = {token: idx for idx, token in enumerate(all_tokens)}
        idx_to_token = dict(enumerate(all_tokens))
        return cls(token_to_idx, idx_to_token)

    def __len__(self) -> int:
        return len(self.token_to_idx)

    @property
    def epsilon_idx(self) -> int:
        return self.token_to_idx[EPSILON_TOKEN]

    @property
    def and_idx(self) -> int:
        return self.token_to_idx[AND_TOKEN]

    @property
    def or_idx(self) -> int:
        return self.token_to_idx[OR_TOKEN]

    @property
    def not_idx(self) -> int:
        return self.token_to_idx[NOT_TOKEN]

    @property
    def true_idx(self) -> int:
        return self.token_to_idx[TRUE_TOKEN]

    @property
    def false_idx(self) -> int:
        return self.token_to_idx[FALSE_TOKEN]


def formula_to_tokens(
    formula: BooleanNode | EpsilonType | None,
) -> list[str]:
    """Convert a Boolean formula to a list of tokens.

    This produces a minimal token representation without superfluous parentheses
    by leveraging operator precedence: NOT > AND > OR.

    Args:
        formula: A BooleanNode tree, EPSILON, or None.

    Returns:
        A list of string tokens representing the formula.
    """
    if formula is None:
        return [TRUE_TOKEN]
    if isinstance(formula, EpsilonType):
        return [EPSILON_TOKEN]

    return _tokenize_node(formula)


def _tokenize_node(node: BooleanNode) -> list[str]:
    """Recursively tokenize a Boolean node.

    Args:
        node: The node to tokenize.

    Returns:
        List of tokens representing this subtree.
    """
    # Handle atom nodes
    if isinstance(node, VarNode):
        return [node.name]
    if isinstance(node, FalseNode):
        return [FALSE_TOKEN]
    if isinstance(node, EmptyNode):
        return [EMPTY_TOKEN]
    # Handle compound nodes
    return _tokenize_compound_node(node)


def _tokenize_compound_node(node: BooleanNode) -> list[str]:
    """Tokenize compound (non-atom) Boolean nodes."""
    # Unary operators
    if isinstance(node, NotNode):
        return [NOT_TOKEN] + _tokenize_node(node.operand)

    # Binary/n-ary AND
    if isinstance(node, AndNode):
        return _tokenize_node(node.left) + [AND_TOKEN] + _tokenize_node(node.right)
    if isinstance(node, MultiAndNode):
        return _tokenize_multi_op(node.operands, AND_TOKEN)

    # Binary/n-ary OR
    if isinstance(node, OrNode):
        return _tokenize_node(node.left) + [OR_TOKEN] + _tokenize_node(node.right)
    if isinstance(node, MultiOrNode):
        return _tokenize_multi_op(node.operands, OR_TOKEN)

    # Implication: a -> b ≡ !a | b
    if isinstance(node, ImplicationNode):
        return (
            [NOT_TOKEN]
            + _tokenize_node(node.left)
            + [OR_TOKEN]
            + _tokenize_node(node.right)
        )

    raise ValueError(f"Unknown node type: {type(node)}")


def _tokenize_multi_op(operands: Sequence[BooleanNode], operator: str) -> list[str]:
    """Tokenize a multi-operand node (MultiAnd or MultiOr).

    Args:
        operands: The operands of the multi-operand node.
        operator: The operator token to use between operands.

    Returns:
        List of tokens.
    """
    tokens: list[str] = []
    for i, operand in enumerate(operands):
        if i > 0:
            tokens.append(operator)
        tokens.extend(_tokenize_node(operand))
    return tokens


def tokenize_reach_avoid_step(
    reach: BooleanNode | EpsilonType | None,
    avoid: BooleanNode | None,
) -> tuple[list[str], list[str]]:
    """Tokenize a single reach-avoid step into separate reach and avoid token lists.

    Args:
        reach: The reach formula.
        avoid: The avoid formula.

    Returns:
        Tuple of (reach_tokens, avoid_tokens).
    """
    reach_tokens = formula_to_tokens(reach)
    avoid_tokens = formula_to_tokens(avoid)
    return reach_tokens, avoid_tokens


def encode_tokens(tokens: list[str], vocab: Vocabulary) -> list[int]:
    """Encode a list of tokens to indices using the vocabulary.

    Args:
        tokens: List of string tokens.
        vocab: The vocabulary to use for encoding.

    Returns:
        List of token indices.
    """
    return [vocab.token_to_idx[token] for token in tokens]


def pad_sequence(
    sequence: list[int],
    max_length: int,
    pad_value: int = -1,
) -> list[int]:
    """Pad a sequence to a fixed length.

    Args:
        sequence: The sequence to pad.
        max_length: The target length.
        pad_value: The value to use for padding.

    Returns:
        Padded sequence of length max_length.
    """
    if len(sequence) > max_length:
        return sequence[:max_length]
    return sequence + [pad_value] * (max_length - len(sequence))
