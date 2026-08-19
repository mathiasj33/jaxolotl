"""Tests for the formula tokenizer."""

from jaxolotl.alg.deep_ltl.reach_avoid.reach_avoid_sequence import EPSILON
from jaxolotl.alg.struct_ltl.reach_avoid.formula_tokenizer import (
    AND_TOKEN,
    EPSILON_TOKEN,
    FALSE_TOKEN,
    NOT_TOKEN,
    OR_TOKEN,
    TRUE_TOKEN,
    Vocabulary,
    encode_tokens,
    formula_to_tokens,
    tokenize_reach_avoid_step,
)
from jaxolotl.ltl.logic.boolean_parser import (
    AndNode,
    FalseNode,
    MultiAndNode,
    MultiOrNode,
    NotNode,
    OrNode,
    VarNode,
)


class TestFormulaToTokens:
    """Tests for the formula_to_tokens function."""

    def test_variable_node(self):
        """Test tokenizing a simple variable."""
        formula = VarNode("a")
        assert formula_to_tokens(formula) == ["a"]

    def test_and_node(self):
        """Test tokenizing a binary AND."""
        formula = AndNode(VarNode("a"), VarNode("b"))
        assert formula_to_tokens(formula) == ["a", AND_TOKEN, "b"]

    def test_or_node(self):
        """Test tokenizing a binary OR."""
        formula = OrNode(VarNode("a"), VarNode("b"))
        assert formula_to_tokens(formula) == ["a", OR_TOKEN, "b"]

    def test_not_node(self):
        """Test tokenizing a NOT."""
        formula = NotNode(VarNode("a"))
        assert formula_to_tokens(formula) == [NOT_TOKEN, "a"]

    def test_multi_and_node(self):
        """Test tokenizing a multi-operand AND."""
        formula = MultiAndNode([VarNode("a"), VarNode("b"), VarNode("c")])
        assert formula_to_tokens(formula) == ["a", AND_TOKEN, "b", AND_TOKEN, "c"]

    def test_multi_or_node(self):
        """Test tokenizing a multi-operand OR."""
        formula = MultiOrNode([VarNode("a"), VarNode("b"), VarNode("c")])
        assert formula_to_tokens(formula) == ["a", OR_TOKEN, "b", OR_TOKEN, "c"]

    def test_nested_formula(self):
        """Test tokenizing a nested formula: (a & !b) | c."""
        inner = AndNode(VarNode("a"), NotNode(VarNode("b")))
        formula = OrNode(inner, VarNode("c"))
        assert formula_to_tokens(formula) == [
            "a",
            AND_TOKEN,
            NOT_TOKEN,
            "b",
            OR_TOKEN,
            "c",
        ]

    def test_none_formula(self):
        """Test tokenizing None (represents True)."""
        assert formula_to_tokens(None) == [TRUE_TOKEN]

    def test_epsilon_formula(self):
        """Test tokenizing EPSILON."""
        assert formula_to_tokens(EPSILON) == [EPSILON_TOKEN]

    def test_false_node(self):
        """Test tokenizing False."""
        assert formula_to_tokens(FalseNode()) == [FALSE_TOKEN]


class TestVocabulary:
    """Tests for the Vocabulary class."""

    def test_from_propositions(self):
        """Test creating vocabulary from propositions."""
        vocab = Vocabulary.from_propositions(["a", "b", "c"])

        # Check special tokens are present
        assert vocab.epsilon_idx == 0
        assert vocab.and_idx > 0
        assert vocab.or_idx > 0

        # Check propositions are present
        assert "a" in vocab.token_to_idx
        assert "b" in vocab.token_to_idx
        assert "c" in vocab.token_to_idx

    def test_vocabulary_length(self):
        """Test vocabulary length includes special tokens and propositions."""
        vocab = Vocabulary.from_propositions(["a", "b"])
        # 10 special tokens + 2 propositions
        assert len(vocab) == 9  # noqa


class TestTokenizeReachAvoidStep:
    """Tests for the tokenize_reach_avoid_step function."""

    def test_simple_reach_avoid(self):
        """Test tokenizing a simple reach-avoid step."""
        reach = VarNode("a")
        avoid = VarNode("b")
        reach_tokens, avoid_tokens = tokenize_reach_avoid_step(reach, avoid)
        assert reach_tokens == ["a"]
        assert avoid_tokens == ["b"]

    def test_reach_with_none_avoid(self):
        """Test tokenizing reach with no avoid (None -> True)."""
        reach = VarNode("a")
        avoid = None
        reach_tokens, avoid_tokens = tokenize_reach_avoid_step(reach, avoid)
        assert reach_tokens == ["a"]
        assert avoid_tokens == [TRUE_TOKEN]


class TestEncodeTokens:
    """Tests for the encode_tokens function."""

    def test_encode_simple_tokens(self):
        """Test encoding tokens to indices."""
        vocab = Vocabulary.from_propositions(["a", "b", "c"])
        tokens = ["a", AND_TOKEN, "b"]
        encoded = encode_tokens(tokens, vocab)

        assert len(encoded) == 3
        assert encoded[0] == vocab.token_to_idx["a"]
        assert encoded[1] == vocab.and_idx
        assert encoded[2] == vocab.token_to_idx["b"]
