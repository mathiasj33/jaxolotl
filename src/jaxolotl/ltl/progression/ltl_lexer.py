from dataclasses import dataclass
from enum import Enum


class LTLTokenType(Enum):
    NOT = "!"
    AND = "&"
    OR = "|"
    IMPLIES = "=>"
    LPAREN = "("
    RPAREN = ")"
    VAR = "VAR"
    EVENTUALLY = "F"
    ALWAYS = "G"
    UNTIL = "U"
    FALSE = "false"
    TRUE = "true"


@dataclass(eq=True)
class LTLToken:
    type: LTLTokenType
    value: str

    def __repr__(self):
        return f"Token({self.type}, {repr(self.value)})"


class LTLLexer:
    def __init__(self, formula: str):
        self.formula = formula
        self.pos = 0
        self.length = len(formula)

    def lex(self) -> list[LTLToken]:
        tokens = []
        while self.pos < self.length:
            current_char = self.formula[self.pos]

            if current_char.isspace():
                self.pos += 1
                continue

            if current_char in ["F", "G", "U"]:
                tokens.append(
                    LTLToken(
                        LTLTokenType.EVENTUALLY
                        if current_char == "F"
                        else LTLTokenType.ALWAYS
                        if current_char == "G"
                        else LTLTokenType.UNTIL,
                        current_char,
                    )
                )
                self.pos += 1
            elif current_char in ["0", "1"]:
                tokens.append(
                    LTLToken(
                        LTLTokenType.FALSE
                        if current_char == "0"
                        else LTLTokenType.TRUE,
                        current_char,
                    )
                )
                self.pos += 1
            elif current_char.isalpha() or current_char == "_":
                tokens.append(self.tokenize_variable())
            elif current_char == "!":
                tokens.append(LTLToken(LTLTokenType.NOT, "!"))
                self.pos += 1
            elif current_char == "&":
                tokens.append(LTLToken(LTLTokenType.AND, "&"))
                self.pos += 1
            elif current_char == "|":
                tokens.append(LTLToken(LTLTokenType.OR, "|"))
                self.pos += 1
            elif current_char == "=":
                if self.peek() == ">":
                    tokens.append(LTLToken(LTLTokenType.IMPLIES, "=>"))
                    self.pos += 2
                else:
                    raise SyntaxError(f"Unexpected token: {current_char}")
            elif current_char in "()":
                tokens.append(
                    LTLToken(
                        LTLTokenType.LPAREN
                        if current_char == "("
                        else LTLTokenType.RPAREN,
                        current_char,
                    )
                )
                self.pos += 1
            else:
                raise SyntaxError(f"Unexpected token: {current_char}")

        return tokens

    def tokenize_variable(self) -> LTLToken:
        start_pos = self.pos
        while self.pos < self.length and (
            self.formula[self.pos].isalnum() or self.formula[self.pos] == "_"
        ):
            self.pos += 1
        value = self.formula[start_pos : self.pos]
        type_ = (
            LTLTokenType.TRUE
            if value == "true"
            else LTLTokenType.FALSE
            if value == "false"
            else LTLTokenType.VAR
        )
        return LTLToken(type_, value)

    def peek(self) -> str | None:
        if self.pos + 1 < self.length:
            return self.formula[self.pos + 1]
        return None


# Usage example
if __name__ == "__main__":
    lexer = LTLLexer("F a & G (green | !c => d) U Frau")
    tokens = lexer.lex()
    for token in tokens:
        print(token)
