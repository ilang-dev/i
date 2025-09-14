use std::fmt;

#[derive(Debug, PartialEq)]
pub enum Token {
    Symbol(String),
    Colon,
    Comma,
    Dot,
    Squiggle,
    Bar,
    Int(String),
    Operator(char),
    EOF,
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Token::Symbol(s) => write!(f, "[{}]", s),
            Token::Colon => write!(f, "[:]"),
            Token::Comma => write!(f, "[,]"),
            Token::Dot => write!(f, "[.]"),
            Token::Squiggle => write!(f, "[~]"),
            Token::Bar => write!(f, "[|]"),
            Token::Int(s) => write!(f, "[{}]", s),
            Token::Operator(op) => write!(f, "Operator [{}]", op),
            Token::EOF => write!(f, "[EOF]"),
        }
    }
}

/// Circular buffer used to hold the peek Tokens
struct PeekBuffer {
    tokens: [Token; 2],
    pos: usize,
}

impl PeekBuffer {
    fn popswap(&mut self, token: Token) -> Token {
        let token = std::mem::replace(&mut self.tokens[self.pos], token);
        self.pos = (self.pos + 1) % 2;
        token
    }

    fn peek(&self) -> [&Token; 2] {
        [&self.tokens[self.pos], &self.tokens[(self.pos + 1) % 2]]
    }
}

pub struct Tokenizer<'a> {
    input: &'a str,
    pos: usize,
    peek: PeekBuffer,
}

impl<'a> Tokenizer<'a> {
    pub fn new(input: &'a str) -> Result<Self, String> {
        let mut tokenizer = Tokenizer {
            input,
            pos: 0,
            peek: PeekBuffer {
                tokens: [Token::EOF, Token::EOF],
                pos: 0,
            },
        };

        // buffer the first two Tokens
        let token = tokenizer.tokenize()?;
        tokenizer.peek.popswap(token);
        let token = tokenizer.tokenize()?;
        tokenizer.peek.popswap(token);

        Ok(tokenizer)
    }

    /// An array of ref to the two upcoming Tokens in the stream
    pub fn peek(&self) -> [&Token; 2] {
        self.peek.peek()
    }

    /// Get the next Token in the stream
    pub fn next(&mut self) -> Token {
        let token = self.tokenize().unwrap();
        self.peek.popswap(token)
    }

    fn tokenize(&mut self) -> Result<Token, String> {
        self.consume_whitespace();

        if self.pos >= self.input.len() {
            return Ok(Token::EOF);
        }

        let c = self.peek_char();

        if c.is_numeric() {
            return Ok(Token::Int(self.consume_int()));
        }

        if c.is_alphabetic() || c.is_numeric() || c == '_' || c == '(' || c == ')' {
            return Ok(Token::Symbol(self.consume_str()));
        }

        match c {
            ':' => {
                self.consume_char();
                Ok(Token::Colon)
            }
            ',' => {
                self.consume_char();
                Ok(Token::Comma)
            }
            '.' => {
                self.consume_char();
                Ok(Token::Dot)
            }
            '~' => {
                self.consume_char();
                Ok(Token::Squiggle)
            }
            '|' => {
                self.consume_char();
                Ok(Token::Bar)
            }
            '+' | '*' | '-' | '/' | '>' | '<' | '^' | '$' | '@' | '#' | '!' => {
                self.consume_char();
                Ok(Token::Operator(c))
            }
            _ => Err(format!("Unexpected character: {}", c)),
        }
    }

    fn peek_char(&self) -> char {
        self.input[self.pos..].chars().next().unwrap()
    }

    fn consume_char(&mut self) {
        self.pos += self.peek_char().len_utf8();
    }

    fn consume_whitespace(&mut self) {
        while self.pos < self.input.len() && self.peek_char().is_whitespace() {
            self.consume_char();
        }
    }

    fn consume_int(&mut self) -> String {
        let start = self.pos;
        while self.pos < self.input.len() && (self.peek_char().is_numeric()) {
            self.consume_char();
        }
        self.input[start..self.pos].to_string()
    }

    fn consume_str(&mut self) -> String {
        let start = self.pos;
        while self.pos < self.input.len()
            && (self.peek_char().is_alphabetic()
                || self.peek_char().is_numeric()
                || self.peek_char() == '_'
                || self.peek_char() == '\''
                || self.peek_char() == '('
                || self.peek_char() == ')')
        {
            self.consume_char();
        }
        self.input[start..self.pos].to_string()
    }
}
