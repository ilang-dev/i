use std::fmt;

#[derive(Debug, Clone, PartialEq)]
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

pub struct Tokenizer<'a> {
    input: &'a str,
    pos: usize,
    pub peek: [Token; 2],
}

impl<'a> Tokenizer<'a> {
    pub fn new(input: &'a str) -> Result<Self, String> {
        let mut t = Tokenizer {
            input,
            pos: 0,
            peek: [Token::EOF, Token::EOF],
        };
        t.peek[0] = t.tokenize()?;
        t.peek[1] = t.tokenize()?;
        Ok(t)
    }

    pub fn next(&mut self) -> Token {
        let out = std::mem::replace(&mut self.peek[0], Token::EOF);
        self.peek[0] = std::mem::replace(&mut self.peek[1], Token::EOF);
        self.peek[1] = self.tokenize().unwrap();
        out
    }

    fn tokenize(&mut self) -> Result<Token, String> {
        self.consume_while(|c| c.is_whitespace());

        let Some(c) = self.peek_char() else {
            return Ok(Token::EOF);
        };

        if c.is_ascii_digit() {
            return Ok(Token::Int(self.consume_while_str(|c| c.is_ascii_digit())));
        }

        if c.is_ascii_alphabetic() || c == '_' || c == '(' || c == ')' {
            return Ok(Token::Symbol(self.consume_while_str(|c| {
                c.is_ascii_alphanumeric() || c == '_' || c == '\'' || c == '(' || c == ')'
            })));
        }

        self.consume_char();
        Ok(match c {
            ':' => Token::Colon,
            ',' => Token::Comma,
            '.' => Token::Dot,
            '~' => Token::Squiggle,
            '|' => Token::Bar,
            '+' | '*' | '-' | '/' | '>' | '<' | '^' | '$' | '@' | '#' | '!' => Token::Operator(c),
            _ => return Err(format!("Unexpected character: {}", c)),
        })
    }

    fn peek_char(&self) -> Option<char> {
        self.input[self.pos..].chars().next()
    }

    fn consume_char(&mut self) {
        if let Some(c) = self.peek_char() {
            self.pos += c.len_utf8();
        }
    }

    fn consume_while<F: FnMut(char) -> bool>(&mut self, mut pred: F) {
        while let Some(c) = self.peek_char() {
            if !pred(c) {
                break;
            }
            self.consume_char();
        }
    }

    fn consume_while_str<F: FnMut(char) -> bool>(&mut self, pred: F) -> String {
        let start = self.pos;
        self.consume_while(pred);
        self.input[start..self.pos].to_string()
    }
}
