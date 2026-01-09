use std::collections::HashMap;
use std::fmt;

use crate::ast::{BinaryOp, Expr, ExprBank, NoOp, ScalarOp, Schedule, Symbol, UnaryOp};
use crate::tokenizer::{Token, Tokenizer};

#[derive(Debug)]
pub enum ParseError {
    InvalidToken { expected: String },
    UnrecognizedSymbol { symbol: Symbol },
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseError::InvalidToken { expected } => {
                write!(f, "Invalid token: Expected {expected}.")
            }
            ParseError::UnrecognizedSymbol { symbol } => {
                write!(f, "Unrecognized Symbol: {}.", symbol.0)
            }
        }
    }
}

impl std::error::Error for ParseError {}

pub struct Parser<'a> {
    tokenizer: Tokenizer<'a>,
}

impl<'a> Parser<'a> {
    pub fn new(input: &'a str) -> Result<Self, String> {
        Ok(Self {
            tokenizer: Tokenizer::new(input)?,
        })
    }

    pub fn parse(&mut self) -> Result<ExprBank, ParseError> {
        let mut expr_bank = ExprBank(Vec::new());
        expr_bank.0.push(self.parse_expr()?);
        Ok(expr_bank)
    }

    fn parse_expr(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_unscheduled_expr()?;
        if matches!(self.peek0(), Token::Bar) {
            let schedule = self.parse_schedule()?;
            expr.schedule = schedule;
        }
        Ok(expr)
    }

    fn parse_unscheduled_expr(&mut self) -> Result<Expr, ParseError> {
        let op = self.parse_scalarop()?;
        self.eat(Token::Squiggle, "Squiggle")?;
        let out = self.parse_symbol()?;
        Ok(Expr {
            op,
            out,
            schedule: empty_schedule(),
        })
    }

    fn parse_schedule(&mut self) -> Result<Schedule, ParseError> {
        self.eat(Token::Bar, "Bar")?;
        let splits = self.parse_splits_body()?;
        self.eat(Token::Bar, "Bar")?;
        let (loop_order, compute_levels) = self.parse_loop_order_body()?;
        Ok(Schedule {
            splits,
            loop_order,
            compute_levels,
        })
    }

    fn parse_splits_body(&mut self) -> Result<HashMap<char, Vec<usize>>, ParseError> {
        let mut splits = HashMap::new();

        while let Token::Symbol(_) = self.peek0() {
            let Token::Symbol(s) = self.next() else {
                unreachable!()
            };
            let c = s.chars().next().ok_or(ParseError::InvalidToken {
                expected: "Non-empty Symbol".to_string(),
            })?;

            let mut factors = Vec::new();
            while matches!(self.peek0(), Token::Colon) {
                self.next();
                let Token::Int(n) = self.next() else {
                    return Err(ParseError::InvalidToken {
                        expected: "Integer".to_string(),
                    });
                };
                factors.push(n.parse::<usize>().unwrap());
            }

            splits.insert(c, factors);

            match self.peek0() {
                Token::Comma => {
                    self.next();
                    continue;
                }
                Token::Bar => break,
                _ => {
                    return Err(ParseError::InvalidToken {
                        expected: "Comma or Bar".to_string(),
                    })
                }
            }
        }

        Ok(splits)
    }

    fn parse_loop_order_body(&mut self) -> Result<(Vec<(char, usize)>, Vec<usize>), ParseError> {
        let Token::Symbol(s) = self.next() else {
            return Err(ParseError::InvalidToken {
                expected: "Symbol".to_string(),
            });
        };

        let mut loop_order = Vec::new();
        let mut compute_levels = Vec::new();

        let mut chars = s.chars().peekable();
        let mut compute_level = 0usize;

        while let Some(c) = chars.next() {
            if c.is_alphabetic() {
                let mut apostrophes = 0usize;
                while matches!(chars.peek(), Some('\'')) {
                    chars.next();
                    apostrophes += 1;
                }
                loop_order.push((c, apostrophes));
            }

            if matches!(chars.peek(), Some('(')) {
                chars.next();
                loop {
                    match chars.peek().copied() {
                        Some(')') => {
                            chars.next();
                            break;
                        }
                        Some(',') => {
                            chars.next();
                        }
                        Some(d) if d.is_ascii_digit() => {
                            let ind = chars.next().unwrap().to_digit(10).unwrap() as usize;
                            if compute_levels.len() <= ind {
                                compute_levels.resize(ind + 1, 0);
                            }
                            compute_levels[ind] = compute_level + 1;
                        }
                        _ => {
                            return Err(ParseError::InvalidToken {
                                expected: "Digit, comma, or ')'".to_string(),
                            })
                        }
                    }
                }
            }

            compute_level += 1;
        }

        Ok((loop_order, compute_levels))
    }

    fn parse_scalarop(&mut self) -> Result<ScalarOp, ParseError> {
        match self.tokenizer.peek {
            [Token::Operator(_), _] => Ok(ScalarOp::UnaryOp(self.parse_unaryop()?)),
            [Token::Symbol(_), Token::Operator(_)] => {
                Ok(ScalarOp::BinaryOp(self.parse_binaryop()?))
            }
            [Token::Symbol(_), Token::Squiggle] => Ok(ScalarOp::NoOp(self.parse_noop()?)),
            _ => Err(ParseError::InvalidToken {
                expected: "[Operator]<Any>, [Symbol][Operator], [Symbol][Squiggle]".to_string(),
            }),
        }
    }

    fn parse_binaryop(&mut self) -> Result<BinaryOp, ParseError> {
        let left = self.parse_symbol()?;
        match self.next() {
            Token::Operator('*') => Ok(BinaryOp::Mul(left, self.parse_symbol()?)),
            Token::Operator('/') => Ok(BinaryOp::Div(left, self.parse_symbol()?)),
            Token::Operator('+') => Ok(BinaryOp::Add(left, self.parse_symbol()?)),
            Token::Operator('-') => Ok(BinaryOp::Sub(left, self.parse_symbol()?)),
            Token::Operator('>') => Ok(BinaryOp::Max(left, self.parse_symbol()?)),
            Token::Operator('<') => Ok(BinaryOp::Min(left, self.parse_symbol()?)),
            _ => Err(ParseError::InvalidToken {
                expected: "Operator".to_string(),
            }),
        }
    }

    fn parse_unaryop(&mut self) -> Result<UnaryOp, ParseError> {
        match self.next() {
            Token::Operator('*') => Ok(UnaryOp::Prod(self.parse_symbol()?)),
            Token::Operator('+') => Ok(UnaryOp::Accum(self.parse_symbol()?)),
            Token::Operator('>') => Ok(UnaryOp::Max(self.parse_symbol()?)),
            Token::Operator('<') => Ok(UnaryOp::Min(self.parse_symbol()?)),
            Token::Operator('!') => Ok(UnaryOp::Relu(self.parse_symbol()?)),
            Token::Operator('-') => Ok(UnaryOp::Neg(self.parse_symbol()?)),
            Token::Operator('/') => Ok(UnaryOp::Recip(self.parse_symbol()?)),
            Token::Operator('^') => Ok(UnaryOp::Exp(self.parse_symbol()?)),
            Token::Operator('$') => Ok(UnaryOp::Log(self.parse_symbol()?)),
            Token::Operator('@') => Ok(UnaryOp::Sqrt(self.parse_symbol()?)),
            Token::Operator('#') => Ok(UnaryOp::Abs(self.parse_symbol()?)),
            _ => Err(ParseError::InvalidToken {
                expected: "Operator".to_string(),
            }),
        }
    }

    fn parse_noop(&mut self) -> Result<NoOp, ParseError> {
        Ok(NoOp(self.parse_symbol()?))
    }

    fn parse_symbol(&mut self) -> Result<Symbol, ParseError> {
        match self.next() {
            Token::Symbol(s) => Ok(Symbol(s)),
            _ => Err(ParseError::InvalidToken {
                expected: "Symbol".to_string(),
            }),
        }
    }

    fn peek0(&self) -> Token {
        self.tokenizer.peek[0].clone()
    }

    fn next(&mut self) -> Token {
        self.tokenizer.next()
    }

    fn eat(&mut self, tok: Token, expected: &str) -> Result<(), ParseError> {
        let got = self.next();
        if std::mem::discriminant(&got) == std::mem::discriminant(&tok) && got == tok {
            Ok(())
        } else {
            Err(ParseError::InvalidToken {
                expected: expected.to_string(),
            })
        }
    }
}

fn empty_schedule() -> Schedule {
    Schedule {
        splits: HashMap::new(),
        loop_order: vec![],
        compute_levels: vec![],
    }
}
