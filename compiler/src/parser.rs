use std::collections::HashMap;
use std::fmt;

use crate::ast::{BinaryOp, Expr, ExprBank, ExprRef, NoOp, ScalarOp, Schedule, Symbol, UnaryOp};
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
    pub symbol_table: HashMap<Symbol, ExprRef>,
}

impl<'a> Parser<'a> {
    pub fn new(input: &'a str) -> Result<Self, String> {
        Ok(Self {
            tokenizer: Tokenizer::new(input)?,
            symbol_table: HashMap::new(),
        })
    }

    pub fn parse(&mut self) -> Result<ExprBank, ParseError> {
        let mut expr_bank = ExprBank(Vec::new());
        let expr = self.parse_expr()?;
        expr_bank.0.push(expr);
        Ok(expr_bank)
    }

    fn parse_expr(&mut self) -> Result<Expr, ParseError> {
        let expr = self.parse_unscheduled_expr()?;
        match self.tokenizer.peek[0] {
            Token::Bar => {
                let splits = self.parse_splits()?;
                let (loop_order, compute_levels) = self.parse_loop_order()?;
                Ok(Expr {
                    op: expr.op,
                    out: expr.out,
                    schedule: Schedule {
                        splits: splits,
                        loop_order: loop_order,
                        compute_levels: compute_levels,
                    },
                })
            }
            _ => Ok(expr),
        }
    }

    fn parse_unscheduled_expr(&mut self) -> Result<Expr, ParseError> {
        let scalarop = self.parse_scalarop()?;
        match self.tokenizer.next() {
            Token::Squiggle => Ok(Expr {
                op: scalarop,
                out: self.parse_symbol()?,
                schedule: Schedule {
                    splits: HashMap::new(),
                    loop_order: vec![],
                    compute_levels: vec![],
                },
            }),
            _ => Err(ParseError::InvalidToken {
                expected: "Squiggle".to_string(),
            }),
        }
    }

    fn parse_splits(&mut self) -> Result<HashMap<char, Vec<usize>>, ParseError> {
        // Skip the initial Bar token
        self.tokenizer.next();

        let mut splits = HashMap::new();

        loop {
            // Parse the index identifier
            match self.tokenizer.peek[0] {
                Token::Symbol(_) => {
                    // consume the Symbol
                    let Token::Symbol(s) = self.tokenizer.next() else {
                        unreachable!()
                    };
                    let c = s
                        .chars()
                        .next()
                        .expect("Expected single-char index in split list.");
                    let mut split_factors = Vec::new();

                    // Keep parsing colon-separated integers
                    loop {
                        match self.tokenizer.peek[0] {
                            Token::Colon => {
                                self.tokenizer.next(); // consume the colon
                                match self.tokenizer.next() {
                                    Token::Int(num) => {
                                        split_factors.push(num.parse::<usize>().unwrap());
                                    }
                                    _ => {
                                        return Err(ParseError::InvalidToken {
                                            expected: "Integer".to_string(),
                                        })
                                    }
                                }
                            }
                            Token::EOF | Token::Squiggle => {
                                splits.insert(c, split_factors);
                                return Ok(splits);
                            }
                            Token::Symbol(_) | Token::Operator(_) | Token::Dot | Token::Bar => {
                                splits.insert(c, split_factors);
                                return Ok(splits);
                            }
                            _ => {
                                // Check if there's a comma indicating another split-list
                                if let Token::Comma = self.tokenizer.next() {
                                    splits.insert(c, split_factors);
                                    break; // Continue to parse the next split-list
                                } else {
                                    return Err(ParseError::InvalidToken {
                                        expected: "Comma or end of schedule".to_string(),
                                    });
                                }
                            }
                        }
                    }
                }
                Token::Bar => return Ok(splits), // empty splits list
                _ => {
                    return Err(ParseError::InvalidToken {
                        expected: "Symbol".to_string(),
                    })
                }
            }
        }
    }

    fn parse_loop_order(&mut self) -> Result<(Vec<(char, usize)>, Vec<usize>), ParseError> {
        // Skip the initial Bar token
        self.tokenizer.next();
        match self.tokenizer.next() {
            Token::Symbol(s) => {
                let mut loop_order = Vec::new();
                let mut compute_levels = Vec::new();
                let mut chars = s.chars().peekable();
                let mut compute_level = 0;
                while let Some(c) = chars.next() {
                    if c.is_alphabetic() {
                        let mut apostrophe_count = 0;

                        while let Some(&next) = chars.peek() {
                            if next == '\'' {
                                chars.next();
                                apostrophe_count += 1;
                            } else {
                                break;
                            }
                        }

                        loop_order.push((c, apostrophe_count));
                    }

                    if let Some(&'(') = chars.peek() {
                        chars.next(); // Consume '('
                        while let Some(inner) = chars.peek() {
                            if *inner == ')' {
                                chars.next(); // Consume ')'
                                break;
                            } else if *inner == ',' {
                                chars.next(); // Consume ')'
                            } else if inner.is_ascii_digit() {
                                let ind: usize =
                                    chars.next().unwrap().to_digit(10).unwrap() as usize;
                                if compute_levels.len() <= ind {
                                    compute_levels.resize(ind + 1, 0);
                                }
                                compute_levels[ind] = compute_level + 1;
                            } else {
                                return Err(ParseError::InvalidToken {
                                    expected: "Digit inside computation level parentheses"
                                        .to_string(),
                                });
                            }
                        }
                    }
                    compute_level += 1;
                }
                Ok((loop_order, compute_levels))
            }
            _ => Err(ParseError::InvalidToken {
                expected: "Comma or end of schedule".to_string(),
            }),
        }
    }

    fn parse_scalarop(&mut self) -> Result<ScalarOp, ParseError> {
        match self.tokenizer.peek {
            [Token::Operator(_), _] => Ok(ScalarOp::UnaryOp(self.parse_unaryop()?)),
            [Token::Symbol(_), Token::Operator(_)] => {
                Ok(ScalarOp::BinaryOp(self.parse_binaryop()?))
            }
            [Token::Symbol(_), Token::Squiggle] => Ok(ScalarOp::NoOp(self.parse_noop()?)),
            _ => Err(ParseError::InvalidToken {
                expected: "[Operator]<Any>, [Symbol][Operator], [Symbol]<Any>".to_string(),
            }),
        }
    }

    fn parse_binaryop(&mut self) -> Result<BinaryOp, ParseError> {
        let left = self.parse_symbol()?;
        match self.tokenizer.next() {
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
        match self.tokenizer.next() {
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
        match self.tokenizer.next() {
            Token::Symbol(s) => Ok(Symbol(s)),
            _ => Err(ParseError::InvalidToken {
                expected: "Symbol".to_string(),
            }),
        }
    }
}
