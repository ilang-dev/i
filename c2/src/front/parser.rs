use std::fmt;

use crate::component;
use crate::ir::common::{AxisRef, Extent, Op, Pattern, Split};
use crate::ir::component::{Component, Expr, Schedule};

pub fn parse_expr(src: &str) -> Result<Expr, ParseError> {
    Parser::new(src).parse_top_level_expr()
}

pub fn parse_component(src: &str) -> Result<Component, ParseError> {
    parse_expr(src).map(component::expr)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParseError {
    pub offset: usize,
    pub message: String,
}

impl ParseError {
    fn new(offset: usize, message: impl Into<String>) -> Self {
        Self {
            offset,
            message: message.into(),
        }
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "parse error at byte {}: {}", self.offset, self.message)
    }
}

impl std::error::Error for ParseError {}

struct Parser<'a> {
    src: &'a str,
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(src: &'a str) -> Self {
        Self { src, pos: 0 }
    }

    fn parse_top_level_expr(mut self) -> Result<Expr, ParseError> {
        self.skip_ws();
        if self.is_eof() {
            return Err(self.error("expected i-expression"));
        }

        let expr = self.parse_inner_expr()?;
        self.skip_ws();
        if !self.is_eof() {
            return Err(self.error("unexpected trailing input"));
        }

        Ok(expr)
    }

    fn parse_inner_expr(&mut self) -> Result<Expr, ParseError> {
        let (op, inputs) = if let Some(op) = self.parse_op_token() {
            let input = self.parse_pattern(false)?;
            (op, vec![input])
        } else {
            let first = self.parse_pattern(false)?;
            self.skip_ws();
            if self.peek_byte() == Some(b'~') {
                (Op::Add, vec![first])
            } else {
                let op = self
                    .parse_op_token()
                    .ok_or_else(|| self.error("expected operator after input pattern"))?;
                let second = self.parse_pattern(false)?;
                (op, vec![first, second])
            }
        };

        self.skip_ws();
        self.expect_byte(b'~', "expected `~` between inputs and output")?;
        let output = self.parse_pattern(true)?;
        let schedule = if self.consume_byte_if(b'|') {
            self.parse_schedule()?
        } else {
            Schedule::default()
        };

        Ok(Expr {
            id: 0,
            op,
            inputs,
            output,
            schedule,
        })
    }

    fn parse_schedule(&mut self) -> Result<Schedule, ParseError> {
        let splits = if self.peek_byte() == Some(b'|') {
            Vec::new()
        } else {
            self.parse_splits()?
        };

        self.expect_byte(b'|', "expected second `|` in schedule")?;
        let (order, compute_at) = self.parse_order()?;

        Ok(Schedule {
            splits,
            order,
            compute_at,
        })
    }

    fn parse_splits(&mut self) -> Result<Vec<Split>, ParseError> {
        let mut splits = Vec::new();

        loop {
            let axis = self.parse_axis()?;
            self.expect_byte(b':', "expected `:` after split axis")?;

            let mut factors = Vec::new();
            factors.push(Extent::Known(self.parse_usize()?));
            while self.consume_byte_if(b':') {
                factors.push(Extent::Known(self.parse_usize()?));
            }

            splits.push(Split { axis, factors });

            if self.consume_byte_if(b',') {
                continue;
            }
            break;
        }

        Ok(splits)
    }

    fn parse_order(&mut self) -> Result<(Vec<AxisRef>, Vec<Option<AxisRef>>), ParseError> {
        let mut order = Vec::new();
        let mut compute_at = Vec::new();

        loop {
            self.skip_ws();
            if self.is_eof() {
                break;
            }

            let axis = self.parse_axis()?;
            let mut part = 0usize;
            while self.consume_byte_if(b'\'') {
                part += 1;
            }

            let axis_ref = AxisRef { axis, part };
            order.push(axis_ref.clone());

            while let Some(byte) = self.peek_byte() {
                if !byte.is_ascii_digit() {
                    break;
                }

                let index = usize::from(byte - b'0');
                self.pos += 1;

                if compute_at.len() <= index {
                    compute_at.resize(index + 1, None);
                }
                if compute_at[index].is_some() {
                    return Err(
                        self.error(format!("duplicate compute directive for input {}", index))
                    );
                }
                compute_at[index] = Some(axis_ref.clone());
            }
        }

        Ok((order, compute_at))
    }

    fn parse_pattern(&mut self, allow_empty: bool) -> Result<Pattern, ParseError> {
        self.skip_ws();

        let mut axes = Vec::new();
        while let Some(byte) = self.peek_byte() {
            if !byte.is_ascii_lowercase() {
                break;
            }
            axes.push(char::from(byte));
            self.pos += 1;
        }

        if axes.is_empty() && !allow_empty {
            return Err(self.error("expected pattern"));
        }

        Ok(Pattern(axes))
    }

    fn parse_axis(&mut self) -> Result<char, ParseError> {
        self.skip_ws();
        match self.peek_byte() {
            Some(byte) if byte.is_ascii_lowercase() => {
                self.pos += 1;
                Ok(char::from(byte))
            }
            _ => Err(self.error("expected axis")),
        }
    }

    fn parse_usize(&mut self) -> Result<usize, ParseError> {
        self.skip_ws();
        let start = self.pos;
        while let Some(byte) = self.peek_byte() {
            if !byte.is_ascii_digit() {
                break;
            }
            self.pos += 1;
        }

        if self.pos == start {
            return Err(self.error("expected integer"));
        }

        self.src[start..self.pos]
            .parse()
            .map_err(|_| ParseError::new(start, "invalid integer"))
    }

    fn parse_op_token(&mut self) -> Option<Op> {
        self.skip_ws();

        for (token, op) in [
            (">>", Op::Gt),
            (">=", Op::Ge),
            ("<<", Op::Lt),
            ("<=", Op::Le),
            ("==", Op::Eq),
            ("!=", Op::Ne),
            ("&&", Op::And),
            ("||", Op::Or),
            ("^^", Op::Xor),
            ("!!", Op::Not),
            ("+", Op::Add),
            ("*", Op::Mul),
            ("-", Op::Sub),
            ("/", Op::Div),
            (">", Op::Max),
            ("<", Op::Min),
            ("^", Op::Pow),
            ("$", Op::Log),
        ] {
            if self.src[self.pos..].starts_with(token) {
                self.pos += token.len();
                return Some(op);
            }
        }

        None
    }

    fn skip_ws(&mut self) {
        while let Some(byte) = self.peek_byte() {
            if !byte.is_ascii_whitespace() {
                break;
            }
            self.pos += 1;
        }
    }

    fn is_eof(&self) -> bool {
        self.peek_byte().is_none()
    }

    fn peek_byte(&self) -> Option<u8> {
        self.src.as_bytes().get(self.pos).copied()
    }

    fn consume_byte_if(&mut self, expected: u8) -> bool {
        self.skip_ws();
        if self.peek_byte() == Some(expected) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn expect_byte(&mut self, expected: u8, message: &'static str) -> Result<(), ParseError> {
        self.skip_ws();
        if self.peek_byte() == Some(expected) {
            self.pos += 1;
            Ok(())
        } else {
            Err(self.error(message))
        }
    }

    fn error(&self, message: impl Into<String>) -> ParseError {
        ParseError::new(self.pos, message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::common::{AxisRef, Extent, Pattern, Split};
    use crate::ir::component::{Component, Expr, Schedule};

    fn parse(src: &str) -> Expr {
        parse_expr(src).unwrap()
    }

    #[test]
    fn parse_component_lifts_expr() {
        let component = parse_component("+ij~ij").unwrap();
        assert!(matches!(component, Component::Expr(_)));
    }

    #[test]
    fn parses_binary_ops() {
        let cases = [
            ("i+i~i", Op::Add),
            ("i*i~i", Op::Mul),
            ("i-i~i", Op::Sub),
            ("i/i~i", Op::Div),
            ("i>i~i", Op::Max),
            ("i<i~i", Op::Min),
            ("i^i~i", Op::Pow),
            ("i$i~i", Op::Log),
            ("i>>i~i", Op::Gt),
            ("i>=i~i", Op::Ge),
            ("i<<i~i", Op::Lt),
            ("i<=i~i", Op::Le),
            ("i==i~i", Op::Eq),
            ("i!=i~i", Op::Ne),
            ("i&&i~i", Op::And),
            ("i||i~i", Op::Or),
            ("i^^i~i", Op::Xor),
        ];

        for (src, op) in cases {
            let expr = parse(src);
            assert_eq!(expr.op, op, "{src}");
            assert_eq!(expr.id, 0, "{src}");
            assert_eq!(
                expr.inputs,
                vec![Pattern(vec!['i']), Pattern(vec!['i'])],
                "{src}"
            );
            assert_eq!(expr.output, Pattern(vec!['i']), "{src}");
        }
    }

    #[test]
    fn parses_unary_ops() {
        let cases = [
            ("+ij~ij", Op::Add),
            ("*ij~ij", Op::Mul),
            ("-ij~ij", Op::Sub),
            ("/ij~ij", Op::Div),
            (">ij~ij", Op::Max),
            ("<ij~ij", Op::Min),
            ("^ij~ij", Op::Pow),
            ("$ij~ij", Op::Log),
            (">>ij~ij", Op::Gt),
            (">=ij~ij", Op::Ge),
            ("<<ij~ij", Op::Lt),
            ("<=ij~ij", Op::Le),
            ("==ij~ij", Op::Eq),
            ("!=ij~ij", Op::Ne),
            ("&&ij~ij", Op::And),
            ("||ij~ij", Op::Or),
            ("^^ij~ij", Op::Xor),
            ("!!ij~ij", Op::Not),
        ];

        for (src, op) in cases {
            let expr = parse(src);
            assert_eq!(expr.op, op, "{src}");
            assert_eq!(expr.inputs, vec![Pattern(vec!['i', 'j'])], "{src}");
            assert_eq!(expr.output, Pattern(vec!['i', 'j']), "{src}");
        }
    }

    #[test]
    fn parses_reduction_and_scalar_output() {
        let expr = parse("+ijk~ij");
        assert_eq!(expr.op, Op::Add);
        assert_eq!(expr.inputs, vec![Pattern(vec!['i', 'j', 'k'])]);
        assert_eq!(expr.output, Pattern(vec!['i', 'j']));

        let scalar = parse("+ij~");
        assert_eq!(scalar.op, Op::Add);
        assert_eq!(scalar.inputs, vec![Pattern(vec!['i', 'j'])]);
        assert_eq!(scalar.output, Pattern(vec![]));
    }

    #[test]
    fn parses_schedule_with_splits_order_and_compute_directives() {
        let expr = parse("ik*kj~ijk|i:2:4,k:8|ik0i'k'1j");
        assert_eq!(
            expr.schedule,
            Schedule {
                splits: vec![
                    Split {
                        axis: 'i',
                        factors: vec![Extent::Known(2), Extent::Known(4)],
                    },
                    Split {
                        axis: 'k',
                        factors: vec![Extent::Known(8)],
                    },
                ],
                order: vec![
                    AxisRef { axis: 'i', part: 0 },
                    AxisRef { axis: 'k', part: 0 },
                    AxisRef { axis: 'i', part: 1 },
                    AxisRef { axis: 'k', part: 1 },
                    AxisRef { axis: 'j', part: 0 },
                ],
                compute_at: vec![
                    Some(AxisRef { axis: 'k', part: 0 }),
                    Some(AxisRef { axis: 'k', part: 1 }),
                ],
            }
        );
    }

    #[test]
    fn parses_empty_split_section() {
        let expr = parse("+ijk~ij||ijkk'0");
        assert!(expr.schedule.splits.is_empty());
        assert_eq!(
            expr.schedule.order,
            vec![
                AxisRef { axis: 'i', part: 0 },
                AxisRef { axis: 'j', part: 0 },
                AxisRef { axis: 'k', part: 0 },
                AxisRef { axis: 'k', part: 1 },
            ]
        );
        assert_eq!(
            expr.schedule.compute_at,
            vec![Some(AxisRef { axis: 'k', part: 1 })]
        );
    }

    #[test]
    fn parses_identity_expression_as_unary_add() {
        let expr = parse("ij~ij");
        assert_eq!(expr.op, Op::Add);
        assert_eq!(expr.inputs, vec![Pattern(vec!['i', 'j'])]);
        assert_eq!(expr.output, Pattern(vec!['i', 'j']));
    }

    #[test]
    fn rejects_missing_rhs_pattern_in_binary_form() {
        let err = parse_component("ij+~ij").unwrap_err();
        assert!(err.message.contains("expected pattern"));
    }

    #[test]
    fn rejects_duplicate_compute_directive_for_same_input() {
        let err = parse_component("+ij~i||i0j0").unwrap_err();
        assert!(err.message.contains("duplicate compute directive"));
    }

    #[test]
    fn rejects_malformed_split_entry() {
        let err = parse_component("+ij~i|i|ij").unwrap_err();
        assert!(err.message.contains("expected `:`"));
    }

    #[test]
    fn rejects_trailing_garbage() {
        let err = parse_component("+ij~i ???").unwrap_err();
        assert!(err.message.contains("unexpected trailing input"));
    }
}
