use std::fmt;

use crate::component;
use crate::ir::common::Op;
use crate::ir::component::Component;
use crate::ir::expr::{Expr, Operand, PermutationAtom};

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
        let (splits, permutation) = if self.consume_byte_if(b'|') {
            self.parse_schedule()?
        } else {
            (Vec::new(), Vec::new())
        };

        Ok(Expr {
            op,
            inputs,
            output,
            splits,
            permutation,
        })
    }

    fn parse_schedule(
        &mut self,
    ) -> Result<(Vec<(char, Vec<usize>)>, Vec<PermutationAtom>), ParseError> {
        self.skip_ws();
        let splits = if self.peek_byte() == Some(b'|') {
            Vec::new()
        } else {
            self.parse_splits()?
        };

        self.expect_byte(b'|', "expected second `|` in schedule")?;
        let permutation = self.parse_permutation()?;

        Ok((splits, permutation))
    }

    fn parse_splits(&mut self) -> Result<Vec<(char, Vec<usize>)>, ParseError> {
        let mut splits = Vec::new();

        loop {
            let axis = self.parse_axis()?;
            self.expect_byte(b':', "expected `:` after split axis")?;

            let mut factors = Vec::new();
            factors.push(self.parse_usize()?);
            while self.consume_byte_if(b':') {
                factors.push(self.parse_usize()?);
            }

            if splits.iter().any(|(existing, _)| *existing == axis) {
                return Err(self.error(format!("duplicate split entry for axis {axis}")));
            }
            splits.push((axis, factors));

            if self.consume_byte_if(b',') {
                continue;
            }
            break;
        }

        Ok(splits)
    }

    fn parse_permutation(&mut self) -> Result<Vec<PermutationAtom>, ParseError> {
        let mut permutation = Vec::new();
        let mut seen_inputs = Vec::new();
        let mut seen_bang = false;

        loop {
            self.skip_ws();
            if self.is_eof() {
                break;
            }

            if matches!(self.peek_byte(), Some(byte) if byte.is_ascii_lowercase()) {
                let axis = self.parse_axis()?;
                let mut part = 0usize;
                while self.consume_byte_if(b'\'') {
                    part += 1;
                }

                permutation.push(PermutationAtom::Axis { axis, part });
                continue;
            }

            if matches!(self.peek_byte(), Some(byte) if byte.is_ascii_digit()) {
                let input = self.parse_operand()?;
                if seen_inputs.contains(&input) {
                    return Err(self.error(format!(
                        "duplicate compute directive for input {}",
                        operand_index(input)
                    )));
                }
                seen_inputs.push(input);
                permutation.push(PermutationAtom::Input(input));
                continue;
            }

            if self.consume_byte_if(b'!') {
                if seen_bang {
                    return Err(self.error("duplicate output init directive"));
                }
                seen_bang = true;
                permutation.push(PermutationAtom::Bang);
                continue;
            }

            return Err(self.error("expected axis, `!`, or operand in permutation"));
        }

        Ok(permutation)
    }

    fn parse_operand(&mut self) -> Result<Operand, ParseError> {
        self.skip_ws();
        match self.peek_byte() {
            Some(b'0') => {
                self.pos += 1;
                Ok(Operand::Left)
            }
            Some(b'1') => {
                self.pos += 1;
                Ok(Operand::Right)
            }
            Some(byte) if byte.is_ascii_digit() => Err(self.error("expected operand `0` or `1`")),
            _ => Err(self.error("expected operand")),
        }
    }

    fn parse_pattern(&mut self, _allow_empty: bool) -> Result<Vec<char>, ParseError> {
        self.skip_ws();

        if self.peek_byte() == Some(b'.') {
            self.pos += 1;
            if matches!(self.peek_byte(), Some(byte) if byte.is_ascii_lowercase() || byte == b'.') {
                return Err(self.error("scalar pattern must be `.`"));
            }
            return Ok(Vec::new());
        }

        let mut axes = Vec::new();
        while let Some(byte) = self.peek_byte() {
            if !byte.is_ascii_lowercase() {
                break;
            }
            axes.push(char::from(byte));
            self.pos += 1;
        }

        if axes.is_empty() {
            return Err(self.error("expected pattern"));
        }

        Ok(axes)
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

fn operand_index(operand: Operand) -> usize {
    match operand {
        Operand::Left => 0,
        Operand::Right => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::component::Component;
    use crate::ir::expr::{Expr, PermutationAtom};

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
            assert_eq!(expr.inputs, vec![vec!['i'], vec!['i']], "{src}");
            assert_eq!(expr.output, vec!['i'], "{src}");
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
            assert_eq!(expr.inputs, vec![vec!['i', 'j']], "{src}");
            assert_eq!(expr.output, vec!['i', 'j'], "{src}");
        }
    }

    #[test]
    fn parses_scalar_inputs_with_single_dot() {
        let expr = parse(".*i~i");
        assert_eq!(expr.op, Op::Mul);
        assert_eq!(expr.inputs, vec![Vec::<char>::new(), vec!['i']]);
        assert_eq!(expr.output, vec!['i']);

        let expr = parse("i/.~i");
        assert_eq!(expr.op, Op::Div);
        assert_eq!(expr.inputs, vec![vec!['i'], Vec::<char>::new()]);
        assert_eq!(expr.output, vec!['i']);
    }

    #[test]
    fn parses_reduction_and_scalar_output() {
        let expr = parse("+ijk~ij");
        assert_eq!(expr.op, Op::Add);
        assert_eq!(expr.inputs, vec![vec!['i', 'j', 'k']]);
        assert_eq!(expr.output, vec!['i', 'j']);

        let scalar = parse("+ij~.");
        assert_eq!(scalar.op, Op::Add);
        assert_eq!(scalar.inputs, vec![vec!['i', 'j']]);
        assert_eq!(scalar.output, Vec::<char>::new());
    }

    #[test]
    fn rejects_mixed_scalar_input_patterns() {
        let err = parse_component(".i+i~i").unwrap_err();
        assert!(err.message.contains("scalar pattern"));

        let err = parse_component("i+.j~i").unwrap_err();
        assert!(err.message.contains("scalar pattern"));
    }

    #[test]
    fn rejects_empty_scalar_output() {
        let err = parse_component("+ij~").unwrap_err();
        assert!(err.message.contains("expected pattern"));
    }

    #[test]
    fn parses_schedule_with_splits_order_and_compute_directives() {
        let expr = parse("ik*kj~ijk|i:2:4,k:8|ik0i'k'1j");
        assert_eq!(expr.splits, vec![('i', vec![2, 4]), ('k', vec![8])]);
        assert_eq!(
            expr.permutation,
            vec![
                PermutationAtom::Axis { axis: 'i', part: 0 },
                PermutationAtom::Axis { axis: 'k', part: 0 },
                PermutationAtom::Input(Operand::Left),
                PermutationAtom::Axis { axis: 'i', part: 1 },
                PermutationAtom::Axis { axis: 'k', part: 1 },
                PermutationAtom::Input(Operand::Right),
                PermutationAtom::Axis { axis: 'j', part: 0 },
            ]
        );
    }

    #[test]
    fn parses_empty_split_section() {
        let expr = parse("+ijk~ij||ij!kk'0");
        assert!(expr.splits.is_empty());
        assert_eq!(
            expr.permutation,
            vec![
                PermutationAtom::Axis { axis: 'i', part: 0 },
                PermutationAtom::Axis { axis: 'j', part: 0 },
                PermutationAtom::Bang,
                PermutationAtom::Axis { axis: 'k', part: 0 },
                PermutationAtom::Axis { axis: 'k', part: 1 },
                PermutationAtom::Input(Operand::Left),
            ]
        );
    }

    #[test]
    fn parses_schedule_with_empty_permutation() {
        let expr = parse("+ijk~ij|i:2,k:4|");
        assert_eq!(expr.splits, vec![('i', vec![2]), ('k', vec![4])]);
        assert!(expr.permutation.is_empty());
    }

    #[test]
    fn parses_whitespace_in_expression_and_schedule() {
        let expr = parse("  ij + i ~ ij |  | i j 1 ");
        assert_eq!(expr.op, Op::Add);
        assert_eq!(expr.inputs, vec![vec!['i', 'j'], vec!['i']]);
        assert_eq!(expr.output, vec!['i', 'j']);
        assert!(expr.splits.is_empty());
        assert_eq!(
            expr.permutation,
            vec![
                PermutationAtom::Axis { axis: 'i', part: 0 },
                PermutationAtom::Axis { axis: 'j', part: 0 },
                PermutationAtom::Input(Operand::Right),
            ]
        );
    }

    #[test]
    fn parses_adjacent_operand_directives() {
        let expr = parse("ij/i~ij|i:2,j:2|iji'j'01");

        assert_eq!(
            expr.permutation,
            vec![
                PermutationAtom::Axis { axis: 'i', part: 0 },
                PermutationAtom::Axis { axis: 'j', part: 0 },
                PermutationAtom::Axis { axis: 'i', part: 1 },
                PermutationAtom::Axis { axis: 'j', part: 1 },
                PermutationAtom::Input(Operand::Left),
                PermutationAtom::Input(Operand::Right),
            ]
        );
    }

    #[test]
    fn parses_identity_expression_as_unary_add() {
        let expr = parse("ij~ij");
        assert_eq!(expr.op, Op::Add);
        assert_eq!(expr.inputs, vec![vec!['i', 'j']]);
        assert_eq!(expr.output, vec!['i', 'j']);
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
    fn rejects_duplicate_output_init_directive() {
        let err = parse_component("+ijk~ij||ij!!k").unwrap_err();
        assert!(err.message.contains("duplicate output init directive"));
    }

    #[test]
    fn rejects_malformed_split_entry() {
        let err = parse_component("+ij~i|i|ij").unwrap_err();
        assert!(err.message.contains("expected `:`"));
    }

    #[test]
    fn rejects_missing_second_schedule_bar() {
        let err = parse_component("+ij~i|i:2").unwrap_err();
        assert!(err.message.contains("expected second `|`"));
    }

    #[test]
    fn rejects_missing_split_factor_integer() {
        let err = parse_component("+ij~i|i:|ij").unwrap_err();
        assert!(err.message.contains("expected integer"));
    }

    #[test]
    fn rejects_duplicate_split_entry() {
        let err = parse_component("+ij~i|i:2,i:4|ij").unwrap_err();
        assert!(err.message.contains("duplicate split entry"));
    }

    #[test]
    fn rejects_invalid_permutation_token() {
        let err = parse_component("+ij~i||i?").unwrap_err();
        assert!(err.message.contains("expected axis, `!`, or operand"));
    }

    #[test]
    fn rejects_empty_source() {
        let err = parse_component("   ").unwrap_err();
        assert!(err.message.contains("expected i-expression"));
    }

    #[test]
    fn rejects_trailing_garbage() {
        let err = parse_component("+ij~i ???").unwrap_err();
        assert!(err.message.contains("unexpected trailing input"));
    }
}
