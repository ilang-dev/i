use super::common::Op;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Expr {
    pub op: Op,
    pub inputs: Vec<Vec<char>>,
    pub output: Vec<char>,
    pub splits: Vec<(char, Vec<usize>)>,
    pub permutation: Vec<PermutationAtom>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PermutationAtom {
    Axis { axis: char, part: usize },
    Input(usize),
}
