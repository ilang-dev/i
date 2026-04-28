//! Module IR.
//!
//! This module defines target-independent generated code.
//! `Module` gives public ABI functions and kernel functions.
//! `Fn` gives one fixed-signature function body.
//! `Stmt` values give ordered structured statements.
//! `Expr` values give scalar expressions.
//! `Place` values give assignable locations.
//!
//! Invariants:
//! - `Module.count.signature == Signature::Count`.
//! - `Module.ranks.signature == Signature::Ranks`.
//! - `Module.shapes.signature == Signature::Shapes`.
//! - `Module.exec.signature == Signature::Exec`.
//! - Every function in `Module.kernels` has signature `Signature::Kernel`.
//! - `Module.kernels` is ordered.
//! - `Fn.ident` is module-local.
//! - `Signature` determines the complete function ABI.
//! - `Block` statements execute in order.
//! - `Stmt::Let` binds one local ident.
//! - `Stmt::Set` writes one assignable location.
//! - `Stmt::Alloc` allocates one runtime buffer.
//! - `Stmt::Free` releases one runtime buffer.
//! - `Stmt::Dispatch` calls one kernel function.
//! - `Stmt::Dispatch.reads` is ordered as kernel `readonlys`.
//! - `Stmt::Dispatch.writes` is ordered as kernel `writeables`.
//! - `Stmt::Loop` is a counted loop over `0..bound`.
//! - `Stmt::If` executes its body when `cond` is nonzero.
//! - `Stmt::Return(None)` returns from a void function.
//! - `Stmt::Return(Some(_))` returns from a value function.
//! - `Expr::Op` is scalar 𝚒 computation.
//! - Integer extent, index, and guard arithmetic is expressed with primitive
//!   arithmetic expressions.
//! - `Place` values are valid assignment destinations.
//! - `Field` names one runtime buffer field.
//! - `Cast` names one ABI conversion.
//! - `Ident` values are module-local.
//!

use super::common::Op;

/// One target-independent generated module.
#[derive(Clone, Debug, PartialEq)]
pub struct Module {
    /// Public `count` function.
    pub count: Fn,
    /// Public `ranks` function.
    pub ranks: Fn,
    /// Public `shapes` function.
    pub shapes: Fn,
    /// Public `exec` function.
    pub exec: Fn,
    /// Kernel functions.
    pub kernels: Vec<Fn>,
}

/// One generated function.
#[derive(Clone, Debug, PartialEq)]
pub struct Fn {
    /// Function identifier.
    pub ident: Ident,
    /// Fixed function signature.
    pub signature: Signature,
    /// Function body.
    pub body: Block,
}

/// Fixed function ABI.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Signature {
    /// `count` function signature.
    Count,
    /// `ranks` function signature.
    Ranks,
    /// `shapes` function signature.
    Shapes,
    /// `exec` function signature.
    Exec,
    /// Kernel function signature.
    Kernel,
}

/// One ordered statement block.
#[derive(Clone, Debug, PartialEq)]
pub struct Block(pub Vec<Stmt>);

/// One structured statement.
#[derive(Clone, Debug, PartialEq)]
pub enum Stmt {
    /// One local binding.
    Let {
        /// Bound identifier.
        ident: Ident,
        /// Binding type.
        ty: Type,
        /// Binding value.
        value: Expr,
    },
    /// One assignment.
    Set {
        /// Assignment destination.
        dst: Place,
        /// Assignment value.
        value: Expr,
    },
    /// One runtime buffer allocation.
    Alloc {
        /// Allocated buffer identifier.
        dst: Ident,
        /// Semantic shape dimensions.
        shape: Vec<Expr>,
        /// Physical layout dimensions.
        layout: Vec<Expr>,
    },
    /// One runtime buffer release.
    Free(Ident),
    /// One kernel dispatch.
    Dispatch {
        /// Kernel function identifier.
        kernel: Ident,
        /// Runtime buffers bound to `readonlys`.
        reads: Vec<Ident>,
        /// Runtime buffers bound to `writeables`.
        writes: Vec<Ident>,
    },
    /// One counted loop.
    Loop {
        /// Loop iterator identifier.
        iter: Ident,
        /// Loop bound.
        bound: Expr,
        /// Loop body.
        body: Block,
    },
    /// One conditional block.
    If {
        /// Condition expression.
        cond: Expr,
        /// Conditional body.
        body: Block,
    },
    /// One function return.
    Return(Option<Expr>),
}

/// One scalar expression.
#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    /// One `usize` literal.
    Usize(usize),
    /// One scalar literal.
    Scalar(f32),
    /// One identifier reference.
    Ident(Ident),
    /// One indexed expression.
    Index { base: Box<Expr>, index: Box<Expr> },
    /// One field projection.
    Field { base: Box<Expr>, field: Field },
    /// One ABI conversion.
    Cast { kind: Cast, value: Box<Expr> },
    /// One scalar 𝚒 operation.
    Op { op: Op, args: Vec<Expr> },
    /// Integer addition.
    Add(Box<Expr>, Box<Expr>),
    /// Integer subtraction.
    Sub(Box<Expr>, Box<Expr>),
    /// Integer multiplication.
    Mul(Box<Expr>, Box<Expr>),
    /// Integer division.
    Div(Box<Expr>, Box<Expr>),
    /// Integer remainder.
    Rem(Box<Expr>, Box<Expr>),
    /// Integer less-than comparison.
    Lt(Box<Expr>, Box<Expr>),
}

/// One assignable location.
#[derive(Clone, Debug, PartialEq)]
pub enum Place {
    /// One identifier location.
    Ident(Ident),
    /// One indexed location.
    Index { base: Box<Place>, index: Expr },
    /// One field location.
    Field { base: Box<Place>, field: Field },
}

/// One runtime buffer field.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Field {
    /// Data pointer field.
    Data,
    /// Shape pointer field.
    Shape,
    /// Layout pointer field.
    Layout,
    /// Rank field.
    Rank,
}

/// One ABI conversion.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Cast {
    /// Convert tensor to immutable view.
    View,
    /// Convert mutable tensor to mutable view.
    ViewMut,
    /// Convert mutable view to immutable view.
    ReadOnly,
}

/// One target-independent type.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Type {
    /// `usize` type.
    Usize,
    /// Scalar element type.
    Scalar,
    /// Immutable public tensor type.
    Tensor,
    /// Mutable public tensor type.
    TensorMut,
    /// Immutable view type.
    View,
    /// Mutable view type.
    ViewMut,
    /// Fixed local array type.
    Array(Box<Type>),
}

/// One generated identifier.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Ident(pub String);
