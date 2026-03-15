#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ValueId(pub u32);

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct StageId(pub u32);

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct AxisId(pub u32);

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct FusionId(pub u32);

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct KernelId(pub u32);

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct BufferId(pub u32);

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct LoopId(pub u32);

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Symbol(pub String);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScalarType {
    Bool,
    I32,
    I64,
    F32,
    F64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Extent {
    Known(u64),
    Param(Symbol),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorType {
    pub scalar: ScalarType,
    pub shape: Vec<Extent>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Axis {
    pub id: AxisId,
    pub extent: Extent,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AffineExpr {
    Const(i64),
    Axis(AxisId),
    Add(Box<AffineExpr>, Box<AffineExpr>),
    Mul(i64, Box<AffineExpr>),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AffineMap {
    pub results: Vec<AffineExpr>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum IndexExpr {
    Const(i64),
    Param(Symbol),
    Loop(LoopId),
    Add(Box<IndexExpr>, Box<IndexExpr>),
    Mul(Box<IndexExpr>, Box<IndexExpr>),
    DivFloor(Box<IndexExpr>, Box<IndexExpr>),
    DivCeil(Box<IndexExpr>, Box<IndexExpr>),
    Mod(Box<IndexExpr>, Box<IndexExpr>),
    Min(Box<IndexExpr>, Box<IndexExpr>),
    Max(Box<IndexExpr>, Box<IndexExpr>),
}
