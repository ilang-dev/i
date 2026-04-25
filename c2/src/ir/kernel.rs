use super::common::Op;

struct Kernel(Block);

struct Block(Vec<Action>);

// top-level statement appearing in kernels
enum Action {
    Loop {
        id: LoopId,
        extent: Extent,
        guarded: bool,
        body: Block,
    },
    Init {
        op: Op,
        write: Access,
        zero_checks: Vec<LoopId>,
    },
    Compute {
        op: Op,
        write: Access,
        reads: Vec<Access>,
    },
}

struct LoopId(usize);

enum Extent {
    Factor(usize),
    Semantic(ShapeRef),
    Split { base: ShapeRef, factors: Vec<usize> },
}

struct ShapeRef {
    buf: Buffer,
    dim: usize,
}

struct Buffer {
    arg: Arg,
    ind: usize,
}

enum Arg {
    Readonly,
    Writeable,
}

struct Access {
    buf: Buffer,
    index: Vec<Iter>,
}

enum Iter {
    Raw(LoopId),
    Reconstructed {
        loops: Vec<LoopId>,
        factors: Vec<usize>,
    },
}
