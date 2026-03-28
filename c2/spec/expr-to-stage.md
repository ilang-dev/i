# Expr To ScheduledStage

This document specifies the transformation
`Expr -> ScheduledStage`.

## Axis Numbering

- Stage axes are numbered by canonical local axis order.
- Canonical local axis order is:
  - output axes in source order
  - then any remaining input axes in first-appearance order over inputs in source order

## Stage

- `Stage.op = Expr.op`.
- `Stage.rank` is the number of canonical local axes.
- Each input pattern lowers to `Index` by replacing each source axis with its canonical stage axis.
- The output pattern lowers in the same way.
- Repeated input axes are preserved.

## Splits

- `Schedule.splits` is indexed by canonical stage axis.
- For each canonical axis:
  - if the expression has a split entry for that axis, its factors are copied in source order
  - otherwise its split list is empty

## Default Schedule

- If `Expr.permutation` is empty:
  - `Schedule.order` is all loop parts in ascending stage-axis order
  - for each stage axis, loop parts are listed in ascending part order
  - `Schedule.compute_sites` is `None` for every input
  - `Schedule.init_site` is determined from the stage and the loop order

## Explicit Schedule

- If `Expr.permutation` is non-empty:
  - each `PermutationAtom::Axis { axis, part }` appends that loop part to `Schedule.order`
  - each `PermutationAtom::Input(i)` sets `Schedule.compute_sites[i]` to the site of the most recent axis atom
  - `PermutationAtom::Input(i)` requires a preceding axis atom
  - `PermutationAtom::Bang` sets the explicit init site to:
    - `Site::Root` if no axis atom has appeared
    - otherwise the site of the most recent axis atom

## Init Site

- A pointwise stage has no init site.
- A reduction stage has an init site.
- If the reduction stage has no output loop parts, the init site is `Site::Root`.
- Otherwise the init site is the site of the last output loop part in `Schedule.order`.
- Therefore every output loop part must appear before every non-output loop part.
- If `Expr.permutation` contains `Bang`, it must match the required init site exactly.

## Validation

- The input expression is validated before lowering.
- The lowered scheduled stage is validated after lowering.

## Source Of Truth

- `src/lower/expr_to_stage.rs`
- `src/check/component.rs`
- `src/check/stage.rs`
