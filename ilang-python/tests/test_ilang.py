#!/usr/bin/env python3

import numpy as np
from ilang import i
from ilang.testing import allclose

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"


def as_numpy(x):
    return np.asarray(x.data).reshape(x.shape)


def as_i(x):
    return i.Tensor(x.tolist())


def assert_i_allclose(name, got, ref, rtol=1e-5, atol=1e-8):
    if not allclose(got, ref, rtol=rtol, atol=atol):
        raise AssertionError(f"{name}: allclose returned False")


def assert_i_not_allclose(name, got, ref, rtol=1e-5, atol=1e-8):
    if allclose(got, ref, rtol=rtol, atol=atol):
        raise AssertionError(f"{name}: allclose returned True")


def assert_i_allclose_shape_mismatch(name, got, ref):
    try:
        allclose(got, ref)
    except AssertionError:
        return
    raise AssertionError(f"{name}: allclose did not reject shape mismatch")


def assert_allclose(name, got, ref, rtol=1e-5, atol=1e-8):
    got = np.asarray(got)
    ref = np.asarray(ref)
    if got.shape != ref.shape:
        raise AssertionError(f"{name}: shape mismatch got={got.shape} ref={ref.shape}")
    if not np.allclose(got, ref, rtol=rtol, atol=atol, equal_nan=True):
        diff = np.abs(got - ref)
        i = np.unravel_index(np.nanargmax(diff), diff.shape)
        raise AssertionError(
            f"{name}: not close at {i}: got={got[i]} ref={ref[i]} absdiff={diff[i]} "
            f"(rtol={rtol} atol={atol})"
        )


def rand(shape, dtype=np.float32, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape, dtype=dtype)


def run_case(name, make_inputs, i_impl, np_impl, rtol=1e-5, atol=1e-6):
    np_inputs = make_inputs()
    i_inputs = tuple(as_i(x) for x in np_inputs)
    got_i = i_impl(*i_inputs)
    got = as_numpy(got_i)
    ref = np_impl(*np_inputs)
    assert_allclose(name, got, ref, rtol=rtol, atol=atol)


# example
def make_matmul_inputs():
    A = rand((8, 16), seed=1)
    B = rand((16, 7), seed=2)
    return (A, B)


def i_matmul(A, B):
    p = i("ik*kj~ijk|i:2,k:2|ii'jkk'")
    a = i("+ijk~ij|i:2,k:2|ii'j!kk'0")
    f = p >> a
    return f.exec(A, B)


def np_matmul(A, B):
    return A @ B


def make_attention_inputs():
    batch, heads, q_len, k_len, d = 2, 3, 4, 4, 8
    q = rand((batch, heads, q_len, d), seed=1)
    k = rand((batch, heads, k_len, d), seed=2)
    v = rand((batch, heads, k_len, d), seed=3)
    return q, k, v


def np_attention(q, k, v):
    scores = q @ np.swapaxes(k, -1, -2)
    scores = scores - scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / weights.sum(axis=-1, keepdims=True)
    return weights @ v


def i_attention(q, k, v):
    mm_t = i("bhij*bhkj~bhikj | i:16,k:16 | bhkii'k'j") >> i(
        "+bhikj~bhik | i:16,k:16 | bhkii'k'j0"
    )
    row_max_shift = (i.I & i(">bhik~bhi | i:16,k:16 | bhkii'k'")) >> i(
        "bhik-bhi~bhik | i:16,k:16 | bh1kii'k'"
    )
    exp = i("^bhik~bhik | i:16,k:16 | bhkii'k'")
    row_normalize = (i.I & i("+bhik~bhi | i:16,k:16 | bhkii'k'")) >> i(
        "bhik/bhi~bhik | i:16,k:16 | bh1kii'k'"
    )
    mm = i("bhik*bhkj~bhijk | i:16,k:16 | bhkii'jk'") >> i(
        "+bhijk~bhij | i:16,k:16 | bhkii'jk'0"
    )
    attn = mm_t >> row_max_shift >> exp >> row_normalize >> mm
    return attn.exec(q, k, v)


def make_mlp_inputs():
    x = rand((8,), seed=1)
    w1 = rand((16, 8), seed=2)
    w2 = rand((7, 16), seed=3)
    return x, w1, w2


def np_mlp(x, w1, w2):
    return w2 @ np.maximum(w1 @ x, 0)


def i_mlp(x, w1, w2):
    mvp = i("ij*j~ij") >> i("+ij~i")
    amvp = mvp >> i(">i~i")
    I = i("ij~ij")  # noqa: E741
    f = (I | amvp) >> mvp
    return f.exec(w2, w1, x)


# --- reductions ---
# TODO make these explicitly test for unfavorable initially conditions


def make_rowmax_inputs():
    return (rand((16, 8), seed=1),)


def np_rowmax(x):
    return np.max(x, axis=1)


def i_rowmax(x):
    return i(">ij~i").exec(x)


def make_rowmin_inputs():
    return (rand((16, 8), seed=1),)


def np_rowmin(x):
    return np.min(x, axis=1)


def i_rowmin(x):
    return i("<ij~i").exec(x)


def make_rowaccum_inputs():
    return (rand((16, 8), seed=1),)


def np_rowaccum(x):
    return np.sum(x, axis=1)


def i_rowaccum(x):
    return i("+ij~i").exec(x)


# --- binding ---


def make_bind_inputs():
    x = rand((8,), seed=1)
    y = rand((8,), seed=2)
    return x, y


def np_bind_sub(x, y):
    return x - y


def i_bind_sub(x, y):
    return i("i-i~i")(x)(y)()


def i_bind_skip_sub(x, y):
    return i("i-i~i")(None, y)(x)()


def np_chain_bound(x, y):
    return y - x


def i_chain_bound(x, y):
    f = i("i~i")
    g = i("i-i~i")
    return (f >> g(y))(x)()


def make_compose_call_inputs():
    return (rand((8,), seed=3),)


def np_compose_call(x):
    return np.exp(np.exp(x))


def i_compose_call(x):
    f = i("^i~i")
    return f(f)(x)()


def i_compose_and_bind_call(x):
    f = i("^i~i")
    return f(f, x)()


def make_literal_inputs():
    return ()


def np_literal_vector_add():
    return np.array([5, 7, 9], dtype=np.float32)


def i_literal_vector_exec():
    return i("i+i~i").exec([1, 2, 3], [4, 5, 6])


def i_literal_vector_bind():
    return i("i+i~i")([1, 2, 3])([4, 5, 6])()


def np_literal_scalar_add():
    return np.array(5, dtype=np.float32)


def i_literal_scalar_exec():
    return i(".+.~.").exec(2.0, 3)


# --- testing helpers ---


def test_allclose_exact_scalar():
    assert_i_allclose("allclose_exact_scalar", i.Tensor(1.0), i.Tensor(1.0), atol=0.0)


def test_allclose_exact_vector():
    assert_i_allclose(
        "allclose_exact_vector",
        i.Tensor([1.0, 2.0, 3.0]),
        i.Tensor([1.0, 2.0, 3.0]),
        atol=0.0,
    )


def test_allclose_with_absolute_tolerance():
    assert_i_allclose(
        "allclose_with_absolute_tolerance_close",
        i.Tensor([1.0, 2.001]),
        i.Tensor([1.0, 2.0]),
        atol=0.01,
    )
    assert_i_not_allclose(
        "allclose_with_absolute_tolerance_far",
        i.Tensor([1.0, 2.1]),
        i.Tensor([1.0, 2.0]),
        atol=0.01,
    )


def test_allclose_shape_mismatch():
    assert_i_allclose_shape_mismatch(
        "allclose_shape_mismatch",
        i.Tensor([1.0, 2.0]),
        i.Tensor([[1.0, 2.0]]),
    )


def test_allclose_detects_different_values():
    assert_i_not_allclose(
        "allclose_detects_different_values",
        i.Tensor([1.0, 2.0]),
        i.Tensor([1.0, 3.0]),
        atol=0.0,
    )


def test_allclose_works_on_component_output():
    got = i("i+i~i").exec([1.0, 2.0], [3.0, 4.0])
    ref = i.Tensor([4.0, 6.0])
    assert_i_allclose("allclose_works_on_component_output", got, ref, atol=0.0)


if __name__ == "__main__":
    cases = [
        ("matmul", make_matmul_inputs, i_matmul, np_matmul),
        ("attention", make_attention_inputs, i_attention, np_attention),
        ("mlp", make_mlp_inputs, i_mlp, np_mlp),
        ("rowmax", make_rowmax_inputs, i_rowmax, np_rowmax),
        ("rowmin", make_rowmin_inputs, i_rowmin, np_rowmin),
        ("rowaccum", make_rowaccum_inputs, i_rowaccum, np_rowaccum),
        ("bind_sub", make_bind_inputs, i_bind_sub, np_bind_sub),
        ("bind_skip_sub", make_bind_inputs, i_bind_skip_sub, np_bind_sub),
        ("chain_bound", make_bind_inputs, i_chain_bound, np_chain_bound),
        ("compose_call", make_compose_call_inputs, i_compose_call, np_compose_call),
        (
            "compose_and_bind_call",
            make_compose_call_inputs,
            i_compose_and_bind_call,
            np_compose_call,
        ),
        (
            "literal_vector_exec",
            make_literal_inputs,
            i_literal_vector_exec,
            np_literal_vector_add,
        ),
        (
            "literal_vector_bind",
            make_literal_inputs,
            i_literal_vector_bind,
            np_literal_vector_add,
        ),
        (
            "literal_scalar_exec",
            make_literal_inputs,
            i_literal_scalar_exec,
            np_literal_scalar_add,
        ),
    ]

    testing_cases = [
        ("allclose_exact_scalar", test_allclose_exact_scalar),
        ("allclose_exact_vector", test_allclose_exact_vector),
        ("allclose_with_absolute_tolerance", test_allclose_with_absolute_tolerance),
        ("allclose_shape_mismatch", test_allclose_shape_mismatch),
        ("allclose_detects_different_values", test_allclose_detects_different_values),
        ("allclose_works_on_component_output", test_allclose_works_on_component_output),
    ]

    fails = 0
    for name, make_inputs, i_impl, np_impl in cases:
        try:
            run_case(name, make_inputs, i_impl, np_impl)
            print(f"{GREEN}ok{RESET}: {name}")
        except Exception as e:
            fails += 1
            print(f"{RED}fail{RESET}: {name}: {e}")

    for name, test in testing_cases:
        try:
            test()
            print(f"{GREEN}ok{RESET}: {name}")
        except Exception as e:
            fails += 1
            print(f"{RED}fail{RESET}: {name}: {e}")

    if fails:
        raise SystemExit(f"{fails} test(s) failed")
