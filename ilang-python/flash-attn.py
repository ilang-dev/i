import numpy as np
from ilang import Component as i, Tensor, I

def rand(shape, dtype=np.float32, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape, dtype=dtype)

def assert_allclose(name, got, ref, rtol=3e-4, atol=5e-5):
    got = np.asarray(got)
    ref = np.asarray(ref)
    if got.shape != ref.shape:
        raise AssertionError(f"{name}: shape mismatch got={got.shape} ref={ref.shape}")
    if not np.allclose(got, ref, rtol=rtol, atol=atol, equal_nan=True):
        finite_match = np.isfinite(got) == np.isfinite(ref)
        if not finite_match.all():
            i = np.unravel_index(np.argmin(finite_match), finite_match.shape)
            raise AssertionError(
                f"{name}: finite mismatch at {i}: got={got[i]} ref={ref[i]} "
                f"got_finite={np.isfinite(got[i])} ref_finite={np.isfinite(ref[i])} "
                f"(rtol={rtol} atol={atol})"
            )
        diff = np.abs(got - ref)
        tol = atol + rtol * np.abs(ref)
        bad = diff > tol
        i = np.unravel_index(np.argmax(diff / tol), diff.shape)
        raise AssertionError(
            f"{name}: not close at {i}: got={got[i]} ref={ref[i]} absdiff={diff[i]} "
            f"tol={tol[i]} (rtol={rtol} atol={atol})"
        )

def make_attention_inputs():
    q_len, k_len, d = 32, 32, 2048
    q = rand((q_len, d), seed=1)
    k = rand((k_len, d), seed=2)
    v = rand((k_len, d), seed=3)
    return q, k, v

def np_attention(q, k, v):
    scores = q @ np.swapaxes(k, -1, -2)
    scores = scores - scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / weights.sum(axis=-1, keepdims=True)
    return weights @ v

def i_attention(q, k, v):
    """FlashAttention. Produces a single kernel with minimal intermediate allocations."""
    mm_t = i("ik*jk~ijk | i:16,j:16 | jii'j'k") >> i("+ijk~ij | i:16,j:16 | jii'j'k0")
    row_max_shift = (I & i(">ij~i | i:16,j:16 | ji0i'j'")) >> i("ij-i~ij | i:16,j:16 | ji01i'j'")
    exp = i("^ij~ij | i:16,j:16 | ji0i'j'")
    row_normalize = (I & i("+ij~i | i:16,j:16 | ji0i'j'")) >> i("ij/i~ij | i:16,j:16 | ji01i'j'")
    mm = i("ij*jk~ikj | i:16,j:16 | ji0i'kj'") >> i("+ikj~ik | i:16,j:16 | jii'kj'0")
    attn = mm_t >> row_max_shift >> exp >> row_normalize >> mm
    print("FlashAttention generated code:\n")
    print(attn._code())
    return attn.exec_numpy(q, k, v)

def i_attuntion(q, k, v):
    """Naive attention. Produces 5 tiled kernels, allocating full intermediate buffers."""
    mm_t = i("ik*jk~ijk | i:16,j:16 | jii'j'k") >> i("+ijk~ij | i:16,j:16 | jii'j'k0")
    row_max_shift = (I & i(">ij~i | i:16,j:16 | jii'j'")) >> i("ij-i~ij | i:16,j:16 | 1jii'j'")
    exp = i("^ij~ij | i:16,j:16 | jii'j'")
    row_normalize = (I & i("+ij~i | i:16,j:16 | jii'j'")) >> i("ij/i~ij | i:16,j:16 | 1jii'j'")
    mm = i("ij*jk~ikj | i:16,j:16 | jii'kj'") >> i("+ikj~ik | i:16,j:16 | jii'kj'0")
    attn = mm_t >> row_max_shift >> exp >> row_normalize >> mm
    return attn.exec_numpy(q, k, v)

q, k, v = make_attention_inputs()

np_out = np_attention(q,k,v)

i_out = i_attention(q,k,v)
assert_allclose("attn", i_out, np_out)
print("FlashAttention output matches NumPy reference to at least rtol=3e-4, atol=5e-5.")

i_uut = i_attuntion(q,k,v)
assert_allclose("attn", i_uut, np_out)
print("Naive Attention output matches NumPy reference to at least rtol=3e-4, atol=5e-5.")

