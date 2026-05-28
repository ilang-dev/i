import time
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
    mm_t = i("ij*kj~ikj | i:16,k:16 | kii'k'j") >> i("+ikj~ik | i:16,k:16 | kii'k'j0")
    row_max_shift = (I & i(">ik~i | i:16,k:16 | ki0i'k'")) >> i("ik-i~ik | i:16,k:16 | ki01i'k'")
    exp = i("^ik~ik | i:16,k:16 | ki0i'k'")
    row_normalize = (I & i("+ik~i | i:16,k:16 | ki0i'k'")) >> i("ik/i~ik | i:16,k:16 | ki01i'k'")
    mm = i("ik*kj~ijk | i:16,k:16 | ki0i'jk'") >> i("+ijk~ij | i:16,k:16 | kii'jk'0")
    attn = mm_t >> row_max_shift >> exp >> row_normalize >> mm
    return attn.exec_numpy(q, k, v)

def i_attuntion(q, k, v):
    mm_t = i("ij*kj~ikj | i:16,k:16 | kii'k'j") >> i("+ikj~ik | i:16,k:16 | kii'k'j0")
    row_max_shift = (I & i(">ik~i | i:16,k:16 | kii'k'")) >> i("ik-i~ik | i:16,k:16 | 1kii'k'")
    exp = i("^ik~ik | i:16,k:16 | kii'k'")
    row_normalize = (I & i("+ik~i | i:16,k:16 | kii'k'")) >> i("ik/i~ik | i:16,k:16 | 1kii'k'")
    mm = i("ik*kj~ijk | i:16,k:16 | kii'jk'") >> i("+ijk~ij | i:16,k:16 | kii'jk'0")
    attn = mm_t >> row_max_shift >> exp >> row_normalize >> mm
    return attn.exec_numpy(q, k, v)

q, k, v = make_attention_inputs()

np_out = np_attention(q,k,v)

t = time.time()
i_out = i_attention(q,k,v)
print(f"flash-attn: {time.time() - t}")
assert_allclose("attn", i_out, np_out)

t = time.time()
i_uut = i_attuntion(q,k,v)
print(f"attn: {time.time() - t}")
assert_allclose("attn", i_uut, np_out)

print("looks good.")

