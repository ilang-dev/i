import numpy as np
from ilang import Component as i, I, Tensor

np.random.seed(0)
def as_np(x): return np.asarray(x.data).reshape(x.shape)
def as_i(x): return Tensor(x.tolist())

def assert_allclose(got, ref, rtol=1e-4, atol=1e-5):
    got = np.asarray(got)
    ref = np.asarray(ref)
    if got.shape != ref.shape:
        raise AssertionError(f"shape mismatch got={got.shape} ref={ref.shape}")
    if not np.allclose(got, ref, rtol=rtol, atol=atol, equal_nan=True):
        diff = np.abs(got - ref)
        i = np.unravel_index(np.nanargmax(diff), diff.shape)
        raise AssertionError(
            f"not close at {i}: got={got[i]} ref={ref[i]} absdiff={diff[i]} "
            f"(rtol={rtol} atol={atol})"
        )

q = np.random.rand(1024,32)
k = np.random.rand(1024,32)
v = np.random.rand(1024,32)

scores = q @ np.swapaxes(k, -1, -2)
scores = scores - scores.max(axis=-1, keepdims=True)
scores = np.exp(scores)
scores = scores / scores.sum(axis=-1, keepdims=True)
ref = scores @ v

mm_t = i("ij*kj~ikj | i:8,k:8 | kii'k'j") | i("+ikj~ik | i:8,k:8 | kii'k'j0")
row_max_shift = (I & i(">ik~i | i:8,k:8 | ki0i'k'")) | i("ik-i~ik | i:8,k:8 | ki01i'k'")
exp = i("^ik~ik | i:8,k:8 | ki0i'k'")
row_normalize = (I & i("+ik~i | i:8,k:8 | ki0i'k'")) | i("ik/i~ik | i:8,k:8 | ki01i'k'")
mm = i("ik*kj~ijk | i:8,k:8 | ki0i'jk'") | i("+ijk~ij | i:8,k:8 | kii'jk'0")
f = mm_t | row_max_shift | exp | row_normalize | mm

q,k,v = as_i(q), as_i(k), as_i(v)
out = as_np(f.exec(q,k,v))

print(f._code())
print(f'\n{ref=}')
print(f'\n{out=}')
assert_allclose(out, ref)
print()


# === 2D matrix tiling =============================================================================
#x = np.random.rand(4,4)
#y = np.random.rand(4,2)
#
#m = x.max(axis=1, keepdims=True)
#ref = np.exp(x-m) @ y
#
#row_max_shift = (I & i(">ik~i | i:2,k:2 | kii'k'")) | i("ik-i~ik | i:2,k:2 | ki1i'k'")
#exp = i("^ik~ik | i:2,k:2 | ki0i'k'")
#mm = i("ik*kj~ijk | i:2,k:2 | ki0i'jk'") | i("+ijk~ij | i:2,k:2 | kii'jk'0")
#f = row_max_shift | exp | mm
#
#x,y = as_i(x), as_i(y)
#out = as_np(f.exec(x,y))
#
#print(f._code())
#print(f'\n{ref=}')
#print(f'\n{out=}')
#print()
# ==================================================================================================

# === vector tiling ================================================================================
#x = np.random.rand(4)
#y = np.random.rand(4)
#
#m = x.max()
#ref = np.dot(np.exp(x-m),y)
#
#max_shift = (I & i(">i~. | i:2 | ii'")) | i("i-.~i | i:2 | i1i'")
#exp = i("^i~i | i:2 | i0i'")
#dot = i("i*i~i | i:2 | i0i'") | i("+i~. | i:2 | ii'0")
#f = max_shift | exp | dot
#
#x,y = as_i(x), as_i(y)
#out = as_np(f.exec(x,y))
#
#print(f._code())
#print(f'\n{ref=}')
#print(f'\n{out=}')
#print()
# ==================================================================================================
