import numpy as np
from ilang import Component as i, I, Tensor

np.random.seed(0)
def as_np(x): return np.asarray(x.data).reshape(x.shape)
def as_i(x): return Tensor(x.tolist())

q = np.random.rand(4,2)
k = np.random.rand(4,2)
v = np.random.rand(4,2)

scores = q @ np.swapaxes(k, -1, -2)
scores = np.exp(scores)
scores = scores / scores.sum(axis=-1, keepdims=True)
ref = scores @ v

mm_t = i("ij*kj~ikj | i:2,k:2 | kii'k'j") | i("+ikj~ik | i:2,k:2 | kii'k'j0")
exp = i("^ik~ik | i:2,k:2 | ki0i'k'")
row_normalize = (I & i("+ik~i | i:2,k:2 | ki0i'k'")) | i("ik/i~ik | i:2,k:2 | ki01i'k'")
mm = i("ik*kj~ijk | i:2,k:2 | ki0i'jk'") | i("+ijk~ij | i:2,k:2 | kii'jk'0")
f = mm_t | exp | row_normalize | mm

q,k,v = as_i(q), as_i(k), as_i(v)
out = as_np(f.exec(q,k,v))

print(f._code())
print(f'\n{ref=}')
print(f'\n{out=}')
print()

