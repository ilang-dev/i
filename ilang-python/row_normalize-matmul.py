import numpy as np
from ilang import Component as i, I, Tensor

np.random.seed(0)
def as_np(x): return np.asarray(x.data).reshape(x.shape)
def as_i(x): return Tensor(x.tolist())

x = np.random.rand(4,4)
y = np.random.rand(4,2)

ref = (x / x.sum(axis=1, keepdims=True)) @ y

row_normalize = (I & i("+ik~i | k:8 | kik'")) | i("ik/i~ik | k:8 | k1ik'")
mm = i("ik*kj~ijk | k:8 | k0ijk'") | i("+ijk~ij | k:8 | kijk'0")

row_normalize = (I & i("+ik~i | i:2,k:2 | kii'k'")) | i("ik/i~ik | i:2,k:2 | ki1i'k'")
mm = i("ik*kj~ijk | i:2,k:2 | ki0i'jk'") | i("+ijk~ij | i:2,k:2 | kii'jk'0")

#row_normalize = (I & i("+ij~i | i:8 | ii'j")) | i("ij/i~ij | i:8 | i1i'j")
#mm = i("ik*kj~ijk | i:8 | i0i'jk") | i("+ijk~ij | i:8 | ii'jk0")
#row_normalize = (I & i("+ij~i")) | i("ij/i~ij")
#mm = i("ik*kj~ijk") | i("+ijk~ij")
f = row_normalize | mm

x,y = as_i(x), as_i(y)
out = as_np(f.exec(x, y))

print(f._code())
print(f'{ref=}')
print(f'{out=}')

