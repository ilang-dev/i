import numpy as np
from ilang import Component as i, I, Tensor

def as_np(x): return np.asarray(x.data).reshape(x.shape)
def as_i(x): return Tensor(x.tolist())

"""

l = 0.
for i0 in range(ceil_div(N,2)):
    for i1 in range(2):
        i = recon(i0, i1, 2)
        l = l + x[i]

for i0 in range(ceil_div(N,2)):
    for i1 in range(2):
        i = recon(i0, i1, 2)
        s = x/l

"""

x = Tensor([1., 2., 3., 2.])
y = Tensor([2., 4., 6., 4.])
#normalize = (I & i("+i~.|i:2|ii'")) | i("i/.~i|i:2|1ii'")
#dot = i("i*i~i|i:2|i0i'") | i("+i~.|i:2|ii'0")
#f = normalize | dot
#print(f._code())
#print(as_np(normalize.exec(x, y)))
#exit()

f = (I & i(">i~.|i:8|ii'")) | i("i-.~i|i:8|ii'") \
    | i("^i~i|i:8|i0i'") \
    | (I & i("+i~.|i:8|ii'")) | i("i/.~i|i:8|ii'") \
    | i("i*i~i|i:8|ii'") | i("+i~.|i:8|ii'0")

mm_t = i("ik*jk~ijk") | i("+ijk~ij")
max_shift = (I & i(">ij~i")) | i("ij-i~ij")
exp = i("^ij~ij")
normalize = (I & i("+ij~i")) | i("ij/i~ij")
softmax = max_shift | exp | normalize
mm = i("ik*kj~ijk") | i("+ijk~ij")
attn = mm_t | softmax | mm




normalize = (I & i("+i~. | i:8 | ii'")) | i("i/.~i | i:8 | i1i'")
dot = i("i*i~i | i:8 | i0i'") | i("+i~. | i:8 | !ii'0")
f = normalize | dot

print(f._code())
print(f.exec(x, y))
exit()

# dot(normalize(x), y)

l = 0.0
s = alloc(Ni)
p = alloc(Ni)
o = 0.0
for i0 in range(ceil_div(Ni,8)):
    _l = l
    for i1 in range(8):
        i = recon(i0, i1, 8)
        l += x[i]
    for i1 in range(8):
        i = recon(i0, i1, 8)
        s[i] = x[i] / l
    for i1 in range(8):
        i = recon(i0, i1, 8)
        o *= _l
    for i1 in range(8):
        i = recon(i0, i1, 8)
        p[i] = s[i] * y[i]
        o += p[i]

