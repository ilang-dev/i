import numpy as np
from ilang import Component as i, Tensor

def as_numpy(x): return np.asarray(x.data).reshape(x.shape)
def as_i(x): return Tensor(x.tolist())

def rand(shape, dtype=np.float32, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape, dtype=dtype)

def assert_allclose(name, got, ref, rtol=1e-4, atol=1e-5):
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

def make_attention_inputs():
    q_len, k_len, d = 64, 64, 4096
    #q_len, k_len, d = 4, 4, 32
    q = rand((q_len, d), seed=1)
    k = rand((k_len, d), seed=2)
    v = rand((k_len, d), seed=3)
    return q, k, v

def np_attention(q, k, v):
    scores = q @ np.swapaxes(k, -1, -2)
    #scores = scores - scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / weights.sum(axis=-1, keepdims=True)
    return weights @ v

def i_attention(q, k, v):
    #mm_t = i("ik*jk~ijk") | i("+ijk~ij")
    #mm = i("ik*kj~ijk") | i("+ijk~ij")

    mm_t = i("ik*jk~ijk|i:2,j:2|iji'j'k") | i("+ijk~ij|i:2,j:2|iji'j'k0")
    mm = i("ik*kj~ijk|i:2,k:2|ik0ji'k'") | i("+ijk~ij|i:2,k:2|ikji'k'0")

    exp = i("^ij~ij|i:2,j:2|iji'j'0")
    row_sum = i("+ij~i|i:2,j:2|iji'j'0")
    row_div = i("ij/i~ij|i:2,j:2|iji'j'01") # TODO need to fuse both inputs

    #attn = mm_t | sm | mm
    attn = mm_t | exp | (mm & row_sum) | row_div
    #attn = mm_t | exp | mm
    return attn.exec(q, k, v)

def i_attuntion(q, k, v):
    mm_t = i("ik*jk~ijk") | i("+ijk~ij")
    mm = i("ik*kj~ijk") | i("+ijk~ij")
    exp = i("^ij~ij")
    I = i("ij~ij")
    row_sum = i("+ij~i")
    row_div = i("ij/i~ij")
    row_normalize = (I & row_sum) | row_div
    sm = exp | row_normalize
    attn = mm_t | sm | mm
    return attn.exec(q, k, v)

q, k, v = make_attention_inputs()

np_out = np_attention(q,k,v)

q = Tensor(q.tolist())
k = Tensor(k.tolist())
v = Tensor(v.tolist())

import time

t = time.time()
i_out = as_numpy(i_attention(q,k,v))
print(f"{time.time() - t}")
assert_allclose("attn", i_out, np_out)

t = time.time()
i_uut = as_numpy(i_attuntion(q,k,v))
print(f"{time.time() - t}")
assert_allclose("attn", i_uut, np_out)

print("looks good.")








exit()



import torch
import torch.nn.functional as F

def as_torch(x): return torch.tensor(x.data).reshape(x.shape)

torch.manual_seed(34)

q_len, k_len, d = 4, 4, 8
q = torch.randn(q_len, d)
k = torch.randn(k_len, d)
v = torch.randn(k_len, d)

out = (q@k.transpose(-2, -1))@v
print(out)

#scores = torch.matmul(q, k.transpose(-2, -1))
#weights = F.softmax(scores, dim=-1)
#out = torch.matmul(weights, v)
#print(out)

#---

from ilang import Component as i, Tensor
def as_torch(x): return torch.tensor(x.data).reshape(x.shape)

q = Tensor(q.tolist())
k = Tensor(k.tolist())
v = Tensor(v.tolist())

#attn = (mm := i("ik*kj~ijk") | i("+ijk~ij||ijk(0)")) | mm
attn = i("ik*jk~ijk|i:2,j:2|iji'j'k") \
    | i("+ijk~ij|i:2,j:2|iji'j'k(0)") \
    | i("ik*kj~ijk|i:2,k:2|ikj(0)i'k'") \
    | i("+ijk~ij|i:2,k:2|ikj(0)i'k'")

out = as_torch(attn.exec(q, k, v))
print(out)
exit()

# naive attention
#mm_t = i("ik*jk~ijk") | i("+ijk~ij")
#mm = i("ik*kj~ijk") | i("+ijk~ij")
#exp = i("^ij~ij")
#I = i("ij~ij")
#row_max = i(">ij~i")
#row_shift = i("ij-i~ij")
#row_max_shift = (I & row_max) | row_shift
#row_sum = i("+ij~i")
#row_div = i("ij/i~ij")
#row_normalize = (I & row_sum) | row_div
#sm = row_max_shift | exp | row_normalize
#attn = mm_t | sm | mm

# rewritten attention
mm_t = i("ik*jk~ijk") | i("+ijk~ij||ijk")
exp = i("^ij~ij||ij(0)")
#row_sum = i("+ij~i")
#row_div = i("ij/i~ij")
#mm = i("ik*kj~ijk") | i("+ijk~ij")
attn = mm_t | exp #| (mm & row_sum) | row_div

#print(as_torch(mm_t.exec(q, k)))
#print(attn._code())
out = attn.exec(q, k) #, v)
print()
print(as_torch(out))

