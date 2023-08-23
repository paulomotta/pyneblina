import numpy as np

import neblina as nbl

n = 3
# Set the seed for reproducibility (optional)
np.random.seed(42)
vec_size = (n)

vec1 = np.random.random(n)
vec2 = np.random.random(n)

###########################################################
# neblina calculations
###########################################################
result_float = vec1 + vec2

vec_c1 = np.random.random(n) + np.random.random(n) * 1j
vec_c2 = np.random.random(n) + np.random.random(n) * 1j
result_c = vec_c1 + vec_c2

nbl.init_engine(nbl.CPU,0)

n_vec1 = nbl.load_numpy_array(vec1)
n_vec2 = nbl.load_numpy_array(vec2)

nbl.move_vector_device(n_vec1)
nbl.move_vector_device(n_vec2)

res = nbl.vec_add(n_vec1, n_vec2)

nbl.move_vector_host(res)

for i in range(n):
    assert nbl.vector_get(res, i) == result_float[i]

n_vec_c1 = nbl.load_numpy_array(vec_c1)
n_vec_c2 = nbl.load_numpy_array(vec_c2)

nbl.move_vector_device(n_vec_c1)
nbl.move_vector_device(n_vec_c2)

res_c = nbl.vec_add(n_vec_c1, n_vec_c2)

nbl.move_vector_host(res_c)

for i in range(0,n):
    # print(i, " ", nbl.vector_get(res_c, 2*i), " ", result_c[i].real, " ", (nbl.vector_get(res_c, 2*i) - result_c[i].real))
    # print(i, " ", nbl.vector_get(res_c, 2*i+1), " ", result_c[i].imag, " ", (nbl.vector_get(res_c, 2*i+1) - result_c[i].imag))
    assert nbl.vector_get(res_c, 2 * i) == result_c[i].real
    assert nbl.vector_get(res_c, 2 * i + 1) == result_c[i].imag
    assert nbl.vector_get(res_c, 2 * i) + nbl.vector_get(res_c, 2 * i + 1)*1j == result_c[i]

nbl.stop_engine()
print("all tests passed")