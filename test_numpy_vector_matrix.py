import numpy as np
import neblina as nbl
import time
import pytest

def current_milli_time():
    return round(time.time() * 1000)

n = 9000
# Set the seed for reproducibility (optional)
np.random.seed(42)
matrix_size = (n, n)

float_vector = np.random.random(n)
matrix1 = np.random.random(matrix_size)

# Perform float vector-sparse matrix multiplication
ini = current_milli_time()
result_float = matrix1.dot(float_vector)
end = current_milli_time()
print("np matrix1.dot(float_vector) total", (end - ini) )


vector_c = np.random.random(n) + np.random.random(n) * 1j
matrix_c = np.random.random(matrix_size) + np.random.random(matrix_size) * 1j

# Perform matrix multiplication
ini = current_milli_time()
result_c = matrix_c.dot(vector_c)
end = current_milli_time()
print("np matrix_c.dot(vector_c) total", (end - ini) )

###########################################################
# neblina calculations
###########################################################

nbl.init_engine(nbl.GPU,0)

vec_f = nbl.load_numpy_array(float_vector)

mat_f1 = nbl.load_numpy_matrix(matrix1)

nbl.move_matrix_device(mat_f1)
nbl.move_vector_device(vec_f)

ini = current_milli_time()
res = nbl.matvec_mul(vec_f, mat_f1)
end = current_milli_time()
print("nbl.matvec_mul(vec_f, mat_f1) total", (end - ini) )


nbl.move_vector_host(res)

np_res = nbl.retrieve_numpy_array(res)
for i in range(n):
    # print(i, " ", nbl.vector_get(res, i), " ", result_float[i], " ", (nbl.vector_get(res, i) - result_float[i]))
    assert nbl.vector_get(res, i) == pytest.approx(result_float[i], 0.000000000001)
    assert np_res[i] == pytest.approx(result_float[i], 0.000000000001)

vec_c = nbl.load_numpy_array(vector_c)

mat_c1 = nbl.load_numpy_matrix(matrix_c)

nbl.move_vector_device(vec_c)
nbl.move_matrix_device(mat_c1)

ini = current_milli_time()
res = nbl.matvec_mul(vec_c, mat_c1)
end = current_milli_time()
print("nbl.matvec_mul(vec_c, mat_c1) total", (end - ini) )


nbl.move_vector_host(res)

np_res = nbl.retrieve_numpy_array(res)

for i in range(0,n):
    # print(i, " ", nbl.vector_get(res, 2*i), " ", result_c[i].real, " ", (nbl.vector_get(res, 2*i) - result_c[i].real))
    # print(i, " ", nbl.vector_get(res, 2*i+1), " ", result_c[i].imag, " ", (nbl.vector_get(res, 2*i+1) - result_c[i].imag))
    assert nbl.vector_get(res, 2 * i) == pytest.approx(result_c[i].real, 0.000000000001)
    assert nbl.vector_get(res, 2 * i + 1) == pytest.approx(result_c[i].imag, 0.000000000001)
    assert nbl.vector_get(res, 2 * i) + nbl.vector_get(res, 2 * i + 1)*1j == pytest.approx(result_c[i], 0.000000000001)
    assert np_res[i] == pytest.approx(result_c[i], 0.000000000001)

nbl.stop_engine()
print("all tests passed")
