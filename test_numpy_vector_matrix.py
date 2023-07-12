import numpy as np

import neblina as nbl

n = 3
# Set the seed for reproducibility (optional)
np.random.seed(42)
matrix_size = (n, n)

float_vector = np.random.random(n)
matrix1 = np.random.random(matrix_size)

# Perform float vector-sparse matrix multiplication
result_float = matrix1.dot(float_vector)


vector_c = np.random.random(n) + np.random.random(n) * 1j
matrix_c = np.random.random(matrix_size) + np.random.random(matrix_size) * 1j

# Perform matrix multiplication
result_c = matrix_c.dot(vector_c)

###########################################################
# neblina calculations
###########################################################

nbl.init_engine(nbl.CPU,0)

vec_f = nbl.vector_new(n, nbl.FLOAT)
for i in range(0,n):
    nbl.vector_set(vec_f, i, float_vector[i], 0)
    # print(i, " ", float_vector[i])

mat_f1 = nbl.matrix_new(n, n, nbl.FLOAT)

for i in range(0,n):
    for j in range(0,n):
        nbl.matrix_set(mat_f1, i, j, matrix1[i,j], 0)
        # print(i , " ", j, " ", matrix1[i,j])

nbl.move_matrix_device(mat_f1)
nbl.move_vector_device(vec_f)

res = nbl.matvec_mul(vec_f, mat_f1)

nbl.move_vector_host(res)

for i in range(n):
    # print(i, " ", nbl.vector_get(res, i), " ", result_float[i], " ", (nbl.vector_get(res, i) - result_float[i]))
    assert nbl.vector_get(res, i) == result_float[i]

vec_c = nbl.vector_new(n, nbl.COMPLEX)
for i in range(n):
    nbl.vector_set(vec_c, i, vector_c[i].real, vector_c[i].imag)
    # print(i, " ", vector_c[i])

mat_c1 = nbl.matrix_new(n, n, nbl.COMPLEX)

for i in range(0,n):
    for j in range(0,n):
        nbl.matrix_set(mat_c1, i, j, matrix_c[i,j].real, matrix_c[i,j].imag)
        # print(i , " ", j, " ", matrix_c[i,j])


nbl.move_vector_device(vec_c)
nbl.move_matrix_device(mat_c1)

res = nbl.matvec_mul(vec_c, mat_c1)

nbl.move_vector_host(res)

for i in range(0,n):
    # print(i, " ", nbl.vector_get(res, 2*i), " ", result_c[i].real, " ", (nbl.vector_get(res, 2*i) - result_c[i].real))
    # print(i, " ", nbl.vector_get(res, 2*i+1), " ", result_c[i].imag, " ", (nbl.vector_get(res, 2*i+1) - result_c[i].imag))
    assert nbl.vector_get(res, 2 * i) == result_c[i].real
    assert nbl.vector_get(res, 2 * i + 1) == result_c[i].imag
    assert nbl.vector_get(res, 2 * i) + nbl.vector_get(res, 2 * i + 1)*1j == result_c[i]

nbl.stop_engine()
