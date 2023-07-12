import numpy as np
import neblina as nbl
import time

def current_milli_time():
    return round(time.time() * 1000)

# n = 26244
n = 2500
# Set the seed for reproducibility (optional)
np.random.seed(42)

# Create the first matrix with random data
matrix1 = np.random.random((n, n))

# Create the second matrix with random data
matrix2 = np.random.random((n, n))

ini = current_milli_time()
# Perform matrix multiplication
result = np.dot(matrix1, matrix2)
end = current_milli_time()
print("np.dot(matrix1, matrix2) total", (end - ini) )

matrixc1 = np.random.random((n, n)) + np.random.random((n, n)) * 1j

# Create the second matrix with random complex data
matrixc2 = np.random.random((n, n)) + np.random.random((n, n)) * 1j

ini = current_milli_time()
resultc = np.dot(matrixc1, matrixc2)
end = current_milli_time()
print("np.dot(matrixc1, matrixc2) total", (end - ini) )

###########################################################
# neblina calculations
###########################################################

nbl.init_engine(nbl.CPU,0)
mat_c1 = nbl.matrix_new(n, n, nbl.FLOAT)
mat_c2 = nbl.matrix_new(n, n, nbl.FLOAT)

ini = current_milli_time()
for i in range(0,n):
    for j in range(0,n):
        nbl.matrix_set(mat_c1, i, j, matrix1[i,j], 0)
        nbl.matrix_set(mat_c2, i, j, matrix2[i,j], 0)
end = current_milli_time()
print("nbl.matrix_set total", (end - ini) )

nbl.move_matrix_device(mat_c1)
nbl.move_matrix_device(mat_c2)

ini = current_milli_time()
res = nbl.mat_mul(mat_c1, mat_c2)
end = current_milli_time()
print("nbl.mat_mul(mat_c1, mat_c2) total", (end - ini) )

nbl.move_matrix_host(res)

for i in range(0,n):
    for j in range(0,n):
        assert nbl.matrix_get(res,i,j) == result[i,j]

mat_c1 = nbl.matrix_new(n, n, nbl.COMPLEX)
mat_c2 = nbl.matrix_new(n, n, nbl.COMPLEX)

ini = current_milli_time()
for i in range(0,n):
    for j in range(0,n):
        nbl.matrix_set(mat_c1, i, j, matrixc1[i,j].real, matrixc1[i,j].imag)
        nbl.matrix_set(mat_c2, i, j, matrixc2[i,j].real, matrixc2[i,j].imag)

end = current_milli_time()
print("nbl.matrix_set complex total", (end - ini) )

nbl.move_matrix_device(mat_c1)
nbl.move_matrix_device(mat_c2)

ini = current_milli_time()
res = nbl.mat_mul(mat_c1, mat_c2)
end = current_milli_time()
print("nbl.mat_mul(mat_c1, mat_c2) complex total", (end - ini) )

nbl.move_matrix_host(res)

for i in range(0,n):
    for j in range(0,n):
        assert nbl.matrix_get(res,2*i,2*j) == resultc[i,j].real
        assert nbl.matrix_get(res,2*i,2*j + 1) == resultc[i,j].imag
        assert nbl.matrix_get(res, 2*i, 2*j) + nbl.matrix_get(res, 2*i, 2*j + 1)*1j == resultc[i,j]



nbl.stop_engine()

# Create another matrix for comparison
# comparison_matrix = np.random.random((n, n))

# Compare the result with the comparison matrix item by item
# comparison_result = np.allclose(result, comparison_matrix)

# Print the comparison result
# if comparison_result:
#     print("The calculated result is the same as the comparison matrix.")
# else:
#     print("The calculated result is different from the comparison matrix.")
