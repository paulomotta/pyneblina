import sys
from neblina import *

float = 2
complex = 13

def test_vec_add():
    print("test_vec_add")
    init_engine()

    n = 3;
    vec_f = vector_new(n, float)
    vec_2 = vector_new(n, float)

    for i in range(n):
        vector_set(vec_f, i, 1.0)
        vector_set(vec_2, i, 1.0)

    move_vector_device(vec_f)
    move_vector_device(vec_2)

    res = vec_add(vec_f, vec_2)

    out = move_vector_host(res)

    for i in range(n):
        print(vector_get(out,i))

    stop_engine()

def test_vector_matrix_multiplication():
    print("test_vector_matrix_multiplication")
    init_engine()

    n = 3;
    vec_f = vector_new(n, float)
    for i in range(n):
        vector_set(vec_f, i, 1.0)

    mat_f = matrix_new(n,n, float)

    for i in range(n):
        for j in range(n):
            matrix_set(mat_f, i, j, 2.0)

    move_vector_device(vec_f)
    move_matrix_device(mat_f)

    res = matvec_mul(vec_f, mat_f)

    out = move_vector_host(res)

    for i in range(n):
        print(vector_get(out,i))

    stop_engine()

def test_vector_sparse_matrix_multiplication():
    print("test_vector_sparse_matrix_multiplication")
    init_engine()

    n = 10
    vec_f = vector_new(n, float)
    for i in range(n):
        vector_set(vec_f, i, 3.0)

    smat_f = sparse_matrix_new(n,n, float)

    sparse_matrix_set_real(smat_f, 0, 0, 3.);
    sparse_matrix_set_real(smat_f, 0, 1, 3.);
    sparse_matrix_set_real(smat_f, 0, 9, 3.);

    sparse_matrix_set_real(smat_f, 1, 1, 3.);
    sparse_matrix_set_real(smat_f, 1, 5, 3.);
    sparse_matrix_set_real(smat_f, 1, 8, 3.);

    sparse_matrix_set_real(smat_f, 2, 2, 3.);
    sparse_matrix_set_real(smat_f, 2, 4, 3.);
    sparse_matrix_set_real(smat_f, 2, 7, 3.);

    sparse_matrix_set_real(smat_f, 3, 3, 3.);
    sparse_matrix_set_real(smat_f, 3, 1, 3.);
    sparse_matrix_set_real(smat_f, 3, 6, 3.);

    sparse_matrix_pack(smat_f)

    move_vector_device(vec_f)
    move_sparse_matrix_device(smat_f)

    res = sparse_matvec_mul(vec_f, smat_f)

    out = move_vector_host(res)

    for i in range(n):
        print(vector_get(out,i))

    stop_engine()

def test_vec_conjugate():
    print("test_vec_conjugate")
    init_engine()
    n = 3;
    v1 = vector_new(n, complex)

    vector_set(v1, 0, 2.0, 2.0)
    vector_set(v1, 1, 2.0, 2.0)
    vector_set(v1, 2, 2.0, 2.0)

    res = vec_conj(v1);

    #TODO completar o get/set para complexos
    for i in range(n*2):
        print(vector_get(out,2*i))
        print(vector_get(out,2*i+1))

    stop_engine()

def test_vec_sum():
    print("test_vec_sum")
    init_engine()
    n = 4;
    v1 = vector_new(n, float)

    for i in range(n):
        vector_set(v1, i, 2.0)

    move_vector_device(v1)

    res = vec_sum(v1)

    print (res)

    stop_engine()

def test_vec_add_off():
    print("test_vec_add_off")
    init_engine()
    n = 4;
    v1 = vector_new(n, 2) # 2 -> float

    vector_set(v1,0,2.0)
    vector_set(v1,1,2.0)
    vector_set(v1,2,2.0)
    vector_set(v1,3,2.0)

    move_vector_device(v1)

    offset=2
    vec_res = vec_add_off(offset, v1)

    out = move_vector_host(vec_res)

    for i in range(offset):
        print(vector_get(out,i))

    stop_engine()

def test_vec_prod():
    print("test_vec_prod")
    init_engine()
    n = 3;
    v1 = vector_new(n, float)
    v2 = vector_new(n, float)

    for i in range(n):
        vector_set(v1, i, 1.0)
        vector_set(v2, i, 1.0)

    move_vector_device(v1)
    move_vector_device(v2)

    vec_res = vec_prod(v1,v2)

    out = move_vector_host(vec_res)

    for i in range(n):
        print(vector_get(out,i))

    stop_engine()

test_vec_add()
test_vector_matrix_multiplication()
test_vector_sparse_matrix_multiplication()
#test_vec_conjugate()
test_vec_sum()
test_vec_add_off()
test_vec_prod()

sys.exit()
