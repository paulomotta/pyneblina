#!/usr/bin/env python3.8
import sys
import time
from datetime import datetime

from neblina import *

float_ = 2
complex_ = 13

def current_milli_time():
    return round(time.time() * 1000)

def test_vec_add():
    print("test_vec_add")
    init_engine(0)

    n = 3
    vec_f = vector_new(n, float_)
    vec_2 = vector_new(n, float_)

    for i in range(n):
        vector_set(vec_f, i, 1.0, 0.0)
        vector_set(vec_2, i, 1.0, 0.0)

    move_vector_device(vec_f)
    move_vector_device(vec_2)

    res = vec_add(vec_f, vec_2)

    out = move_vector_host(res)

    for i in range(n):
        print(vector_get(out, i))

    stop_engine()


def test_vector_matrix_multiplication():
    print("test_vector_matrix_multiplication")
    init_engine(0)

    n = 3
    vec_f = vector_new(n, float_)
    for i in range(n):
        vector_set(vec_f, i, 1.0, 0.0)

    mat_f = matrix_new(n, n, float_)

    for i in range(n):
        for j in range(n):
            matrix_set(mat_f, i, j, 2.0, 0.0)

    move_vector_device(vec_f)
    move_matrix_device(mat_f)

    res = matvec_mul(vec_f, mat_f)

    out = move_vector_host(res)

    for i in range(n):
        print(vector_get(out, i))

    stop_engine()


def test_vector_matrix_multiplication_complex():
    print("test_vector_matrix_multiplication_complex")
    init_engine(0)

    n = 7000
    vec_f = vector_new(n, complex_)
    for i in range(n):
        vector_set(vec_f, i, 2.0, 2.0)

    mat_f = matrix_new(n, n, complex_)

    for i in range(n):
        for j in range(n):
            matrix_set(mat_f, i, j, 3.0, 3.0)

    #dt = datetime.now()
    #ini = dt.microsecond
    ini = current_milli_time()
    print(ini)
    move_vector_device(vec_f)
    move_matrix_device(mat_f)

    res = matvec_mul(vec_f, mat_f)

    #dt = datetime.now()
    #end = dt.microsecond
    end = current_milli_time()
    print(end - ini)

    out = move_vector_host(res)

    #for i in range(n):
        #print(str(i) + " " + str(vector_get(out, 2 * i)) + " " + str(vector_get(out, 2 * i + 1)) + "i")

    stop_engine()


def test_vector_sparse_matrix_multiplication():
    print("test_vector_sparse_matrix_multiplication")
    init_engine(0)

    n = 10
    vec_f = vector_new(n, float_)
    for i in range(n):
        vector_set(vec_f, i, 3.0, 0.0)

    smat_f = sparse_matrix_new(n, n, float_)

    sparse_matrix_set(smat_f, 0, 0, 3., 0.0)
    sparse_matrix_set(smat_f, 0, 1, 3., 0.0)
    sparse_matrix_set(smat_f, 0, 9, 3., 0.0)

    sparse_matrix_set(smat_f, 1, 1, 3., 0.0)
    sparse_matrix_set(smat_f, 1, 5, 3., 0.0)
    sparse_matrix_set(smat_f, 1, 8, 3., 0.0)

    sparse_matrix_set(smat_f, 2, 2, 3., 0.0)
    sparse_matrix_set(smat_f, 2, 4, 3., 0.0)
    sparse_matrix_set(smat_f, 2, 7, 3., 0.0)

    sparse_matrix_set(smat_f, 3, 3, 3., 0.0)
    sparse_matrix_set(smat_f, 3, 1, 3., 0.0)
    sparse_matrix_set(smat_f, 3, 6, 3., 0.0)

    sparse_matrix_pack(smat_f)

    move_vector_device(vec_f)
    move_sparse_matrix_device(smat_f)

    res = sparse_matvec_mul(vec_f, smat_f)

    out = move_vector_host(res)

    for i in range(n):
        print(vector_get(out, i))

    stop_engine()


def test_vector_sparse_matrix_multiplication_complex():
    print("test_vector_sparse_matrix_multiplication_complex")
    init_engine(0)

    n = 10
    vec_f = vector_new(n, complex_)
    for i in range(n):
        vector_set(vec_f, i, 3.0, 3.0)

    smat_f = sparse_matrix_new(n, n, complex_)

    sparse_matrix_set(smat_f, 0, 0, 3., 3.0)
    sparse_matrix_set(smat_f, 0, 1, 3., 3.0)
    sparse_matrix_set(smat_f, 0, 9, 3., 3.0)

    sparse_matrix_set(smat_f, 1, 1, 3., 3.0)
    sparse_matrix_set(smat_f, 1, 5, 3., 3.0)
    sparse_matrix_set(smat_f, 1, 8, 3., 3.0)

    sparse_matrix_set(smat_f, 2, 2, 3., 3.0)
    sparse_matrix_set(smat_f, 2, 4, 3., 3.0)
    sparse_matrix_set(smat_f, 2, 7, 3., 3.0)

    sparse_matrix_set(smat_f, 3, 3, 3., 3.0)
    sparse_matrix_set(smat_f, 3, 1, 3., 3.0)
    sparse_matrix_set(smat_f, 3, 6, 3., 3.0)

    sparse_matrix_pack(smat_f)

    move_vector_device(vec_f)
    move_sparse_matrix_device(smat_f)

    res = sparse_matvec_mul(vec_f, smat_f)

    out = move_vector_host(res)

    for i in range(n):
        print(str(i) + " " + str(vector_get(out, 2 * i)) + " " + str(vector_get(out, 2 * i + 1)) + "i")

    stop_engine()


def test_vec_conjugate():
    print("test_vec_conjugate")
    init_engine(0)
    n = 3
    v1 = vector_new(n, complex_)

    for i in range(n):
        vector_set(v1, i, 2.0, 2.0)

    res = vec_conj(v1)

    out = move_vector_host(res)

    for i in range(n):
        print(str(i) + " " + str(vector_get(out, 2 * i)) + " " + str(vector_get(out, 2 * i + 1)) + "i")

    stop_engine()


def test_vec_sum():
    print("test_vec_sum")
    init_engine(0)
    n = 4
    v1 = vector_new(n, float_)

    for i in range(n):
        vector_set(v1, i, 2.0, 0.0)

    move_vector_device(v1)

    res = vec_sum(v1)

    print(res)

    stop_engine()


def test_vec_add_off():
    sparse_matrix_set(smat_f, 0, 0, 3., 3.0)
    sparse_matrix_set(smat_f, 0, 1, 3., 3.0)
    sparse_matrix_set(smat_f, 0, 9, 3., 3.0)

    sparse_matrix_set(smat_f, 1, 1, 3., 3.0)
    sparse_matrix_set(smat_f, 1, 5, 3., 3.0)
    sparse_matrix_set(smat_f, 1, 8, 3., 3.0)

    sparse_matrix_set(smat_f, 2, 2, 3., 3.0)
    sparse_matrix_set(smat_f, 2, 4, 3., 3.0)
    sparse_matrix_set(smat_f, 2, 7, 3., 3.0)

    sparse_matrix_set(smat_f, 3, 3, 3., 3.0)
    sparse_matrix_set(smat_f, 3, 1, 3., 3.0)
    sparse_matrix_set(smat_f, 3, 6, 3., 3.0)
    print("test_vec_add_off")
    init_engine(0)
    n = 4
    v1 = vector_new(n, float_)

    for i in range(n):
        vector_set(v1, i, 2.0, 0.0)

    move_vector_device(v1)

    offset = 2
    vec_res = vec_add_off(offset, v1)

    out = move_vector_host(vec_res)

    for i in range(offset):
        print(vector_get(out, i))

    stop_engine()


def test_vec_prod():
    print("test_vec_prod")
    init_engine(0)
    n = 3
    v1 = vector_new(n, float_)
    v2 = vector_new(n, float_)

    for i in range(n):
        vector_set(v1, i, 1.0, 0.0)
        vector_set(v2, i, 1.0, 0.0)

    move_vector_device(v1)
    move_vector_device(v2)

    vec_res = vec_prod(v1, v2)

    out = move_vector_host(vec_res)

    for i in range(n):
        print(vector_get(out, i))

    stop_engine()


def test_vec_prod_complex():
    print("test_vec_prod_complex")
    init_engine(0)
    n = 3
    v1 = vector_new(n, complex_)
    v2 = vector_new(n, complex_)

    for i in range(n):
        vector_set(v1, i, 1.0, 2.0)
        vector_set(v2, i, 1.0, 2.0)

    move_vector_device(v1)
    move_vector_device(v2)

    vec_res = vec_prod(v1, v2)

    out = move_vector_host(vec_res)

    for i in range(n):
        print(str(i) + " " + str(vector_get(out, 2 * i)) + " " + str(vector_get(out, 2 * i + 1)) + "i")

    stop_engine()


#test_vec_add()
#test_vector_matrix_multiplication()
test_vector_matrix_multiplication_complex()
#test_vector_sparse_matrix_multiplication()
#test_vector_sparse_matrix_multiplication_complex()
#test_vec_conjugate()
#test_vec_sum()
#test_vec_add_off()
#test_vec_prod()
#test_vec_prod_complex()
