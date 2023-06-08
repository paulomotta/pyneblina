#!/usr/bin/env python3.8
import sys
import time
from datetime import datetime

from neblina import *

float_ = 2
complex_ = 3

def current_milli_time():
    return round(time.time() * 1000)

def test_vec_add():
    print("test_vec_add")

    n = 3
    vec_f = vector_new(n, float_)
    vec_2 = vector_new(n, float_)

    for i in range(n):
        vector_set(vec_f, i,
                   1.0, 0.0)
        vector_set(vec_2,
                   i, 1.0, 0.0)

    move_vector_device(vec_f)
    move_vector_device(vec_2)

    res = vec_add(vec_f, vec_2)

    move_vector_host(res)

    for i in range(n):
        print(vector_get(res, i))

def test_vec_add_complex():
    print("test_vec_add_complex")

    n = 3
    vec_f = vector_new(n, complex_)
    vec_2 = vector_new(n, complex_)

    for i in range(n):
        vector_set(vec_f, i, 2*i, 3*i)
        vector_set(vec_2, i, 1.0, 1.0)

    move_vector_device(vec_f)
    move_vector_device(vec_2)

    res = vec_add(vec_f, vec_2)

    move_vector_host(res)

    for i in range(n):
        print(str(i) + " " + str(vector_get(res, 2 * i)) + " " + str(vector_get(res, 2 * i + 1)) + "i")

def test_mat_add():
    print("test_mat_add")

    n = 3
    mat_a = matrix_new(n, n, float_)
    mat_b = matrix_new(n, n, float_)

    for i in range(n):
        for j in range(n):
            matrix_set(mat_a, i, j, 2.0, 0.0)
            matrix_set(mat_b, i, j, 2.0, 0.0)

    move_matrix_device(mat_a)
    move_matrix_device(mat_b)

    res = mat_add(mat_a, mat_b)

    move_matrix_host(res)

    for i in range(n):
        for j in range(n):
            print(matrix_get(res, i, j))

def test_scalar_mat_mul():
    print("test_scalar_mat_mul")

    n = 3
    scalar = 2;
    mat_a = matrix_new(n, n, float_)

    for i in range(n):
        for j in range(n):
            matrix_set(mat_a, i, j, 2.0, 0.0)

    move_matrix_device(mat_a)

    res = scalar_mat_mul(scalar, mat_a)

    move_matrix_host(res)

    for i in range(n):
        for j in range(n):
            print(matrix_get(res, i, j))

def test_scalar_vec_mul():
    print("test_scalar_vec_mul")

    n = 3
    scalar = 2;
    vec_a = vector_new(n, float_)

    for i in range(n):
        vector_set(vec_a, i, 2.0, 0.0)

    move_vector_device(vec_a)

    res = scalar_vec_mul(scalar, vec_a)

    move_vector_host(res)

    for i in range(n):
        print(vector_get(res, i))


def test_vector_matrix_multiplication():
    print("test_vector_matrix_multiplication")

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

    move_vector_host(res)

    for i in range(n):
        print(vector_get(res, i))



def test_vector_matrix_multiplication_complex():
    print("test_vector_matrix_multiplication_complex")
    
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
    #tmp = res
    #vector_delete(res)
    res = matvec_mul(res, mat_f)

    #dt = datetime.now()
    #end = dt.microsecond
    end = current_milli_time()
    print(end - ini)

    move_vector_host(res)

    #for i in range(n):
        #print(str(i) + " " + str(vector_get(out, 2 * i)) + " " + str(vector_get(out, 2 * i + 1)) + "i")



def test_vector_sparse_matrix_multiplication():
    print("test_vector_sparse_matrix_multiplication")

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

    move_vector_host(res)

    for i in range(n):
        print(vector_get(res, i))



def test_vector_sparse_matrix_multiplication_complex():
    print("test_vector_sparse_matrix_multiplication_complex")

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

    move_vector_host(res)

    for i in range(n):
        print(str(i) + " " + str(vector_get(res, 2 * i)) + " " + str(vector_get(res, 2 * i + 1)) + "i")



def test_vec_conjugate():
    print("test_vec_conjugate")
    n = 3
    v1 = vector_new(n, complex_)

    for i in range(n):
        vector_set(v1, i, 2.0, 2.0)

    res = vec_conj(v1)

    move_vector_host(res)

    for i in range(n):
        print(str(i) + " " + str(vector_get(res, 2 * i)) + " " + str(vector_get(res, 2 * i + 1)) + "i")



def test_vec_sum():
    print("test_vec_sum")
    n = 4
    v1 = vector_new(n, float_)

    for i in range(n):
        vector_set(v1, i, 2.0, 0.0)

    move_vector_device(v1)

    res = vec_sum(v1)

    print(res)



def test_vec_add_off():
    print("test_vec_add_off")

    n = 4
    v1 = vector_new(n, float_)

    for i in range(n):
        vector_set(v1, i, 2.0, 0.0)

    move_vector_device(v1)

    offset = 2
    vec_res = vec_add_off(offset, v1)

    move_vector_host(vec_res)

    for i in range(offset):
        print(vector_get(vec_res, i))



def test_vec_prod():
    print("test_vec_prod")

    n = 3
    v1 = vector_new(n, float_)
    v2 = vector_new(n, float_)

    for i in range(n):
        vector_set(v1, i, 1.0, 0.0)
        vector_set(v2, i, 1.0, 0.0)

    move_vector_device(v1)
    move_vector_device(v2)

    vec_res = vec_prod(v1, v2)

    move_vector_host(vec_res)

    for i in range(n):
        print(vector_get(vec_res, i))




def test_vec_prod_complex():
    print("test_vec_prod_complex")
    n = 3
    v1 = vector_new(n, complex_)
    v2 = vector_new(n, complex_)

    for i in range(n):
        vector_set(v1, i, 1.0, 2.0)
        vector_set(v2, i, 1.0, 2.0)

    move_vector_device(v1)
    move_vector_device(v2)

    vec_res = vec_prod(v1, v2)

    move_vector_host(vec_res)

    for i in range(n):
        print(str(i) + " " + str(vector_get(vec_res, 2 * i)) + " " + str(vector_get(vec_res, 2 * i + 1)) + "i")


def test_complex_scalar_new():
    print("test_complex_scalar_new")
    scalar = complex_new(2,2);
    

def test_complex_scalar_mat_mul():
    print("test_complex_scalar_mat_mul")

    n = 3
    scalar = complex_new(2,2);
    mat_a = matrix_new(n, n, float_)
    mat_b = matrix_new(n, n, complex_)

    for i in range(n):
        for j in range(n):
            matrix_set(mat_a, i, j, 2.0, 0.0)
            matrix_set(mat_b, i, j, 2.0, 2.0)

    move_matrix_device(mat_a)
    move_matrix_device(mat_b)

    res = complex_scalar_mat_mul(scalar, mat_a)

    move_matrix_host(res)

    for i in range(n):
        for j in range(n):
            print(matrix_get(res, i, j))
    
    res = complex_scalar_mat_mul(scalar, mat_b)

    move_matrix_host(res)

    for i in range(n):
        for j in range(n):
            print(matrix_get(res, i, j))

def test_complex_scalar_vec_mul():
    print("test_complex_scalar_vec_mul")

    n = 3
    scalar = complex_new(2,2);
    vec_a = vector_new(n, FLOAT)
    vec_b = vector_new(n, complex_)

    for i in range(n):
        vector_set(vec_a, i, 2.0, 0.0)
        vector_set(vec_b, i, 2.0, 2.0)

    move_vector_device(vec_a)
    move_vector_device(vec_b)

    print("1")
    res = complex_scalar_vec_mul(scalar, vec_a)
    print("2")
    move_vector_host(res)

    for i in range(n):
        print(str(i) + " " + str(vector_get(res, 2 * i)) + " " + str(vector_get(res, 2 * i + 1)) + "i")

    res = complex_scalar_vec_mul(scalar, vec_b)

    move_vector_host(res)

    for i in range(n):
        print(str(i) + " " + str(vector_get(res, 2 * i)) + " " + str(vector_get(res, 2 * i + 1)) + "i")


init_engine(GPU,0)

test_vec_add()
test_vec_add_complex()
test_mat_add()
test_scalar_mat_mul()
test_scalar_vec_mul()
test_vector_matrix_multiplication()
test_vector_matrix_multiplication_complex()
test_vector_sparse_matrix_multiplication()
test_vector_sparse_matrix_multiplication_complex()
test_vec_conjugate()
test_vec_sum()
test_vec_add_off()
test_vec_prod()
test_vec_prod_complex()
test_complex_scalar_new()
test_complex_scalar_vec_mul()
test_complex_scalar_mat_mul()
stop_engine()
