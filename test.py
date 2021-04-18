from neblina import *


init_engine()

n = 3;
vec_f = vector_new(n, 2) # 2 -> float
print(vec_f)
vec_2 = vector_new(n, 2)
print(vec_2)

vector_set(vec_f,0,1.0)
vector_set(vec_f,1,1.0)
vector_set(vec_f,2,1.0)

vector_set(vec_2,0,1.0)
vector_set(vec_2,1,1.0)
vector_set(vec_2,2,1.0)

move_vector_device(vec_f)
move_vector_device(vec_2)

res = vec_add(vec_f, vec_2)

out = move_vector_host(res)

print(vector_get(out,0))
print(vector_get(out,1))
print(vector_get(out,2))

mat_f = matrix_new(n,n, 2) # 2 -> float
print(mat_f)

for i in range(n):
    for j in range(n):
        matrix_set(mat_f,i,j,2.0)

move_matrix_device(mat_f)

#res = vec_add(vec_f, vec_2)

#out = move_vector_host(res)

for i in range(n):
    for j in range(n):
        print(matrix_get(mat_f,i,j))

n = 10
smat_f = sparse_matrix_new(n,n, 2) # 2 -> float

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

vec_res = matvec_mul(vec_f,mat_f)

out = move_vector_host(vec_res)
n=3
for i in range(n):
    print(vector_get(out,i))


n = 3;
v1 = vector_new(n, 2) # 2 -> float
v2 = vector_new(n, 2)

vector_set(v1,0,1.0)
vector_set(v1,1,1.0)
vector_set(v1,2,1.0)

vector_set(v2,0,1.0)
vector_set(v2,1,1.0)
vector_set(v2,2,1.0)

move_vector_device(v1)
move_vector_device(v2)


vec_res = vec_prod(v1,v2)

out = move_vector_host(vec_res)
n=3
for i in range(n):
    print(vector_get(out,i))

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

n = 4;
v1 = vector_new(n, 2) # 2 -> float

vector_set(v1,0,2.0)
vector_set(v1,1,2.0)
vector_set(v1,2,2.0)
vector_set(v1,3,2.0)

move_vector_device(v1)

res = vec_sum(v1)

print (res)

n = 3;
v1 = vector_new(n, 13) # 13 -> complex

vector_set(v1,0, 2.0, 2.0)
vector_set(v1,1, 2.0, 2.0)
vector_set(v1,2, 2.0, 2.0)
    
res = vec_conj(v1);

#TODO completar o get/set para complexos
for i in range(offset):
    print(vector_get(out,i))

stop_engine()