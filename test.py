from cmath import factorial
from cmath import create
from cmath import prin
from cmath import *

#print(factorial(6))

#vec = create(5);
#print(vec)
#prin(vec,2);

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

stop_engine()