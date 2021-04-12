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
vec_f = vector_new(n, 2)

vector_set(vec_f,0,1.0)

move_vector_device(vec_f)

stop_engine()