#include "Python.h"
#include "neblina.h"
#include "neblina_std.h"
#include "libneblina.h"

int fastfactorial(int n) {
    if (n <= 1)
        return 1;
    else
        return n * fastfactorial(n - 1);
}

static PyObject* factorial(PyObject* self, PyObject* args) {
    int n;
    if (!PyArg_ParseTuple(args, "i", &n))
        return NULL;
    int result = fastfactorial(n);
    
    // https://docs.python.org/3/c-api/capsule.html
    // 
    return Py_BuildValue("i", result);
}

static PyObject* py_init_engine(PyObject* self, PyObject* args){
    cl_int err;
    cl_uint num_platforms;
    
    err = clGetPlatformIDs(0, NULL, &num_platforms);
    if (err == CL_SUCCESS) {
            //std::cout << "Success. Platforms available: " << num_platforms
            //        << std::endl;
    } else {
            //std::cout << "Error. Platforms available: " << num_platforms
            //        << std::endl;
    }

    InitCLEngine();

    Py_RETURN_NONE;
}

static PyObject* py_stop_engine(PyObject* self, PyObject* args){
    ReleaseCLInfo(clinfo);

    Py_RETURN_NONE;
}

static void delete(PyObject* self) {
    free ((int*)PyCapsule_GetPointer(self, "remove"));
}

static void py_vector_delete(PyObject* self) {
    vector_t* vec = (vector_t*)PyCapsule_GetPointer(self, "py_vector_new");
    printf("vec %p\n",vec);
    printf("vec->value %p\n",&(vec->value));
//    free ((void *)&(vec->value));
    free ((void *)vec);
}


static PyObject* py_vector_new(PyObject* self, PyObject* args){
    int len;
    int data_type;
    if (!PyArg_ParseTuple(args, "ii", &len,&data_type)) return NULL;
    printf("create %d\n",len);
    vector_t * a = vector_new(len, data_type);
    printf("malloc %p\n",a);
    PyObject* po = PyCapsule_New((void*)a, "py_vector_new", py_vector_delete);
    printf("capsule_new %p\n",po);
    return po;
}

static PyObject* py_vector_set(PyObject* self, PyObject* args) {
    
    PyObject* pf = NULL;
    int n;
    double value;
    if(!PyArg_ParseTuple(args, "Oid:py_vector_set", &pf, &n, &value)) return NULL;

    printf("print %d\n",n);
    printf("pf %p\n",pf);
    vector_t * vec = (vector_t *)PyCapsule_GetPointer(pf, "py_vector_new");
    printf("vec %p\n",vec);
    vec->value.f[n] = value;
    printf("%lf\n",vec->value.f[n]);
    return Py_None;
}

static PyObject* py_vector_get(PyObject* self, PyObject* args) {
    
    PyObject* pf = NULL;
    int n;
    if(!PyArg_ParseTuple(args, "Oi:py_vector_set", &pf, &n)) return NULL;

    printf("print %d\n",n);
    printf("pf %p\n",pf);
    vector_t * vec = (vector_t *)PyCapsule_GetPointer(pf, "py_vector_new");
    printf("vec %p\n",vec);
    printf("%lf\n",vec->value.f[n]);
    PyObject * result = PyFloat_FromDouble((double)vec->value.f[n]);
    return result;
}

static PyObject* py_move_vector_device(PyObject* self, PyObject* args) {
    
    PyObject* pf = NULL;
    if(!PyArg_ParseTuple(args, "O:py_move_vector_device", &pf)) return NULL;

    printf("pf %p\n",pf);
    
    vector_t * vec = (vector_t *)PyCapsule_GetPointer(pf, "py_vector_new");
    printf("vec %p\n",vec);
    vecreqdev(vec);
    return Py_None;
}

static PyObject* py_move_vector_host(PyObject* self, PyObject* args) {
    
    PyObject* pf = NULL;
    if(!PyArg_ParseTuple(args, "O:py_move_vector_device", &pf)) return NULL;

    printf("pf %p\n",pf);
    
    vector_t * vec = (vector_t *)PyCapsule_GetPointer(pf, "py_vector_new");
    printf("vec %p\n",vec);
    int n = (vec->type==T_FLOAT?vec->len:2*vec->len);
    vector_t * out = vector_new(vec->len, vec->type);
    cl_int status = clEnqueueReadBuffer(clinfo.q, vec->mem, CL_TRUE, 0, n * sizeof (double), out->value.f, 0, NULL, NULL);
    CLERR
    PyObject* po = PyCapsule_New((void*)out, "py_vector_new", py_vector_delete);
    return po;
}

static PyObject* py_vec_add(PyObject* self, PyObject* args) {
    
    PyObject* a = NULL;
    PyObject* b = NULL;
    if(!PyArg_ParseTuple(args, "OO:py_vec_add", &a, &b)) return NULL;

    printf("a %p\n",a);
    printf("b %p\n",b);
    
    vector_t * vec_a = (vector_t *)PyCapsule_GetPointer(a, "py_vector_new");
        printf("vec_a %p\n",vec_a);
    vector_t * vec_b = (vector_t *)PyCapsule_GetPointer(b, "py_vector_new");
        printf("vec_b %p\n",vec_b);

    
    //TODO completar o vec_add
    object_t ** in = convertToObject(vec_a,vec_b);
    
    vector_t * r = (vector_t *) vec_add((void **) in, NULL );
    

    PyObject* po = PyCapsule_New((void*)r, "py_vector_new", py_vector_delete);
    return po;
}

static void py_matrix_delete(PyObject* self) {
    matrix_t* mat = (matrix_t*)PyCapsule_GetPointer(self, "py_matrix_new");
    printf("mat %p\n",mat);
    printf("mat->value %p\n",&(mat->value));
//    free ((void *)&(mat->value));
    free ((void *)mat);
}


static PyObject* py_matrix_new(PyObject* self, PyObject* args){
    int rows;
    int cols;
    int data_type;
    if (!PyArg_ParseTuple(args, "iii", &rows,&cols,&data_type)) return NULL;
    printf("create %d\n",rows);
    printf("create %d\n",cols);
    matrix_t * a = matrix_new(rows, cols, data_type);
    printf("malloc %p\n",a);
    PyObject* po = PyCapsule_New((void*)a, "py_matrix_new", py_matrix_delete);
    printf("capsule_new %p\n",po);
    return po;
}

static PyObject* py_matrix_set(PyObject* self, PyObject* args) {
    
    PyObject* pf = NULL;
    int i;
    int j;
    double value;
    if(!PyArg_ParseTuple(args, "Oiid:py_matrix_set", &pf, &i, &j, &value)) return NULL;

    printf("print (%d,%d)\n",i,j);
    printf("pf %p\n",pf);
    matrix_t * mat = (matrix_t *)PyCapsule_GetPointer(pf, "py_matrix_new");
    printf("mat %p\n",mat);
    mat->value.f[i*mat->ncol + j] = value;
    printf("%lf\n",mat->value.f[i*mat->ncol + j]);
    return Py_None;
}

static PyObject* py_matrix_get(PyObject* self, PyObject* args) {
    
    PyObject* pf = NULL;
    int i;
    int j;
    if(!PyArg_ParseTuple(args, "Oii:py_matrix_set", &pf, &i, &j)) return NULL;

    printf("print (%d,%d)\n",i,j);
    printf("pf %p\n",pf);
    matrix_t * mat = (matrix_t *)PyCapsule_GetPointer(pf, "py_matrix_new");
    printf("mat %p\n",mat);
    printf("%lf\n",mat->value.f[i*mat->ncol + j]);
    PyObject * result = PyFloat_FromDouble((double)mat->value.f[i*mat->ncol + j]);
    return result;
}


static PyObject* py_move_matrix_device(PyObject* self, PyObject* args) {
    
    PyObject* pf = NULL;
    if(!PyArg_ParseTuple(args, "O:py_move_matrix_device", &pf)) return NULL;

    printf("pf %p\n",pf);
    
    matrix_t * mat = (matrix_t *)PyCapsule_GetPointer(pf, "py_matrix_new");
    printf("mat %p\n",mat);
    matreqdev(mat);
    return Py_None;
}

static PyObject* py_move_matrix_host(PyObject* self, PyObject* args) {
    
    PyObject* pf = NULL;
    if(!PyArg_ParseTuple(args, "O:py_move_matrix_device", &pf)) return NULL;

    printf("pf %p\n",pf);
    
    matrix_t * mat = (matrix_t *)PyCapsule_GetPointer(pf, "py_matrix_new");
    printf("mat %p\n",mat);
    int n = (mat->type==T_FLOAT?mat->nrow*mat->ncol:2*mat->nrow*mat->ncol);
    matrix_t * out = matrix_new(mat->nrow, mat->ncol, mat->type);
    cl_int status = clEnqueueReadBuffer(clinfo.q, mat->mem, CL_TRUE, 0, n * sizeof (double), out->value.f, 0, NULL, NULL);
    CLERR
    PyObject* po = PyCapsule_New((void*)out, "py_matrix_new", py_matrix_delete);
    return po;
}

static void py_sparse_matrix_delete(PyObject* self) {
    smatrix_t* mat = (smatrix_t*)PyCapsule_GetPointer(self, "py_sparse_matrix_new");
    printf("smat %p\n",mat);
    free ((void *)mat);
}


static PyObject* py_sparse_matrix_new(PyObject* self, PyObject* args){
    int rows;
    int cols;
    int data_type;
    if (!PyArg_ParseTuple(args, "iii", &rows,&cols,&data_type)) return NULL;
    printf("create %d\n",rows);
    printf("create %d\n",cols);
    smatrix_t * a = smatrix_new(rows, cols, data_type);
    printf("malloc %p\n",a);
    PyObject* po = PyCapsule_New((void*)a, "py_sparse_matrix_new", py_sparse_matrix_delete);
    printf("capsule_new %p\n",po);
    return po;
}

static PyObject* create(PyObject* self, PyObject* args) {
    int n;
    if (!PyArg_ParseTuple(args, "i", &n))
        return NULL;
    printf("create %d\n",n);
    int * vec = (int *)malloc(n*sizeof(int));
    printf("malloc %p\n",vec);
    vec[2]=42; 
    return PyCapsule_New((void*)vec, "create", delete);
}

static PyObject* print(PyObject* self, PyObject* args) {
    
    PyObject* pf = NULL;
    int n;
    //if(!PyArg_UnpackTuple(args, "print", 1, 2, &pf,&n)) return NULL;
    if(!PyArg_ParseTuple(args, "Oi:print", &pf, &n)) return NULL;

    printf("print %d\n",n);
    printf("pf %p\n",pf);
    int* vec = (int *)PyCapsule_GetPointer(pf, "create");
    printf("vec %p\n",vec);
    
    printf("%d\n",vec[n]);
    return Py_None;
}



static PyMethodDef mainMethods[] = {
    {"factorial", factorial, METH_VARARGS, "Calculate the factorial of n"},
    {"create", create, METH_VARARGS, "create"},
    {"prin", print, METH_VARARGS, "print"},
    {"init_engine", py_init_engine, METH_VARARGS, "init_engine"},
    {"stop_engine", py_stop_engine, METH_VARARGS, "stop_engine"},
    {"vector_new", py_vector_new, METH_VARARGS, "vector_new"},
    {"vector_set", py_vector_set, METH_VARARGS, "vector_set"},
    {"vector_get", py_vector_get, METH_VARARGS, "vector_get"},
    {"move_vector_device", py_move_vector_device, METH_VARARGS, "move_vector_device"},
    {"move_vector_host", py_move_vector_host, METH_VARARGS, "move_vector_host"},
    {"matrix_new", py_matrix_new, METH_VARARGS, "matrix_new"},
    {"matrix_set", py_matrix_set, METH_VARARGS, "matrix_set"},
    {"matrix_get", py_matrix_get, METH_VARARGS, "matrix_get"},
    {"move_matrix_device", py_move_matrix_device, METH_VARARGS, "move_matrix_device"},
    {"move_matrix_host", py_move_matrix_host, METH_VARARGS, "move_matrix_host"},
    {"sparse_matrix_new", py_sparse_matrix_new, METH_VARARGS, "sparse_matrix_new"},
    {"vec_add", py_vec_add, METH_VARARGS, "vec_add"},
    {NULL, NULL, 0, NULL}
};

static PyModuleDef cmath = {
    PyModuleDef_HEAD_INIT,
    "cmath", "Factorial Calculation",
    -1,
    mainMethods
};

PyMODINIT_FUNC PyInit_cmath(void) {
    return PyModule_Create(&cmath);
}


