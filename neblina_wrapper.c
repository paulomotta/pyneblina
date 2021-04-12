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

static void delete(PyObject* self) {
    free ((int*)PyCapsule_GetPointer(self, "remove"));
}

static void py_vector_delete(PyObject* self) {
    vector_t* vec = (vector_t*)PyCapsule_GetPointer(self, "py_vector_new");
    printf("vec %p\n",vec);
    printf("vec->value %p\n",&vec->value);
    free ((void *)&vec->value);
    free ((void *)vec);
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

static PyObject* py_move_vector_device(PyObject* self, PyObject* args) {
    
    PyObject* pf = NULL;
    if(!PyArg_ParseTuple(args, "O:py_move_vector_device", &pf)) return NULL;

    printf("pf %p\n",pf);
    
    vector_t * vec = (vector_t *)PyCapsule_GetPointer(pf, "py_vector_new");
    printf("vec %p\n",vec);
    vecreqdev(vec);
    return Py_None;
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
    {"move_vector_device", py_move_vector_device, METH_VARARGS, "move_vector_device"},
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


