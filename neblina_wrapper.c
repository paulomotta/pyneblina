#include "Python.h"
#include <numpy/arrayobject.h>
#include "neblina.h"
#include "neblina_std.h"
#include "neblina_vector.h"
#include "neblina_matrix.h"
#include "neblina_smatrix.h"
#include "neblina_complex.h"
#include "bridge_api.h"
#include "libneblina.h"

bridge_manager_t bridge_manager;
int bridge_index = 0;

static PyObject* py_init_engine(PyObject* self, PyObject* args){
    //cl_int err;
    //cl_uint num_platforms;
    int device;
    int bridge;
    
    if (!PyArg_ParseTuple(args, "ii", &bridge, &device)) return NULL;
    //err = clGetPlatformIDs(0, NULL, &num_platforms);
    //if (err == CL_SUCCESS) {
            //std::cout << "Success. Platforms available: " << num_platforms
            //        << std::endl;
    //} else {
            //std::cout << "Error. Platforms available: " << num_platforms
            //        << std::endl;
    //}

    //InitCLEngine(device);
    bridge_index = device;
    char * lib_name;
    switch(bridge){
        case 0:
            lib_name = "/usr/local/lib64/libneblina-cpu-bridge.so";
            break;
        case 1:
            lib_name = "/usr/local/lib64/libneblina-opencl-bridge.so";
            break;
        default:
            lib_name = "/usr/local/lib64/libneblina-cpu-bridge.so";
            break;
    }
    load_plugin(&bridge_manager, lib_name, bridge_index);
    bridge_manager.bridges[bridge_index].InitEngine_f(device);
    // printf("3\n");
    Py_RETURN_NONE;
}

static PyObject* py_stop_engine(PyObject* self, PyObject* args){
    //ReleaseCLInfo(clinfo);
    release_plugin(&bridge_manager, bridge_index);
//    double v1[2];
//    double v2[2];
//    double *res;
//    v1[0] = 1;
//    v1[1] = 2;
//    v2[0] = 1;
//    v2[1] = 2;
//    res = bridge_manager.bridges[bridge_index].addVectorF_f(&v1,&v2,2);
//    for (int i=0;i<2;i++) {
//        printf("%f\n",res[i]);
//    }
    Py_RETURN_NONE;
}

static void py_complex_delete(PyObject* self) {
    complex_t* comp = (complex_t*)PyCapsule_GetPointer(self, "py_complex_new");
    bridge_manager.bridges[bridge_index].complex_delete(comp);
}


static PyObject* py_complex_new(PyObject* self, PyObject* args){
    double real;
    double imag;
    if (!PyArg_ParseTuple(args, "dd", &real, &imag)) return NULL;

    complex_t * a = bridge_manager.bridges[bridge_index].complex_new(real, imag);

    PyObject* po = PyCapsule_New((void*)a, "py_complex_new", py_complex_delete);

    return po;
}

static void py_vector_delete(PyObject* self) {
    vector_t* vec = (vector_t*)PyCapsule_GetPointer(self, "py_vector_new");
    //printf("vec %p\n",vec);
    //printf("vec->value %p\n",&(vec->value));
    //free ((void *)vec->value.f);
    //free ((void *)vec);
    bridge_manager.bridges[bridge_index].vector_delete(vec);
}


static PyObject* py_vector_new(PyObject* self, PyObject* args){
    int len;
    int data_type;
    if (!PyArg_ParseTuple(args, "ii", &len,&data_type)) return NULL;
    //printf("create %d\n",len);
    vector_t * a = bridge_manager.bridges[bridge_index].vector_new(len, data_type, 1, NULL);
    //printf("malloc %p\n",a);
    PyObject* po = PyCapsule_New((void*)a, "py_vector_new", py_vector_delete);
    //printf("capsule_new %p\n",po);
    return po;
}

static PyObject* py_load_numpy_array(PyObject* self, PyObject* args){
    PyObject* a = NULL;

    int i = 1;
    if(!PyArg_ParseTuple(args, "O:py_load_numpy_array", &a)) return NULL;

    float* dataArrayA = (float*)PyArray_DATA((PyArrayObject*)a);
    // printf("%p\n",dataArrayA);
    // printf("%d\n",i++);

    npy_intp* shape = PyArray_DIMS((PyArrayObject*)a);
    // printf("%d\n",i++);

    int rows = shape[0];
    int cols = shape[1];
    // printf("rows=%d cols=%d sizeof(double)=%ld\n",rows, cols, sizeof(double));
    int len = shape[0];
    PyArray_Descr* dtype = PyArray_DESCR(a);
    char typekind = dtype->kind;  // 'i' for signed integer, 'f' for float, etc.
    int itemsize = dtype->elsize; // Size of each element in bytes
    //printf("Data Type: %c, Item Size: %d\n", typekind, itemsize);

    int data_type = (dtype->kind == 'f' ? T_FLOAT : T_COMPLEX);
    // printf("%d\n",i++);

    vector_t * vec_a = bridge_manager.bridges[bridge_index].vector_new(len, data_type, 0, dataArrayA);
    // printf("%d\n",i++);

    // printf("%p\n",vec_a->value.f);

    PyObject* po = PyCapsule_New((void*)vec_a, "py_vector_new", py_vector_delete);

    return po;
}

static PyObject* py_retrieve_numpy_array(PyObject* self, PyObject* args){
    
    PyObject* pf = NULL;

    if(!PyArg_ParseTuple(args, "O:py_retrieve_numpy_array", &pf)) return NULL;

    // printf("pf %p\n",pf);
    vector_t * vec = (vector_t *)PyCapsule_GetPointer(pf, "py_vector_new");
    // printf("vec %p\n",vec);
    // printf("%p\n",vec->value.f);
    bridge_manager.bridges[bridge_index].vecreqhost(vec);
    vec->externalData = 1; //to make sure it will not be deallocated

    // for (int i=0; i < vec->len; i++){
    //     printf("%d %lf\n", i, vec->value.f[i]);
    // }
    int rows = vec->len;
    npy_intp dims[1] = {rows};  // Array dimensions
    int data_type = (vec->type == T_FLOAT ? NPY_FLOAT64 : NPY_COMPLEX128);

    PyObject* numpyArray = PyArray_SimpleNewFromData(1, dims, data_type, vec->value.f);

    // double* outputData = (double*)PyArray_DATA((PyArrayObject*)numpyArray);

    // for (int i=0; i < dims[0]; i++) {
    //     printf("%d %f\n",i,outputData[i]);
    // }

    // Make sure to set flags to the NumPy array to manage memory correctly
    PyArray_ENABLEFLAGS((PyArrayObject*)numpyArray, NPY_ARRAY_OWNDATA);
    
    return numpyArray;
}

static PyObject* py_vector_set(PyObject* self, PyObject* args) {
    
    PyObject* pf = NULL;
    int n;
    double real;
    double imag;
    if(!PyArg_ParseTuple(args, "Oidd:py_vector_set", &pf, &n, &real, &imag)) return NULL;

    //printf("print %d\n",n);
    //printf("pf %p\n",pf);
    vector_t * vec = (vector_t *)PyCapsule_GetPointer(pf, "py_vector_new");
    //printf("vec %p\n",vec);
    if (vec->type == T_COMPLEX) {
        vec->value.f[2*n] = real;
        vec->value.f[2*n+1] = imag;
    } else {
        vec->value.f[n] = real;
    }
    //printf("%lf\n",vec->value.f[n]);
    Py_RETURN_NONE;
}

static PyObject* py_vector_get(PyObject* self, PyObject* args) {
    
    PyObject* pf = NULL;
    int n;
    if(!PyArg_ParseTuple(args, "Oi:py_vector_set", &pf, &n)) return NULL;

    // printf("print %d\n",n);
    // printf("pf %p\n",pf);
    vector_t * vec = (vector_t *)PyCapsule_GetPointer(pf, "py_vector_new");
    // printf("vec %p\n",vec);
    // printf("%lf\n",vec->value.f[n]);
    PyObject * result = PyFloat_FromDouble((double)vec->value.f[n]);
    return result;
}

static PyObject* py_move_vector_device(PyObject* self, PyObject* args) {
    
    PyObject* pf = NULL;
    if(!PyArg_ParseTuple(args, "O:py_move_vector_device", &pf)) return NULL;

    //printf("pf %p\n",pf);
    
    vector_t * vec = (vector_t *)PyCapsule_GetPointer(pf, "py_vector_new");
    //printf("vec %p\n",vec);
    bridge_manager.bridges[bridge_index].vecreqdev(vec);
    Py_RETURN_NONE;
}

static PyObject* py_move_vector_host(PyObject* self, PyObject* args) {


    PyObject* pf = NULL;
    if(!PyArg_ParseTuple(args, "O:py_move_vector_host", &pf)) return NULL;

    //printf("pf %p\n",pf);
    
    vector_t * vec = (vector_t *)PyCapsule_GetPointer(pf, "py_vector_new");
    //printf("vec %p\n",vec);
    bridge_manager.bridges[bridge_index].vecreqhost(vec);
    Py_RETURN_NONE;
    //vecreqdev(vec);
    //PyObject* po = PyCapsule_New((void*)vec, "py_vector_new", py_vector_delete);
    //return po;

}
//
static PyObject* py_vec_add(PyObject* self, PyObject* args) {
    
    PyObject* a = NULL;
    PyObject* b = NULL;
    if(!PyArg_ParseTuple(args, "OO:py_vec_add", &a, &b)) return NULL;

    //printf("a %p\n",a);
    //printf("b %p\n",b);
    
    vector_t * vec_a = (vector_t *)PyCapsule_GetPointer(a, "py_vector_new");
        //printf("vec_a %p\n",vec_a);
    vector_t * vec_b = (vector_t *)PyCapsule_GetPointer(b, "py_vector_new");
        //printf("vec_b %p\n",vec_b);

    
    //TODO completar o vec_add
    object_t ** in = convertToObject(vec_a,vec_b);
    
    //printf("vec add to call\n");
    vector_t * r = (vector_t *) vec_add(&bridge_manager, bridge_index, (void **) in, NULL );
    
    //TODO this part returns a numpy_array (not working yet)
    // PyObject* outputArray = PyArray_SimpleNew(2, shape, data_type);
    // printf("%d\n",i++);

    // float* outputData = (float*)PyArray_DATA((PyArrayObject*)outputArray);
    // printf("%d\n",i++);

    // printf("%p %p\n",outputData,r->value.f);
    // for (int i=0; i < r->len; i++) {
    //     printf("%d %f\n",i,r->value.f[i]);
    // }
    // memcpy(outputData, r->value.f, r->len);
    // printf("%d\n",i++);

    PyObject* po = PyCapsule_New((void*)r, "py_vector_new", py_vector_delete);
    return po;
}

static PyObject* py_matvec_mul(PyObject* self, PyObject* args) {
    
    PyObject* a = NULL;
    PyObject* b = NULL;
    if(!PyArg_ParseTuple(args, "OO:py_matvec_mul", &a, &b)) return NULL;

    //printf("a %p\n",a);
    //printf("b %p\n",b);
    
    vector_t * vec_a = (vector_t *)PyCapsule_GetPointer(a, "py_vector_new");
        //printf("vec_a %p\n",vec_a);
    matrix_t * mat_b = (matrix_t *)PyCapsule_GetPointer(b, "py_matrix_new");
        //printf("mat_b %p\n",mat_b);

    
    object_t ** in = convertToObject3(vec_a, mat_b);
    
    vector_t * r = (vector_t *) matvec_mul3(&bridge_manager, bridge_index, (void **) in, NULL );

    PyObject* po = PyCapsule_New((void*)r, "py_vector_new", py_vector_delete);
    return po;
}

static PyObject* py_sparse_matvec_mul(PyObject* self, PyObject* args) {
    
    PyObject* a = NULL;
    PyObject* b = NULL;
    if(!PyArg_ParseTuple(args, "OO:py_sparse_matvec_mul", &a, &b)) return NULL;

    //printf("a %p\n",a);
    //printf("b %p\n",b);
    
    vector_t * vec_a = (vector_t *)PyCapsule_GetPointer(a, "py_vector_new");
//        printf("vec_a %p\n",vec_a);
    smatrix_t * smat_b = (smatrix_t *)PyCapsule_GetPointer(b, "py_sparse_matrix_new");
//        printf("smat_b %p\n",smat_b);

    
    object_t ** in = convertToObject4(vec_a, smat_b);
    
    vector_t * r = (vector_t *) matvec_mul3(&bridge_manager, bridge_index, (void **) in, NULL );

    PyObject* po = PyCapsule_New((void*)r, "py_vector_new", py_vector_delete);
    return po;
}

static PyObject* py_vec_prod(PyObject* self, PyObject* args) {
    
    PyObject* a = NULL;
    PyObject* b = NULL;
    if(!PyArg_ParseTuple(args, "OO:py_vec_add", &a, &b)) return NULL;

    //printf("a %p\n",a);
    //printf("b %p\n",b);
    
    vector_t * vec_a = (vector_t *)PyCapsule_GetPointer(a, "py_vector_new");
        //printf("vec_a %p\n",vec_a);
    vector_t * vec_b = (vector_t *)PyCapsule_GetPointer(b, "py_vector_new");
        //printf("vec_b %p\n",vec_b);

    
    object_t ** in = convertToObject(vec_a,vec_b);
    
    vector_t * r = (vector_t *) vec_prod(&bridge_manager, bridge_index, (void **) in, NULL );
    

    PyObject* po = PyCapsule_New((void*)r, "py_vector_new", py_vector_delete);
    return po;
}

static PyObject* py_vec_add_off(PyObject* self, PyObject* args) {
    
    PyObject* a = NULL;
    int offset;
    if(!PyArg_ParseTuple(args, "iO:py_vec_add_off", &offset, &a)) return NULL;

    vector_t * vec_a = (vector_t *)PyCapsule_GetPointer(a, "py_vector_new");

    object_t ** in = convertToObject2(offset, vec_a);
    
    vector_t * r = (vector_t *) vec_add_off(&bridge_manager, bridge_index, (void **) in, NULL );   

    PyObject* po = PyCapsule_New((void*)r, "py_vector_new", py_vector_delete);
    return po;
}

static PyObject* py_vec_sum(PyObject* self, PyObject* args) {
    
    PyObject* a = NULL;
    if(!PyArg_ParseTuple(args, "O:py_vec_sum", &a)) return NULL;

    vector_t * vec_a = (vector_t *)PyCapsule_GetPointer(a, "py_vector_new");
    
    object_t ** in = convertToObject(vec_a, NULL);
    
    object_t * r = (object_t *) vec_sum(&bridge_manager, bridge_index, (void **) in, NULL );

    PyObject * result = PyFloat_FromDouble((double)r->value.f);
    
    return result;
}

static PyObject* py_vec_conj(PyObject* self, PyObject* args) {
    
    PyObject* a = NULL;
    if(!PyArg_ParseTuple(args, "O:py_vec_sum", &a)) return NULL;

    vector_t * vec_a = (vector_t *)PyCapsule_GetPointer(a, "py_vector_new");
    
    object_t ** in = convertToObject(vec_a, NULL);
    
    vector_t * r = (vector_t *) vec_conj(&bridge_manager, bridge_index, (void **) in, NULL );

    PyObject* po = PyCapsule_New((void*)r, "py_vector_new", py_vector_delete);
    return po;
}

//

static void py_matrix_delete(PyObject* self) {
    matrix_t* mat = (matrix_t*)PyCapsule_GetPointer(self, "py_matrix_new");
    //printf("mat %p\n",mat);
    //printf("mat->value %p\n",&(mat->value));
    //free ((void *)mat->value.f);
    //free ((void *)mat);
    bridge_manager.bridges[bridge_index].matrix_delete(mat);
}


static PyObject* py_matrix_new(PyObject* self, PyObject* args){
    int rows;
    int cols;
    int data_type;
    if (!PyArg_ParseTuple(args, "iii", &rows,&cols,&data_type)) return NULL;
    //printf("create %d\n",rows);
    //printf("create %d\n",cols);
    matrix_t * a = bridge_manager.bridges[bridge_index].matrix_new(rows, cols, data_type, 1);
    //printf("malloc %p\n",a);
    PyObject* po = PyCapsule_New((void*)a, "py_matrix_new", py_matrix_delete);
    //printf("capsule_new %p\n",po);
    return po;
}

static PyObject* py_matrix_set(PyObject* self, PyObject* args) {
    
    PyObject* pf = NULL;
    int i;
    int j;
    double real;
    double imag;
    if(!PyArg_ParseTuple(args, "Oiidd:py_matrix_set", &pf, &i, &j, &real, &imag)) return NULL;

    //printf("print (%d,%d)\n",i,j);
    //printf("pf %p\n",pf);
    matrix_t * mat = (matrix_t *)PyCapsule_GetPointer(pf, "py_matrix_new");
    //printf("mat %p\n",mat);
    int idx = (i*mat->ncol + j);
    if (mat->type == T_COMPLEX) {
        mat->value.f[2 * idx] = real;
        mat->value.f[2 * idx + 1] = imag;
    } else {
        mat->value.f[idx] = real;
    }
    //printf("%lf\n",mat->value.f[i*mat->ncol + j]);
    Py_RETURN_NONE;
}

static PyObject* py_matrix_get(PyObject* self, PyObject* args) {
    
    PyObject* pf = NULL;
    int i;
    int j;
    if(!PyArg_ParseTuple(args, "Oii:py_matrix_get", &pf, &i, &j)) return NULL;

    //printf("print (%d,%d)\n",i,j);
    //printf("pf %p\n",pf);
    matrix_t * mat = (matrix_t *)PyCapsule_GetPointer(pf, "py_matrix_new");
    //printf("mat %p\n",mat);
    //printf("%lf\n",mat->value.f[i*mat->ncol + j]);
    PyObject * result = PyFloat_FromDouble((double)mat->value.f[i*mat->ncol + j]);
    return result;
}


static PyObject* py_move_matrix_device(PyObject* self, PyObject* args) {
    
    PyObject* pf = NULL;
    if(!PyArg_ParseTuple(args, "O:py_move_matrix_device", &pf)) return NULL;

    //printf("pf %p\n",pf);
    
    matrix_t * mat = (matrix_t *)PyCapsule_GetPointer(pf, "py_matrix_new");
    //printf("mat %p\n",mat);
    bridge_manager.bridges[bridge_index].matreqdev(mat);
    Py_RETURN_NONE;
}

static PyObject* py_move_matrix_host(PyObject* self, PyObject* args) {
    
    PyObject* pf = NULL;
    if(!PyArg_ParseTuple(args, "O:py_move_matrix_device", &pf)) return NULL;

    //printf("pf %p\n",pf);
    
    matrix_t * mat = (matrix_t *)PyCapsule_GetPointer(pf, "py_matrix_new");
    bridge_manager.bridges[bridge_index].matreqhost(mat); //should use this function? It seems that it creates the object in the stack
    //    //printf("mat %p\n",mat);
//    int n = (mat->type==T_FLOAT?mat->nrow*mat->ncol:2*mat->nrow*mat->ncol);
//    matrix_t * out = matrix_new(mat->nrow, mat->ncol, mat->type);
//    cl_int status = clEnqueueReadBuffer(clinfo.q, mat->extra, CL_TRUE, 0, n * sizeof (double), out->value.f, 0, NULL, NULL);
//    CLERR
//    PyObject* po = PyCapsule_New((void*)out, "py_matrix_new", py_matrix_delete);
//    return po;
    Py_RETURN_NONE;
}

static void py_sparse_matrix_delete(PyObject* self) {
    smatrix_t* mat = (smatrix_t*)PyCapsule_GetPointer(self, "py_sparse_matrix_new");
    //printf("smat %p\n",mat);
    //free ((void *)mat->m);
    //free ((void *)mat);
    bridge_manager.bridges[bridge_index].smatrix_delete(mat);
}


static PyObject* py_sparse_matrix_new(PyObject* self, PyObject* args){
    int rows;
    int cols;
    int data_type;
    if (!PyArg_ParseTuple(args, "iii", &rows,&cols,&data_type)) return NULL;
    //printf("create %d\n",rows);
    //printf("create %d\n",cols);
    smatrix_t * a = bridge_manager.bridges[bridge_index].smatrix_new(rows, cols, data_type);
    //printf("malloc %p\n",a);
    PyObject* po = PyCapsule_New((void*)a, "py_sparse_matrix_new", py_sparse_matrix_delete);
    //printf("capsule_new %p\n",po);
    return po;
}

static PyObject* py_sparse_matrix_set(PyObject* self, PyObject* args) {
    
    PyObject* pf = NULL;
    int i;
    int j;
    double real;
    double imag;
    if(!PyArg_ParseTuple(args, "Oiidd:py_sparse_matrix_set", &pf, &i, &j, &real, &imag)) return NULL;

    //printf("print (%d,%d)\n",i,j);
    //printf("pf %p\n",pf);
    smatrix_t * mat = (smatrix_t *)PyCapsule_GetPointer(pf, "py_sparse_matrix_new");
    //printf("smat %p\n",mat);
    if(mat->type == T_COMPLEX) {
        bridge_manager.bridges[bridge_index].smatrix_set_complex_value(mat,i,j,real, imag);
    } else if(mat->type == T_FLOAT) {
        bridge_manager.bridges[bridge_index].smatrix_set_real_value(mat,i,j,real);
    }
    Py_RETURN_NONE;
}

static PyObject* py_sparse_matrix_pack(PyObject* self, PyObject* args) {
    
    PyObject* pf = NULL;
    if(!PyArg_ParseTuple(args, "O:py_sparse_matrix_set", &pf)) return NULL;

    //printf("pf %p\n",pf);
    smatrix_t * mat = (smatrix_t *)PyCapsule_GetPointer(pf, "py_sparse_matrix_new");
    if (mat->type == T_FLOAT) {
        bridge_manager.bridges[bridge_index].smatrix_pack(mat);
    } else {
        bridge_manager.bridges[bridge_index].smatrix_pack_complex(mat);
    }
    Py_RETURN_NONE;
}

static PyObject* py_move_sparse_matrix_device(PyObject* self, PyObject* args) {
    
    PyObject* pf = NULL;
    if(!PyArg_ParseTuple(args, "O:py_move_sparse_matrix_device", &pf)) return NULL;

    smatrix_t * smat = (smatrix_t *)PyCapsule_GetPointer(pf, "py_sparse_matrix_new");
    bridge_manager.bridges[bridge_index].smatreqdev(smat);
    Py_RETURN_NONE;
}

static PyObject* py_move_sparse_matrix_host(PyObject* self, PyObject* args) {
    
    PyObject* pf = NULL;
    if(!PyArg_ParseTuple(args, "O:py_move_sparse_matrix_device", &pf)) return NULL;

    smatrix_t * smat = (smatrix_t *)PyCapsule_GetPointer(pf, "py_sparse_matrix_new");
    bridge_manager.bridges[bridge_index].smatreqhost(smat); //should use this function? It seems that it creates the object in the stack
    //printf("mat %p\n",mat);
//    int n = (mat->type==T_FLOAT?mat->nrow*mat->ncol:2*mat->nrow*mat->ncol);
//    matrix_t * out = matrix_new(mat->nrow, mat->ncol, mat->type);
//    cl_int status = clEnqueueReadBuffer(clinfo.q, mat->mem, CL_TRUE, 0, n * sizeof (double), out->value.f, 0, NULL, NULL);
//    CLERR
//    PyObject* po = PyCapsule_New((void*)smat, "py_sparse_matrix_new", py_sparse_matrix_delete);
//    return po;
    Py_RETURN_NONE;
}

static PyObject* py_mat_add(PyObject* self, PyObject* args) {
    
    PyObject* a = NULL;
    PyObject* b = NULL;
    if(!PyArg_ParseTuple(args, "OO:py_mat_add", &a, &b)) return NULL;

    //printf("a %p\n",a);
    //printf("b %p\n",b);
    
    matrix_t * mat_a = (matrix_t *)PyCapsule_GetPointer(a, "py_matrix_new");
        //printf("vec_a %p\n",vec_a);
    matrix_t * mat_b = (matrix_t *)PyCapsule_GetPointer(b, "py_matrix_new");
        //printf("vec_b %p\n",vec_b);

    
    //TODO completar o vec_add
    object_t ** in = convertMatMatToObject(mat_a,mat_b);
    
    matrix_t * r = (matrix_t *) mat_add(&bridge_manager, bridge_index, (void **) in, NULL );
    

    PyObject* po = PyCapsule_New((void*)r, "py_matrix_new", py_matrix_delete);
    return po;
}

static PyObject* py_mat_mul(PyObject* self, PyObject* args) {
    
    PyObject* a = NULL;
    PyObject* b = NULL;
    if(!PyArg_ParseTuple(args, "OO:py_mat_mul", &a, &b)) return NULL;

    //printf("a %p\n",a);
    //printf("b %p\n",b);
    
    matrix_t * mat_a = (matrix_t *)PyCapsule_GetPointer(a, "py_matrix_new");
        //printf("vec_a %p\n",vec_a);
    matrix_t * mat_b = (matrix_t *)PyCapsule_GetPointer(b, "py_matrix_new");
        //printf("vec_b %p\n",vec_b);

    
    //TODO completar o vec_add
    object_t ** in = convertMatMatToObject(mat_a,mat_b);
    
    matrix_t * r = (matrix_t *) mat_mul(&bridge_manager, bridge_index, (void **) in, NULL );
    

    PyObject* po = PyCapsule_New((void*)r, "py_matrix_new", py_matrix_delete);
    return po;
}

static PyObject* py_scalar_mat_mul(PyObject* self, PyObject* args) {
    
    PyObject* a = NULL;
    double scalar;
    if(!PyArg_ParseTuple(args, "dO:py_scalar_mat_mul", &scalar,&a)) return NULL;

    matrix_t * mat_a = (matrix_t *)PyCapsule_GetPointer(a, "py_matrix_new");
    
    object_t ** in = convertScaMatToObject(scalar, mat_a);
    
    matrix_t * r = (matrix_t *) mat_mulsc(&bridge_manager, bridge_index, (void **) in, NULL );

    PyObject* po = PyCapsule_New((void*)r, "py_matrix_new", py_matrix_delete);
    return po;
}

static PyObject* py_scalar_vec_mul(PyObject* self, PyObject* args) {
    
    PyObject* a = NULL;
    double scalar;
    if(!PyArg_ParseTuple(args, "dO:py_scalar_vec_mul", &scalar,&a)) return NULL;

    //printf("a %p\n",a);
    vector_t * vec_a = (vector_t *)PyCapsule_GetPointer(a, "py_vector_new");
    //printf("vec_a %p\n",vec_a);
    
    object_t ** in = convertScaVecToObject(scalar, vec_a);
    
    vector_t * r = (vector_t *) vec_mulsc(&bridge_manager, bridge_index, (void **) in, NULL );
    //    printf("r %p\n",r);
    PyObject* po = PyCapsule_New((void*)r, "py_vector_new", py_vector_delete);
    return po;
}

static PyObject* py_complex_scalar_vec_mul(PyObject* self, PyObject* args) {
    
    PyObject* scalar = NULL;
    PyObject* a = NULL;
    if(!PyArg_ParseTuple(args, "OO:py_complex_scalar_vec_mul", &scalar,&a)) return NULL;

    complex_t * complex_scalar = (complex_t *)PyCapsule_GetPointer(scalar, "py_complex_new");
    vector_t * vec_a = (vector_t *)PyCapsule_GetPointer(a, "py_vector_new");
    
    vector_t * r = NULL;
    if (vec_a->type == T_FLOAT) {
        r = (vector_t *) vec_mul_complex_scalar (&bridge_manager, bridge_index,  complex_scalar, vec_a); 
    } else if (vec_a->type == T_COMPLEX) {
        r = (vector_t *) mul_complex_scalar_complex_vec(&bridge_manager, bridge_index,  complex_scalar, vec_a);
    }

    PyObject* po = PyCapsule_New((void*)r, "py_vector_new", py_vector_delete);
    return po;
}

static PyObject* py_complex_scalar_mat_mul(PyObject* self, PyObject* args) {
    
    PyObject* scalar = NULL;
    PyObject* a = NULL;
    if(!PyArg_ParseTuple(args, "OO:py_complex_scalar_mat_mul", &scalar,&a)) return NULL;

    complex_t * complex_scalar = (complex_t *)PyCapsule_GetPointer(scalar, "py_complex_new");
    matrix_t * mat_a = (matrix_t *)PyCapsule_GetPointer(a, "py_matrix_new");
    
    matrix_t * r = NULL;
    if (mat_a->type == T_FLOAT) {
        r = (matrix_t *) mul_complex_scalar_float_mat (&bridge_manager, bridge_index,  complex_scalar, mat_a); 
    } else if (mat_a->type == T_COMPLEX) {
        r = (matrix_t *) mul_complex_scalar_complex_mat(&bridge_manager, bridge_index,  complex_scalar, mat_a);
    }

    PyObject* po = PyCapsule_New((void*)r, "py_matrix_new", py_matrix_delete);
    return po;
}

static PyObject* cpu_constant;
static PyObject* gpu_constant;
static PyObject* float_constant;
static PyObject* complex_constant;

static PyObject* get_cpu_constant(PyObject* self, PyObject* args)
{
    Py_INCREF(cpu_constant);
    return cpu_constant;
}

static PyObject* get_gpu_constant(PyObject* self, PyObject* args)
{
    Py_INCREF(gpu_constant);
    return gpu_constant;
}

static PyObject* get_float_constant(PyObject* self, PyObject* args)
{
    Py_INCREF(float_constant);
    return float_constant;
}

static PyObject* get_complex_constant(PyObject* self, PyObject* args)
{
    Py_INCREF(complex_constant);
    return complex_constant;
}

static PyMethodDef mainMethods[] = {
    {"init_engine", py_init_engine, METH_VARARGS, "init_engine"},
    {"stop_engine", py_stop_engine, METH_VARARGS, "stop_engine"},
    {"vector_new", py_vector_new, METH_VARARGS, "vector_new"},
    {"load_numpy_array", py_load_numpy_array, METH_VARARGS, "load_numpy_array"},
    {"retrieve_numpy_array", py_retrieve_numpy_array, METH_VARARGS, "retrieve_numpy_array"},
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
    {"sparse_matrix_set", py_sparse_matrix_set, METH_VARARGS, "sparse_matrix_set"},
    {"sparse_matrix_pack", py_sparse_matrix_pack, METH_VARARGS, "sparse_matrix_pack"},
    {"move_sparse_matrix_device", py_move_sparse_matrix_device, METH_VARARGS, "move_sparse_matrix_device"},
    {"move_sparse_matrix_host", py_move_sparse_matrix_host, METH_VARARGS, "move_sparse_matrix_host"},
    {"vec_add", py_vec_add, METH_VARARGS, "vec_add"},
    {"matvec_mul", py_matvec_mul, METH_VARARGS, "matvec_mul"},
    {"sparse_matvec_mul", py_sparse_matvec_mul, METH_VARARGS, "sparse_matvec_mul"},
    {"vec_prod", py_vec_prod, METH_VARARGS, "vec_prod"},
    {"vec_add_off", py_vec_add_off, METH_VARARGS, "vec_add_off"},
    {"vec_sum", py_vec_sum, METH_VARARGS, "vec_sum"},
    {"vec_conj", py_vec_conj, METH_VARARGS, "vec_conj"},
    {"mat_add", py_mat_add, METH_VARARGS, "mat_add"},
    {"mat_mul", py_mat_mul, METH_VARARGS, "mat_mul"},
    {"scalar_mat_mul", py_scalar_mat_mul, METH_VARARGS, "scalar_mat_mul"},
    {"scalar_vec_mul", py_scalar_vec_mul, METH_VARARGS, "scalar_vec_mul"},
    {"complex_scalar_vec_mul", py_complex_scalar_vec_mul, METH_VARARGS, "complex_scalar_vec_mul"},
    {"complex_scalar_mat_mul", py_complex_scalar_mat_mul, METH_VARARGS, "complex_scalar_mat_mul"},
    {"complex_new", py_complex_new, METH_VARARGS, "complex_new"},
    {"get_cpu_constant", get_cpu_constant, METH_NOARGS, "Get the CPU constant value."},
    {"get_gpu_constant", get_gpu_constant, METH_NOARGS, "Get the GPU constant value."},
    {"get_float_constant", get_float_constant, METH_NOARGS, "Get the FLOAT constant value."},
    {"get_complex_constant", get_complex_constant, METH_NOARGS, "Get COMPLEX the constant value."},
    {NULL, NULL, 0, NULL}
};

static PyModuleDef neblina = {
    PyModuleDef_HEAD_INIT,
    "neblina", "Neblina Core",
    -1,
    mainMethods
};

PyMODINIT_FUNC PyInit_neblina(void) {
    PyObject* module =  PyModule_Create(&neblina);
    cpu_constant = PyLong_FromLong(0);
    gpu_constant = PyLong_FromLong(1);
    float_constant = PyLong_FromLong(2);
    complex_constant = PyLong_FromLong(3);
    
    PyModule_AddObject(module, "CPU", cpu_constant);
    PyModule_AddObject(module, "GPU", gpu_constant);
    PyModule_AddObject(module, "FLOAT", float_constant);
    PyModule_AddObject(module, "COMPLEX", complex_constant);

    import_array();
    return module;
}


