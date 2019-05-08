//#include <Python.h>
//#include <stdio.h>
//#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
//#include "/home/zhen/anaconda3/envs/py27/lib/python2.7/site-packages/numpy/core/include/numpy/arrayobject.h"

#include <iostream>
#include <stdio.h>
#include <fcntl.h>

#include "ClebschGordan.hpp"

#ifndef MIN 
#define MIN(X,Y) ((X) < (Y) ? (X) : (Y)) 
#endif

#ifndef MAX 
#define MAX(X,Y) ((X) > (Y) ? (X) : (Y)) 
#endif

//using namespace std;

#include "Python.h"
#include "numpy/arrayobject.h"

//NOT used
static PyObject*
precompute_half (PyObject *dummy, PyObject *args)
{
    int L = 0;
    PyArrayObject * Cmat = NULL;
    if (!PyArg_ParseTuple(args, "i", &L)) return NULL;
    std::cout << "L is " << L << std::endl;
    PyObject* ret = PyDict_New();

    ClebschGordan CG(L,0);

    for (int l1 = 0; l1 <= L; l1++){
        for (int l2 = 0; l2 <= L; l2++){
            for (int l = 0; l <= L; l++){
                if (l2 <= l1 && l >= l1 - l2 && l <= MIN(l1+l2,L)){
                    long idx = l + (L+1) * (l2 + (L+1)*l1);
                    std::cout<<idx<<std::endl;
                    
                    npy_intp dims[] = {(2*l+1), (2*l1+1)*(2*l2+1)};
                    Cmat = (PyArrayObject*) PyArray_SimpleNew(2, dims, NPY_DOUBLE);

                    //actually working on the CG matrix
                    for (int m = -l; m <= l; ++m) {
                        for (int m1 = -l1; m1 <= l1; ++m1){
                            for (int m2 = -l2; m2 <= l2; ++m2){
                                int i = m + l;
                                int j = (m1+l1) * (2*l2 + 1) + (m2+l2);
                                double *v = (double*) PyArray_GETPTR2(Cmat, i, j);
                                *v = CG(l1,l2,l,m1,m2,m);
                            }
                        }
                    }

                    PyDict_SetItem(ret, PyLong_FromLong(idx), (PyObject*) Cmat);
                } else {
                    continue;
                }
            }
        }
    }
    std::cout << "Safely exitting "<< std::endl;
    return ret;
}

static PyObject*
precompute (PyObject *dummy, PyObject *args)
{
    int L = 0;
    PyArrayObject * Cmat = NULL;
    if (!PyArg_ParseTuple(args, "i", &L)) return NULL;
    std::cout << "L is " << L << std::endl;
    PyObject* ret = PyDict_New();

    ClebschGordan CG(L,0);

    for (int l1 = 0; l1 <= L; l1++){
        for (int l2 = 0; l2 <= L; l2++){
            for (int l = 0; l <= L; l++){
                if (l >= MAX(l1-l2,l2-l1) && l <= MIN(l1+l2,L)){
                    long idx = l + (L+1) * (l2 + (L+1)*l1);
                    //std::cout << l1 << " " << l2 <<  " " << l <<  " idx: " << idx <<std::endl;
                    npy_intp dims[] = {(2*l+1), (2*l1+1)*(2*l2+1)};
                    Cmat = (PyArrayObject*) PyArray_SimpleNew(2, dims, NPY_DOUBLE);

                    for (int m = -l; m <= l; ++m) {
                        for (int m1 = -l1; m1 <= l1; ++m1){
                            for (int m2 = -l2; m2 <= l2; ++m2){
                                //std::cout <<"m "<<m<<" m1 "<<m1<<" m2 "<<m2<<std::endl;
                                int i = m + l;
                                //std::cout<<"here"<<std::endl;
                                int j = (m1+l1) * (2*l2 + 1) + (m2+l2);
                                //std::cout << "i: " << i << " j: " << j;
                                double *v = (double*) PyArray_GETPTR2(Cmat, i, j);
                                *v = CG(l1,l2,l,m1,m2,m);
                                //if (*v && m == 0 && l1 == 2 && l2 == 2) {
                                //    //printf("v:%f - l:%d m:%d m1:%d m2:%d\n", *v, l, l1, l2, m, m1, m2);
                                //    printf("v:%f - l:%d m:%d m1:%d m2:%d\n", *v, l, m, m1, m2);
                                //}
                                //std::cout << " ..good" << std::endl;;
                            }
                        }
                    }
                    //std::cout << "Setting dictionary " <<std::endl;
                    PyDict_SetItem(ret, PyLong_FromLong(idx), (PyObject*) Cmat);
                } else {
                    continue;
                }
                
            }
        }
    }
    std::cout << "Safely exitting "<< std::endl;
    return ret;
}

static PyObject*
precompute_flat (PyObject *dummy, PyObject *args)
{
    int L = 0;
    PyArrayObject * Cmat = NULL;
    if (!PyArg_ParseTuple(args, "i", &L)) return NULL;
    std::cout << "L is " << L << std::endl;
    PyObject* ret = PyDict_New();

    ClebschGordan CG(L,0);

    int total_cnt = 0;
    for (int l1 = 0; l1 <= L; l1++){
        for (int l2 = 0; l2 <= l1; l2++){
            for (int l = l1 - l2; l <= L && l <= l1 + l2; l++){
                total_cnt += (2 * l1 + 1) * (2 * l2 + 1);
            }
        }
    }
    //We need a long array of 
    for (int l1 = 0; l1 <= L; l1++){
        for (int l2 = 0; l2 <= L; l2++){
            for (int l = 0; l <= L; l++){
                if (l >= MAX(l1-l2,l2-l1) && l <= MIN(l1+l2,L)){
                    long idx = l + (L+1) * (l2 + (L+1)*l1);
                    //std::cout << l1 << " " << l2 <<  " " << l <<  " idx: " << idx <<std::endl;
                    npy_intp dims[] = {(2*l+1), (2*l1+1)*(2*l2+1)};
                    Cmat = (PyArrayObject*) PyArray_SimpleNew(2, dims, NPY_DOUBLE);

                    for (int m = -l; m <= l; ++m) {
                        for (int m1 = -l1; m1 <= l1; ++m1){
                            for (int m2 = -l2; m2 <= l2; ++m2){
                                //std::cout <<"m "<<m<<" m1 "<<m1<<" m2 "<<m2<<std::endl;
                                int i = m + l;
                                //std::cout<<"here"<<std::endl;
                                int j = (m1+l1) * (2*l2 + 1) + (m2+l2);
                                //std::cout << "i: " << i << " j: " << j;
                                double *v = (double*) PyArray_GETPTR2(Cmat, i, j);
                                *v = CG(l1,l2,l,m1,m2,m);
                                //if (*v && m == 0 && l1 == 2 && l2 == 2) {
                                //    //printf("v:%f - l:%d m:%d m1:%d m2:%d\n", *v, l, l1, l2, m, m1, m2);
                                //    printf("v:%f - l:%d m:%d m1:%d m2:%d\n", *v, l, m, m1, m2);
                                //}
                                //std::cout << " ..good" << std::endl;;
                            }
                        }
                    }
                    //std::cout << "Setting dictionary " <<std::endl;
                    PyDict_SetItem(ret, PyLong_FromLong(idx), (PyObject*) Cmat);
                } else {
                    continue;
                }
                
            }
        }
    }
    std::cout << "Safely exitting "<< std::endl;
    return ret;
}
static struct PyMethodDef methods[] = {
    {"precompute", precompute, METH_VARARGS, "precompute: input L, precompute dictionary of ndarrays"},
    {NULL, NULL, 0, NULL}
};

//Python 2:
/*
PyMODINIT_FUNC initClebschGordan (void)
{
    (void)Py_InitModule("ClebschGordan", methods);
    import_array();
}

PyMODINIT_FUNC initExample(void)
{
    (void) Py_InitModule("Example", ExampleMethods);
}
*/
//Python 3
static struct PyModuleDef ClabschGordan =
{
    PyModuleDef_HEAD_INIT,
    "ClabschGordan", /* name of module */
    "usage: .....)\n", /* module documentation, may be NULL */
    -1,   /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    methods
};

PyMODINIT_FUNC PyInit_ClebschGordan(void)
{
    import_array();
    return PyModule_Create(&ClabschGordan);
}
