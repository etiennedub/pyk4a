#include <Python.h>

#include <k4a/k4a.h>
#include <stdio.h>

static PyObject* hello_world(PyObject* self, PyObject* args)
{
    uint32_t device_count = 0;
    k4a_device_t device = NULL;
    k4a_device_configuration_t config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;

    device_count = k4a_device_get_installed_count();

    if (device_count == 0)
    {
        printf("No K4A devices found\n");
        return Py_BuildValue("I", 1);
    }

    if (K4A_RESULT_SUCCEEDED != k4a_device_open(K4A_DEVICE_DEFAULT, &device))
    {
        printf("Failed to open device\n");
        return Py_BuildValue("I", 1);
    }

    return Py_BuildValue("I", 0);
}

// Source : https://github.com/MathGaron/pyvicon/blob/master/pyvicon/pyvicon.cpp
//###################
//Module initialisation
//###################

struct module_state
{
    PyObject *error;
};

#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))


//#####################
// Methods
//#####################
static PyMethodDef Pyk4aMethods[] = {
     {"hello_world", hello_world, METH_VARARGS, "Hello world"},
     {NULL, NULL, 0, NULL}
};

static int pyk4a_traverse(PyObject *m, visitproc visit, void *arg)
{
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int pyk4a_clear(PyObject *m)
{
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static struct PyModuleDef moduledef =
{
    PyModuleDef_HEAD_INIT,
    "k4a_module",
    NULL,
    sizeof(struct module_state),
    Pyk4aMethods,
    NULL,
    pyk4a_traverse,
    pyk4a_clear,
    NULL
};
#define INITERROR return NULL


//########################
// Module init function
//########################
PyMODINIT_FUNC PyInit_k4a_module(void) {
    PyObject *module = PyModule_Create(&moduledef);

    if (module == NULL)
    INITERROR;
    struct module_state *st = GETSTATE(module);

    st->error = PyErr_NewException("pyvicon_module.Error", NULL, NULL);
    if (st->error == NULL)
    {
        Py_DECREF(module);
        INITERROR;
    }
return module;
}
