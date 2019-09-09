#include <Python.h>
#include <numpy/arrayobject.h>

#include <k4a/k4a.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

    k4a_capture_t capture;

    static PyObject* device_open(PyObject* self, PyObject* args){
        int device_id;
        PyArg_ParseTuple(args, "I", &device_id);
        k4a_device_t device = NULL;
        k4a_result_t result = k4a_device_open(device_id, &device);

        return Py_BuildValue("In", result, device);
    }

    static PyObject* device_close(PyObject* self, PyObject* args){
        k4a_device_t device = NULL;
        PyArg_ParseTuple(args, "n", &device);

        /* k4a_device_close(device); */
        return NULL;
    }

    static PyObject* device_get_sync_jack(PyObject* self, PyObject* args){
        k4a_device_t device = NULL;
        PyArg_ParseTuple(args, "n", &device);

        bool in_jack = 0;
        bool out_jack = 0;
        k4a_result_t result = k4a_device_get_sync_jack(device, &in_jack, &out_jack);

        return Py_BuildValue("III", result, in_jack, out_jack);
    }

    static PyObject* device_start_cameras(PyObject* self, PyObject* args){
        k4a_device_t device = NULL;
        k4a_device_configuration_t config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
        PyArg_ParseTuple(args, "nIIIIIIIII", &device, &config.color_format,
                &config.color_resolution, &config.depth_mode,
                &config.camera_fps, &config.synchronized_images_only,
                &config.depth_delay_off_color_usec, &config.wired_sync_mode,
                &config.subordinate_delay_off_master_usec,
                &config.disable_streaming_indicator);

        k4a_result_t result = k4a_device_start_cameras(device, &config);
        return Py_BuildValue("I", result);
    }

    static PyObject* device_get_capture(PyObject* self, PyObject* args){
        k4a_device_t device = NULL;
        int32_t timeout;
        PyArg_ParseTuple(args, "nI", &device, &timeout);

        k4a_capture_create(&capture);
        k4a_wait_result_t result = k4a_device_get_capture(device, &capture, timeout);
        return Py_BuildValue("I", result);
    }

    static PyObject* device_get_color_image(PyObject* self, PyObject* args){
        k4a_image_t color_image = k4a_capture_get_color_image(capture);
        uint8_t* buffer = k4a_image_get_buffer(color_image);
        unsigned long dims[2];
        dims[0] = (unsigned long) k4a_image_get_height_pixels(color_image);
        dims[1] = (unsigned long) k4a_image_get_width_pixels(color_image);
        dims[2] = (unsigned long) 4;
        PyArrayObject* np_color_image = (PyArrayObject*) PyArray_SimpleNewFromData(3, (npy_intp*) dims, NPY_UINT8, buffer);
        return PyArray_Return(np_color_image);
    }

    static PyObject* capture_release(PyObject* self, PyObject* args){
        k4a_capture_release(capture);
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
        {"device_open", device_open, METH_VARARGS, "Open an Azure Kinect device"},
        {"device_start_cameras", device_start_cameras, METH_VARARGS, "Starts color and depth camera capture"},
        {"device_get_capture", device_get_capture, METH_VARARGS, "Reads a sensor capture"},
        {"capture_release", capture_release, METH_VARARGS, "Release a capture"},
        {"device_get_color_image", device_get_color_image, METH_VARARGS, "Get the color image associated with the given capture"},
        {"device_close", device_close, METH_VARARGS, "Close an Azure Kinect device"},
        {"device_get_sync_jack", device_get_sync_jack, METH_VARARGS, "Get the device jack status for the synchronization in and synchronization out connectors."},
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
        import_array();
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

#ifdef __cplusplus
}
#endif
