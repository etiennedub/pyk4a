#include <Python.h>
#include <numpy/arrayobject.h>

#include <k4a/k4a.h>
#ifdef ENABLE_BODY_TRACKING
    #message "Body tracking is enabled."
    #include <k4abt.h>
#endif
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif
    k4a_capture_t capture;
    k4a_transformation_t transformation_handle;
    k4a_device_t device;
    k4a_calibration_t calibration;
#ifdef
    k4abt_tracker_t tracker;
    #define NUM_JOINTS = 32;
    #define NUM_DATA = 10;
#endif

    static PyObject* device_open(PyObject* self, PyObject* args){
        int device_id;
        PyArg_ParseTuple(args, "I", &device_id);
        k4a_result_t result = k4a_device_open(device_id, &device);

        return Py_BuildValue("I", result);
    }

    static PyObject* device_close(PyObject* self, PyObject* args){
        k4a_device_close(device);
        return Py_BuildValue("I", K4A_RESULT_SUCCEEDED);
    }

    static PyObject* device_get_sync_jack(PyObject* self, PyObject* args){
        bool in_jack = 0;
        bool out_jack = 0;
        k4a_result_t result = k4a_device_get_sync_jack(device, &in_jack, &out_jack);

        return Py_BuildValue("III", result, in_jack, out_jack);
    }

    static PyObject* device_get_color_control(PyObject* self, PyObject* args){
        k4a_color_control_command_t command;
        k4a_color_control_mode_t mode;
        int32_t value = 0;
        PyArg_ParseTuple(args, "I", &command);

        k4a_result_t result = k4a_device_get_color_control(device, command, &mode, &value);
        if (result == K4A_RESULT_FAILED) {
            return Py_BuildValue("III", 0, 0, 0);
        }
        return Py_BuildValue("III", result, mode, value);
    }

    static PyObject* device_set_color_control(PyObject* self, PyObject* args){
        k4a_color_control_command_t command = K4A_COLOR_CONTROL_EXPOSURE_TIME_ABSOLUTE;
        k4a_color_control_mode_t mode = K4A_COLOR_CONTROL_MODE_MANUAL;
        int32_t value = 0;
        PyArg_ParseTuple(args, "III", &command, &mode, &value);

        k4a_result_t result = k4a_device_set_color_control(device, command, mode, value);
        if (result == K4A_RESULT_FAILED) {
            return Py_BuildValue("I", K4A_RESULT_FAILED);
        }
        return Py_BuildValue("I", result);
    }

    static PyObject* device_get_color_control_capabilities(PyObject* self, PyObject* args){
        k4a_color_control_command_t command;
        bool supports_auto;
        int32_t min_value;
        int32_t max_value;
        int32_t step_value;
        int32_t default_value;
        k4a_color_control_mode_t default_mode;
        PyArg_ParseTuple(args, "I", &command);


        k4a_result_t result = k4a_device_get_color_control_capabilities(device, command, &supports_auto, &min_value, &max_value, &step_value, &default_value, &default_mode);
        if (result == K4A_RESULT_FAILED) {
            return Py_BuildValue("IIIIIII", 0, 0, 0, 0, 0, 0, 0);
        }
        return Py_BuildValue("IIIIIII", result, supports_auto, min_value, max_value, step_value, default_value, default_mode);
    }

    static PyObject* device_start_cameras(PyObject* self, PyObject* args){
        k4a_device_configuration_t config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
        PyArg_ParseTuple(args, "IIIIIIIII", &config.color_format,
                &config.color_resolution, &config.depth_mode,
                &config.camera_fps, &config.synchronized_images_only,
                &config.depth_delay_off_color_usec, &config.wired_sync_mode,
                &config.subordinate_delay_off_master_usec,
                &config.disable_streaming_indicator);

        k4a_result_t result;
        result = k4a_device_get_calibration(device, config.depth_mode, config.color_resolution, &calibration);
        if (result == K4A_RESULT_FAILED) {
            return Py_BuildValue("I", K4A_RESULT_FAILED);
        }

        transformation_handle = k4a_transformation_create(&calibration);
        if (transformation_handle == NULL) {
            return Py_BuildValue("I", K4A_RESULT_FAILED);
        }

        result = k4a_device_start_cameras(device, &config);
        if (result == K4A_RESULT_FAILED) {
            return Py_BuildValue("I", K4A_RESULT_FAILED);
        }

#ifdef ENABLE_BODY_TRACKING
        k4abt_tracker_configuration_t tracker_calibration = K4ABT_TRACKER_CONFIG_DEFAULT;
        result = k4abt_tracker_create(&calibration, tracker_calibration, &tracker);
        if (result == K4A_RESULT_FAILED) {
            return Py_BuildValue("I", K4A_RESULT_FAILED);
        }
#endif
        return Py_BuildValue("I", result);
    }

    static PyObject* device_stop_cameras(PyObject* self, PyObject* args){
        if (transformation_handle) k4a_transformation_destroy(transformation_handle);
        if (capture) k4a_capture_release(capture);
#ifdef ENABLE_BODY_TRACKING
        if (tracker) k4abt_tracker_destroy(tracker);
#endif
        k4a_device_stop_cameras(device);
        return Py_BuildValue("I", K4A_RESULT_SUCCEEDED);
    }

    static PyObject* device_get_capture(PyObject* self, PyObject* args){
        int32_t timeout;
        PyArg_ParseTuple(args, "I", &timeout);
        if (capture) k4a_capture_release(capture);
        k4a_capture_create(&capture);
        k4a_wait_result_t result = k4a_device_get_capture(device, &capture, timeout);
        return Py_BuildValue("I", result);
    }

    static void capsule_cleanup(PyObject *capsule) {
        k4a_image_t *image = (k4a_image_t*)PyCapsule_GetContext(capsule);
        k4a_image_release(*image);
        free(image);
    }

    static PyObject* device_get_color_image(PyObject* self, PyObject* args){
        k4a_image_t* color_image = (k4a_image_t*) malloc(sizeof(k4a_image_t));
        *color_image = k4a_capture_get_color_image(capture);
        if (color_image) {
            uint8_t* buffer = k4a_image_get_buffer(*color_image);
            npy_intp dims[3];
            dims[0] = k4a_image_get_height_pixels(*color_image);
            dims[1] = k4a_image_get_width_pixels(*color_image);
            dims[2] = 4;

            PyArrayObject* np_color_image = (PyArrayObject*) PyArray_SimpleNewFromData(3, dims, NPY_UINT8, buffer);
            PyObject *capsule = PyCapsule_New(buffer, NULL, capsule_cleanup);
            PyCapsule_SetContext(capsule, color_image);
            PyArray_SetBaseObject((PyArrayObject *) np_color_image, capsule);
            return PyArray_Return(np_color_image);
        }
        else {
            free(color_image);
            return Py_BuildValue("");
        }
    }

    static PyObject* device_get_depth_image(PyObject* self, PyObject* args){
        int is_transform_enabled;
        PyArg_ParseTuple(args, "p", &is_transform_enabled);

        k4a_image_t* depth_image = (k4a_image_t*) malloc(sizeof(k4a_image_t));
        *depth_image = k4a_capture_get_depth_image(capture);
        if (is_transform_enabled && *depth_image) {
            k4a_image_t color_image = k4a_capture_get_color_image(capture);
            if (color_image) {
                k4a_image_t depth_image_transformed;
                k4a_image_create(
                        k4a_image_get_format(*depth_image),
                        k4a_image_get_width_pixels(color_image),
                        k4a_image_get_height_pixels(color_image),
                        k4a_image_get_width_pixels(color_image) * (int)sizeof(uint16_t),
                        &depth_image_transformed);
                k4a_result_t res = k4a_transformation_depth_image_to_color_camera(
                        transformation_handle,
                        *depth_image, depth_image_transformed);
                if (res == K4A_RESULT_FAILED){
                    free(depth_image);
                    return Py_BuildValue("");
                }

                k4a_image_release(color_image);
                k4a_image_release(*depth_image);
                *depth_image = depth_image_transformed;
            }
        }

        if (*depth_image) {
            uint8_t* buffer = k4a_image_get_buffer(*depth_image);
            npy_intp dims[2];
            dims[0] = k4a_image_get_height_pixels(*depth_image);
            dims[1] = k4a_image_get_width_pixels(*depth_image);
            PyArrayObject* np_depth_image = (PyArrayObject*) PyArray_SimpleNewFromData(2, dims, NPY_UINT16, buffer);
            PyObject *capsule = PyCapsule_New(buffer, NULL, capsule_cleanup);
            PyCapsule_SetContext(capsule, depth_image);
            PyArray_SetBaseObject((PyArrayObject *) np_depth_image, capsule);
            return PyArray_Return(np_depth_image);
        }
        else {
            free(depth_image);
            return Py_BuildValue("");
        }
    }

    static PyObject* is_body_tracking_supported(PyObject* self, PyObject* args){
        return Py_BuildValue("I", #ifdef ENABLE_BODY_TRACKING 1 #else 0 #endif);
    }

#ifdef ENABLE_BODY_TRACKING
    static PyObject* device_get_pose_data(PyObject* self, PyObject* args){
        k4abt_tracker_enqueue_capture(tracker, capture, 0);
        k4abt_frame_t body_frame = NULL;
        k4a_wait_result_t pop_frame_result = k4abt_tracker_pop_result(tracker, &body_frame, 0);

        if (pop_frame_result == K4A_WAIT_RESULT_SUCCEEDED)
        {
            size_t num_bodies = k4abt_frame_get_num_bodies(body_frame);
            double_t* buffer = new double_t[num_bodies*NUM_JOINTS*NUM_DATA];

            for (size_t i = 0; i < num_bodies; i++)
            {
                k4abt_skeleton_t skeleton;
                k4abt_frame_get_body_skeleton(body_frame, i, &skeleton);
                for (int j = 0; j < (int)K4ABT_JOINT_COUNT; j++)
                {
                    k4a_float3_t position = skeleton.joints[j].position;
                    k4a_float2_t position_image;
                    int valid;
                    //Convert 3d points in mm to image coordinates
                    k4a_calibration_3d_to_2d(&calibration,
                                             &position,
                                             K4A_CALIBRATION_TYPE_DEPTH,
                                             K4A_CALIBRATION_TYPE_COLOR,
                                             &position_image, &valid);

                    buffer[(i * NUM_JOINTS * NUM_DATA) + (j * NUM_DATA) + 0] = position_image.v[0];
                    buffer[(i * NUM_JOINTS * NUM_DATA) + (j * NUM_DATA) + 1] = position_image.v[1];

                    buffer[(i * NUM_JOINTS * NUM_DATA) + (j * NUM_DATA) + 2] = position.v[0];
                    buffer[(i * NUM_JOINTS * NUM_DATA) + (j * NUM_DATA) + 3] = position.v[1];
                    buffer[(i * NUM_JOINTS * NUM_DATA) + (j * NUM_DATA) + 4] = position.v[2];

                    k4a_quaternion_t orientation = skeleton.joints[j].orientation;
                    buffer[(i * NUM_JOINTS * NUM_DATA) + (j * NUM_DATA) + 5] = orientation.v[0];
                    buffer[(i * NUM_JOINTS * NUM_DATA) + (j * NUM_DATA) + 6] = orientation.v[1];
                    buffer[(i * NUM_JOINTS * NUM_DATA) + (j * NUM_DATA) + 7] = orientation.v[2];
                    buffer[(i * NUM_JOINTS * NUM_DATA) + (j * NUM_DATA) + 8] = orientation.v[3];

                    k4abt_joint_confidence_level_t confidence_level = skeleton.joints[j].confidence_level;
                    buffer[(i * NUM_JOINTS * NUM_DATA) + (j * NUM_DATA) + 9] = confidence_level;
                }
            }

            k4abt_frame_release(body_frame);

            npy_intp dims[3];
            dims[0] = num_bodies;
            dims[1] = NUM_JOINTS;
            dims[2] = NUM_DATA;

            PyArrayObject* np_pose_data = (PyArrayObject*) PyArray_SimpleNewFromData(3, dims, NPY_DOUBLE, buffer);
            PyObject *capsule = PyCapsule_New(buffer, NULL, pose_capsule_cleanup);
            PyCapsule_SetContext(capsule, buffer);
            PyArray_SetBaseObject((PyArrayObject *) np_pose_data, capsule);
            return PyArray_Return(np_pose_data);
        }
        else {
            return Py_BuildValue("");
        }
    }

    static void pose_capsule_cleanup(PyObject *capsule) {
        double_t *buffer = (double_t*)PyCapsule_GetContext(capsule);
        delete buffer;
    }
#endif



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
        {"device_stop_cameras", device_stop_cameras, METH_VARARGS, "Stops the color and depth camera capture"},
        {"device_get_capture", device_get_capture, METH_VARARGS, "Reads a sensor capture"},
        {"device_get_color_image", device_get_color_image, METH_VARARGS, "Get the color image associated with the given capture"},
        {"device_get_depth_image", device_get_depth_image, METH_VARARGS, "Set or add a depth image to the associated capture"},
        {"device_close", device_close, METH_VARARGS, "Close an Azure Kinect device"},
        {"device_get_sync_jack", device_get_sync_jack, METH_VARARGS, "Get the device jack status for the synchronization in and synchronization out connectors."},
        {"device_get_color_control", device_get_color_control, METH_VARARGS, "Get device color control."},
        {"device_set_color_control", device_set_color_control, METH_VARARGS, "Set device color control."},
        {"device_get_color_control_capabilities", device_get_color_control_capabilities, METH_VARARGS, "Get device color control capabilities."},
        {"is_body_tracking_supported", is_body_tracking_supported, METH_VARARGS, "Checks if compiled with body tracking support."}
#ifdef ENABLE_BODY_TRACKING
        {"device_get_pose_data", device_get_pose_data, METH_VARARGS, "Get the body pose estimates associated with the given capture"},
#endif
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

        st->error = PyErr_NewException("pyk4a_module.Error", NULL, NULL);
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
