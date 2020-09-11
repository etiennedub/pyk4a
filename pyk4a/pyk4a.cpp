#include <Python.h>
#include <numpy/arrayobject.h>

#include <k4a/k4a.h>
#include <k4arecord/playback.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif
    // to debug, use fprintf(stdout, "debug msg\n") or fprintf(stderr, "debug msg\n");;

    #define NON_THREAD_SAFE 0
    // Simple way to map k4a_color_resolution_t to dimensions
    const int RESOLUTION_TO_DIMS[][2] = {{0, 0}, {1280, 720},
                                    {1920, 1080}, {2560, 1440},
                                    {2048, 1536}, {3840, 2160},
                                    {4096, 3072}};

    const char* CAPSULE_PLAYBACK_NAME = "pyk4a playback handle";
    const char* CAPSULE_DEVICE_NAME = "pyk4a device handle";
    const char* CAPSULE_CALIBRATION_NAME = "pyk4a calibration handle";
    const char* CAPSULE_TRANSFORMATION_NAME = "pyk4a transformation handle";
    const char* CAPSULE_CAPTURE_NAME = "pyk4a capture handle";

    static PyThreadState* _gil_release(int thread_safe) {
        PyThreadState *thread_state = NULL;
        if (thread_safe == NON_THREAD_SAFE) {
            thread_state = PyEval_SaveThread();
        }
        return thread_state;
    }

    static void _gil_restore(PyThreadState *thread_state) {
        if (thread_state != NULL) {
            PyEval_RestoreThread(thread_state);
        }
    }

    static void capsule_cleanup_device(PyObject *capsule) {
        k4a_device_t* device_handle;

        device_handle = (k4a_device_t*)PyCapsule_GetPointer(capsule, CAPSULE_DEVICE_NAME);
        free(device_handle);
    }

    static void capsule_cleanup_image(PyObject *capsule) {
        k4a_image_t *image = (k4a_image_t*)PyCapsule_GetContext(capsule);
        k4a_image_release(*image);
        free(image);
    }

    static void capsule_cleanup_calibration(PyObject *capsule) {
        k4a_calibration_t *calibration = (k4a_calibration_t*)PyCapsule_GetPointer(capsule, CAPSULE_CALIBRATION_NAME);
        free(calibration);
    }

    static void capsule_cleanup_capture(PyObject *capsule) {
        k4a_capture_t *capture = (k4a_capture_t*)PyCapsule_GetPointer(capsule, CAPSULE_CAPTURE_NAME);
        k4a_capture_release(*capture);
        free(capture);
    }

    static void capsule_cleanup_playback(PyObject *capsule) {
        k4a_playback_t* playback_handle;

        playback_handle = (k4a_playback_t*)PyCapsule_GetPointer(capsule, CAPSULE_PLAYBACK_NAME);
        free(playback_handle);
    }

    static void capsule_cleanup_transformation(PyObject *capsule) {
        k4a_transformation_t *transformation = (k4a_transformation_t*)PyCapsule_GetPointer(capsule, CAPSULE_TRANSFORMATION_NAME);
        k4a_transformation_destroy(*transformation);
        free(transformation);
    }

    static PyObject* device_open(PyObject* self, PyObject* args){
        uint32_t device_id;
        int thread_safe;
        PyThreadState *thread_state;
        PyArg_ParseTuple(args, "Ip", &device_id, &thread_safe);
        k4a_device_t* device_handle = (k4a_device_t*) malloc(sizeof(k4a_device_t));

        if (device_handle == NULL) {
            fprintf(stderr, "Cannot allocate memory");
            return Py_BuildValue("IN", K4A_RESULT_FAILED, Py_None);
        }

        thread_state = _gil_release(thread_safe);
        k4a_result_t result = k4a_device_open(device_id, device_handle);
        _gil_restore(thread_state);

        if (result == K4A_RESULT_FAILED ) {
            free(device_handle);
            return Py_BuildValue("IN", result, Py_None);
        }

        PyObject *capsule = PyCapsule_New(device_handle, CAPSULE_DEVICE_NAME, capsule_cleanup_device);

        return Py_BuildValue("IN", result, capsule);
    }

    static PyObject* device_close(PyObject* self, PyObject* args){
        k4a_device_t* device_handle;
        PyObject *capsule;
        int thread_safe;
        PyThreadState *thread_state;

        PyArg_ParseTuple(args, "Op", &capsule, &thread_safe);
        device_handle = (k4a_device_t*)PyCapsule_GetPointer(capsule, CAPSULE_DEVICE_NAME);

        thread_state = _gil_release(thread_safe);
        k4a_device_close(*device_handle);
        _gil_restore(thread_state);

        return Py_BuildValue("I", K4A_RESULT_SUCCEEDED);
    }

    static PyObject* device_get_sync_jack(PyObject* self, PyObject* args){
        k4a_device_t* device_handle;
        PyObject *capsule;
        int thread_safe;
        PyThreadState *thread_state;
        bool in_jack = 0;
        bool out_jack = 0;

        PyArg_ParseTuple(args, "Op", &capsule, &thread_safe);
        device_handle = (k4a_device_t*)PyCapsule_GetPointer(capsule, CAPSULE_DEVICE_NAME);

        thread_state = _gil_release(thread_safe);
        k4a_result_t result = k4a_device_get_sync_jack(*device_handle, &in_jack, &out_jack);
        _gil_restore(thread_state);

        return Py_BuildValue("III", result, in_jack, out_jack);
    }

    static PyObject* device_get_color_control(PyObject* self, PyObject* args){
        k4a_device_t* device_handle;
        PyObject *capsule;
        int thread_safe;
        PyThreadState *thread_state;
        k4a_color_control_command_t command;
        k4a_color_control_mode_t mode;
        int32_t value = 0;

        PyArg_ParseTuple(args, "OpI", &capsule, &thread_safe, &command);
        device_handle = (k4a_device_t*)PyCapsule_GetPointer(capsule, CAPSULE_DEVICE_NAME);

        thread_state = _gil_release(thread_safe);
        k4a_result_t result = k4a_device_get_color_control(*device_handle, command, &mode, &value);
        _gil_restore(thread_state);

        if (result == K4A_RESULT_FAILED) {
            return Py_BuildValue("IIi", 0, 0, 0);
        }
        return Py_BuildValue("IIi", result, mode, value);
    }

    static PyObject* device_set_color_control(PyObject* self, PyObject* args){
        k4a_device_t* device_handle;
        PyObject *capsule;
        int thread_safe;
        PyThreadState *thread_state;
        k4a_color_control_command_t command = K4A_COLOR_CONTROL_EXPOSURE_TIME_ABSOLUTE;
        k4a_color_control_mode_t mode = K4A_COLOR_CONTROL_MODE_MANUAL;
        int32_t value = 0;

        PyArg_ParseTuple(args, "OpIIi", &capsule, &thread_safe, &command, &mode, &value);
        device_handle = (k4a_device_t*)PyCapsule_GetPointer(capsule, CAPSULE_DEVICE_NAME);

        thread_state = _gil_release(thread_safe);
        k4a_result_t result = k4a_device_set_color_control(*device_handle, command, mode, value);
        _gil_restore(thread_state);

        if (result == K4A_RESULT_FAILED) {
            return Py_BuildValue("I", K4A_RESULT_FAILED);
        }
        return Py_BuildValue("I", result);
    }

    static PyObject* device_get_color_control_capabilities(PyObject* self, PyObject* args){
        k4a_device_t* device_handle;
        PyObject *capsule;
        int thread_safe;
        PyThreadState *thread_state;
        k4a_color_control_command_t command;
        bool supports_auto;
        int min_value;
        int max_value;
        int step_value;
        int default_value;
        k4a_color_control_mode_t default_mode;

        PyArg_ParseTuple(args, "OpI", &capsule, &thread_safe, &command);
        device_handle = (k4a_device_t*)PyCapsule_GetPointer(capsule, CAPSULE_DEVICE_NAME);

        thread_state = _gil_release(thread_safe);
        k4a_result_t result = k4a_device_get_color_control_capabilities(*device_handle, command, &supports_auto, &min_value, &max_value, &step_value, &default_value, &default_mode);
        _gil_restore(thread_state);

        if (result == K4A_RESULT_FAILED) {
            return Py_BuildValue("IN", result, Py_None);
        }

        return Py_BuildValue("I{s:I,s:N,s:i,s:i,s:i,s:i,s:I}", result,
                "color_control_command", command,
                "supports_auto", supports_auto ? Py_True: Py_False,
                "min_value", min_value,
                "max_value", max_value,
                "step_value", step_value,
                "default_value", default_value,
                "default_mode", default_mode);
    }

    static PyObject* device_start_cameras(PyObject* self, PyObject* args){
        k4a_device_t* device_handle;
        PyObject *capsule;
        int thread_safe;
        PyThreadState *thread_state;
        k4a_device_configuration_t config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;

        PyArg_ParseTuple(args, "OpIIIIpiIIp", &capsule, &thread_safe,
                &config.color_format,
                &config.color_resolution, &config.depth_mode,
                &config.camera_fps, &config.synchronized_images_only,
                &config.depth_delay_off_color_usec, &config.wired_sync_mode,
                &config.subordinate_delay_off_master_usec,
                &config.disable_streaming_indicator);
        device_handle = (k4a_device_t*)PyCapsule_GetPointer(capsule, CAPSULE_DEVICE_NAME);

        k4a_result_t result;
        thread_state = _gil_release(thread_safe);
        result = k4a_device_start_cameras(*device_handle, &config);
        _gil_restore(thread_state);
        return Py_BuildValue("I", result);
    }

    static PyObject* device_start_imu(PyObject* self, PyObject* args){
        k4a_device_t* device_handle;
        PyObject *capsule;
        int thread_safe;
        PyThreadState *thread_state;
        k4a_result_t result;

        PyArg_ParseTuple(args, "Op", &capsule, &thread_safe);
        device_handle = (k4a_device_t*)PyCapsule_GetPointer(capsule, CAPSULE_DEVICE_NAME);

        thread_state = _gil_release(thread_safe);
        result = k4a_device_start_imu(*device_handle);
        _gil_restore(thread_state);

        return Py_BuildValue("I", result);
    }

    static PyObject* device_stop_cameras(PyObject* self, PyObject* args){
        k4a_device_t* device_handle;
        PyObject *capsule;
        int thread_safe;
        PyThreadState *thread_state;

        PyArg_ParseTuple(args, "Op", &capsule, &thread_safe);
        device_handle = (k4a_device_t*)PyCapsule_GetPointer(capsule, CAPSULE_DEVICE_NAME);

        thread_state = _gil_release(thread_safe);
        k4a_device_stop_cameras(*device_handle);
        _gil_restore(thread_state);

        return Py_BuildValue("I", K4A_RESULT_SUCCEEDED);
    }

    static PyObject* device_stop_imu(PyObject* self, PyObject* args){
        k4a_device_t* device_handle;
        PyObject *capsule;
        int thread_safe;
        PyThreadState *thread_state;

        PyArg_ParseTuple(args, "Op", &capsule, &thread_safe);
        device_handle = (k4a_device_t*)PyCapsule_GetPointer(capsule, CAPSULE_DEVICE_NAME);

        thread_state = _gil_release(thread_safe);
        k4a_device_stop_imu(*device_handle);
        _gil_restore(thread_state);

        return Py_BuildValue("I", K4A_RESULT_SUCCEEDED);
    }

    static PyObject* device_get_capture(PyObject* self, PyObject* args){
        k4a_device_t* device_handle;
        PyObject *capsule;
        int thread_safe;
        PyThreadState *thread_state;
        long long timeout;
        k4a_wait_result_t result;

        PyArg_ParseTuple(args, "OpL", &capsule, &thread_safe, &timeout);
        device_handle = (k4a_device_t*)PyCapsule_GetPointer(capsule, CAPSULE_DEVICE_NAME);

        k4a_capture_t* capture = (k4a_capture_t*) malloc(sizeof(k4a_capture_t));
        if (capture == NULL) {
            fprintf(stderr, "Cannot allocate memory");
            return Py_BuildValue("IN", K4A_RESULT_FAILED, Py_None);
        }
        k4a_capture_create(capture);
        PyObject* capsule_capture = PyCapsule_New(capture, CAPSULE_CAPTURE_NAME, capsule_cleanup_capture);

        thread_state = _gil_release(thread_safe);
        result = k4a_device_get_capture(*device_handle, capture, timeout);
        _gil_restore(thread_state);

        return Py_BuildValue("IN", result, capsule_capture);
    }

    static PyObject* device_get_imu_sample(PyObject* self, PyObject* args){
        k4a_device_t* device_handle;
        PyObject *capsule;
        int thread_safe;
        PyThreadState *thread_state;
        long long timeout;
        k4a_imu_sample_t imu_sample;
        k4a_wait_result_t result;

        PyArg_ParseTuple(args, "OpL", &capsule, &thread_safe, &timeout);
        device_handle = (k4a_device_t*)PyCapsule_GetPointer(capsule, CAPSULE_DEVICE_NAME);

        thread_state = _gil_release(thread_safe);
        result = k4a_device_get_imu_sample(*device_handle, &imu_sample, timeout);
        _gil_restore(thread_state);

        if (K4A_WAIT_RESULT_SUCCEEDED == result) {
            return Py_BuildValue("I{s:f,s:(fff),s:L,s:(fff),s:L}", result,
                    "temperature", imu_sample.temperature,
                    "acc_sample", imu_sample.acc_sample.xyz.x, imu_sample.acc_sample.xyz.y, imu_sample.acc_sample.xyz.z,
                    "acc_timestamp", imu_sample.acc_timestamp_usec,
                    "gyro_sample", imu_sample.gyro_sample.xyz.x, imu_sample.gyro_sample.xyz.y, imu_sample.gyro_sample.xyz.z,
                    "gyro_timestamp", imu_sample.gyro_timestamp_usec);
        }

        return Py_BuildValue("IN", result, Py_None);
    }

    static PyObject* calibration_get_from_raw(PyObject* self, PyObject* args){
        k4a_calibration_t* calibration_handle;
        int thread_safe;
        PyThreadState *thread_state;
        char * raw_calibration;
        k4a_depth_mode_t depth_mode;
        k4a_color_resolution_t color_resolution;
        k4a_result_t result;

        PyArg_ParseTuple(args, "psII", &thread_safe, &raw_calibration, &depth_mode, &color_resolution);

        size_t raw_calibration_size = strlen(raw_calibration) + 1;

        calibration_handle = (k4a_calibration_t*) malloc(sizeof(k4a_calibration_t));
        if (calibration_handle == NULL) {
            fprintf(stderr, "Cannot allocate memory");
            return Py_BuildValue("IN", K4A_RESULT_FAILED, Py_None);
        }

        thread_state = _gil_release(thread_safe);
        result = k4a_calibration_get_from_raw(raw_calibration,
                raw_calibration_size, depth_mode,
                color_resolution, calibration_handle);
        if (result == K4A_RESULT_FAILED) {
            _gil_restore(thread_state);
            return Py_BuildValue("IN", result, Py_None);
        }
        _gil_restore(thread_state);
        PyObject *calibration_capsule = PyCapsule_New(calibration_handle, CAPSULE_CALIBRATION_NAME, capsule_cleanup_calibration);

        return Py_BuildValue("IN", K4A_RESULT_SUCCEEDED, calibration_capsule);
    }

    static PyObject* device_get_calibration(PyObject* self, PyObject* args){
        k4a_device_t* device_handle;
        PyObject *capsule;
        int thread_safe;
        PyThreadState *thread_state;
        k4a_depth_mode_t depth_mode;
        k4a_color_resolution_t color_resolution;
        k4a_result_t result;

        PyArg_ParseTuple(args, "OpII", &capsule, &thread_safe, &depth_mode, &color_resolution);
        device_handle = (k4a_device_t*)PyCapsule_GetPointer(capsule, CAPSULE_DEVICE_NAME);

        k4a_calibration_t* calibration_handle = (k4a_calibration_t*) malloc(sizeof(k4a_calibration_t));
        if (calibration_handle == NULL) {
            fprintf(stderr, "Cannot allocate memory");
            return Py_BuildValue("IN", K4A_RESULT_FAILED, Py_None);
        }

        thread_state = _gil_release(thread_safe);
        result = k4a_device_get_calibration(*device_handle, depth_mode, color_resolution, calibration_handle);
        _gil_restore(thread_state);

        if (result != K4A_RESULT_SUCCEEDED) {
            free(calibration_handle);
            return Py_BuildValue("IN", K4A_RESULT_FAILED, Py_None);
        }

        PyObject *calibration_capsule = PyCapsule_New(calibration_handle, CAPSULE_CALIBRATION_NAME, capsule_cleanup_calibration);

        return Py_BuildValue("IN", result, calibration_capsule);

    }

    static PyObject* device_get_raw_calibration(PyObject* self, PyObject* args){
        k4a_device_t* device_handle;
        PyObject *capsule;
        int thread_safe;
        PyThreadState *thread_state;
        k4a_buffer_result_t result;
        size_t data_size;

        PyArg_ParseTuple(args, "Op", &capsule, &thread_safe);
        device_handle = (k4a_device_t*)PyCapsule_GetPointer(capsule, CAPSULE_DEVICE_NAME);

        thread_state = _gil_release(thread_safe);
        result = k4a_device_get_raw_calibration(*device_handle, NULL, &data_size);
        if (result == K4A_BUFFER_RESULT_FAILED) {
            _gil_restore(thread_state);
            return Py_BuildValue("s", "");
        }
        uint8_t* data = (uint8_t*) malloc(data_size);
        if (data == NULL) {
            _gil_restore(thread_state);
            fprintf(stderr, "Cannot allocate memory");
            return Py_BuildValue("s", "");
        }
        result = k4a_device_get_raw_calibration(*device_handle, data, &data_size);
        _gil_restore(thread_state);
        if (result == K4A_BUFFER_RESULT_FAILED) {
            return Py_BuildValue("s", "");
        }

        PyObject* res = Py_BuildValue("s", data);
        free(data);
        return res;
    }

    k4a_result_t k4a_image_to_numpy(k4a_image_t* img_src, PyArrayObject** img_dst){
        uint8_t* buffer = k4a_image_get_buffer(*img_src);
        npy_intp dims[3];

        k4a_image_format_t format = k4a_image_get_format(*img_src);
        switch (format){
            case K4A_IMAGE_FORMAT_COLOR_BGRA32:
                dims[0] = k4a_image_get_height_pixels(*img_src);
                dims[1] = k4a_image_get_width_pixels(*img_src);
                dims[2] = 4;
                *img_dst = (PyArrayObject*) PyArray_SimpleNewFromData(3, dims, NPY_UINT8, buffer);
                break;
            case K4A_IMAGE_FORMAT_COLOR_MJPG:
                dims[0] = k4a_image_get_size(*img_src);
                *img_dst = (PyArrayObject*) PyArray_SimpleNewFromData(1, dims, NPY_UINT8, buffer);
                break;
            case K4A_IMAGE_FORMAT_COLOR_YUY2:
                dims[0] = k4a_image_get_height_pixels(*img_src);
                dims[1] = k4a_image_get_width_pixels(*img_src);
                dims[2] = 2;
                *img_dst = (PyArrayObject*) PyArray_SimpleNewFromData(3, dims, NPY_UINT8, buffer);
                break;
            case K4A_IMAGE_FORMAT_COLOR_NV12:
                dims[0] = k4a_image_get_height_pixels(*img_src);
                dims[0] += dims[0] /2;
                dims[1] = k4a_image_get_width_pixels(*img_src);
                dims[2] = 1;
                *img_dst = (PyArrayObject*) PyArray_SimpleNewFromData(3, dims, NPY_UINT8, buffer);
                break;
            case K4A_IMAGE_FORMAT_DEPTH16:
            case K4A_IMAGE_FORMAT_IR16:
                dims[0] = k4a_image_get_height_pixels(*img_src);
                dims[1] = k4a_image_get_width_pixels(*img_src);
                *img_dst = (PyArrayObject*) PyArray_SimpleNewFromData(2, dims, NPY_UINT16, buffer);
                break;
            case K4A_IMAGE_FORMAT_CUSTOM:
                // xyz in uint16 format
                dims[0] = k4a_image_get_height_pixels(*img_src);
                dims[1] = k4a_image_get_width_pixels(*img_src);
                dims[2] = 3;
                *img_dst = (PyArrayObject*) PyArray_SimpleNewFromData(3, dims, NPY_INT16, buffer);
                break;
            default:
                // Not supported
                return K4A_RESULT_FAILED;
        }

        PyObject *capsule = PyCapsule_New(buffer, NULL, capsule_cleanup_image);
        PyCapsule_SetContext(capsule, img_src);
        PyArray_SetBaseObject((PyArrayObject *) *img_dst, capsule);

        return K4A_RESULT_SUCCEEDED;
    }

    k4a_result_t numpy_to_k4a_image(PyArrayObject* img_src, k4a_image_t* img_dst,
            k4a_image_format_t format){

        int width_pixels = img_src->dimensions[1];
        int height_pixels = img_src->dimensions[0];
        int pixel_size;

        switch (format){
            case K4A_IMAGE_FORMAT_DEPTH16:
                pixel_size = (int)sizeof(uint16_t);
                break;
            case K4A_IMAGE_FORMAT_COLOR_BGRA32:
                pixel_size = (int)sizeof(uint32_t);
                break;
            default:
                // Not supported
                return K4A_RESULT_FAILED;
        }

        return k4a_image_create_from_buffer(
                format,
                width_pixels, height_pixels,
                width_pixels * pixel_size,
                (uint8_t*) img_src->data,
                width_pixels * height_pixels * pixel_size,
                NULL, NULL, img_dst);
    }




    static PyObject* transformation_create(PyObject* self, PyObject *args){
        k4a_calibration_t* calibration_handle;
        PyObject *capsule;
        int thread_safe;
        PyThreadState *thread_state;

        PyArg_ParseTuple(args, "Op", &capsule, &thread_safe);
        calibration_handle = (k4a_calibration_t*)PyCapsule_GetPointer(capsule, CAPSULE_CALIBRATION_NAME);

        thread_state = _gil_release(thread_safe);

        k4a_transformation_t* transformation_handle = (k4a_transformation_t*) malloc(sizeof(k4a_transformation_t));
        if (calibration_handle == NULL) {
            fprintf(stderr, "Cannot allocate memory");
            return Py_BuildValue("N", Py_None);
        }
        *transformation_handle = k4a_transformation_create(calibration_handle);
        _gil_restore(thread_state);
        if (transformation_handle == NULL ) {
            return Py_BuildValue("N", Py_None);
        }
        PyObject *transformation_capsule = PyCapsule_New(transformation_handle, CAPSULE_TRANSFORMATION_NAME, capsule_cleanup_transformation);

        return Py_BuildValue("N", transformation_capsule);
    }

    static PyObject* transformation_depth_image_to_color_camera(PyObject* self, PyObject* args){
        k4a_transformation_t* transformation_handle;
        PyObject *capsule;
        int thread_safe;
        PyThreadState *thread_state;
        k4a_result_t res;
        PyArrayObject *in_array;
        k4a_color_resolution_t color_resolution;

        PyArg_ParseTuple(args, "OpO!I", &capsule, &thread_safe, &PyArray_Type, &in_array, &color_resolution);
        transformation_handle = (k4a_transformation_t*)PyCapsule_GetPointer(capsule, CAPSULE_TRANSFORMATION_NAME);

        k4a_image_t* depth_image_transformed = (k4a_image_t*) malloc(sizeof(k4a_image_t));

        k4a_image_t depth_image;
        res = numpy_to_k4a_image(in_array, &depth_image, K4A_IMAGE_FORMAT_DEPTH16);
        thread_state = _gil_release(thread_safe);
        if (K4A_RESULT_SUCCEEDED == res) {
            res = k4a_image_create(
                    k4a_image_get_format(depth_image),
                    RESOLUTION_TO_DIMS[color_resolution][0],
                    RESOLUTION_TO_DIMS[color_resolution][1],
                    RESOLUTION_TO_DIMS[color_resolution][0] * (int)sizeof(uint16_t),
                    depth_image_transformed);
        }

        if (K4A_RESULT_SUCCEEDED == res) {
            res = k4a_transformation_depth_image_to_color_camera(
                    *transformation_handle,
                    depth_image, *depth_image_transformed);
            k4a_image_release(depth_image);
        }
        _gil_restore(thread_state);
        PyArrayObject* np_depth_image;
        if (K4A_RESULT_SUCCEEDED == res) {
            res = k4a_image_to_numpy(depth_image_transformed, &np_depth_image);
        }

        if (K4A_RESULT_SUCCEEDED == res) {
            return PyArray_Return(np_depth_image);
        }
        else {
            free(depth_image_transformed);
            return Py_BuildValue("");
        }
    }

    static PyObject* transformation_depth_image_to_point_cloud(PyObject* self, PyObject* args) {
        k4a_transformation_t* transformation_handle;
        PyObject *capsule;
        int thread_safe;
        PyThreadState *thread_state;
        k4a_result_t res;
        PyArrayObject *depth_in_array;
        int calibration_type_depth;

        PyArg_ParseTuple(args, "OpO!p", &capsule, &thread_safe, &PyArray_Type, &depth_in_array, &calibration_type_depth);
        transformation_handle = (k4a_transformation_t*)PyCapsule_GetPointer(capsule, CAPSULE_TRANSFORMATION_NAME);

        k4a_calibration_type_t camera;
        if (calibration_type_depth == 1) {
            camera = K4A_CALIBRATION_TYPE_DEPTH;
        } else {
            camera = K4A_CALIBRATION_TYPE_COLOR;
        }
        k4a_image_t *xyz_image = (k4a_image_t *) malloc(sizeof(k4a_image_t));

        k4a_image_t depth_image;
        res = numpy_to_k4a_image(depth_in_array, &depth_image, K4A_IMAGE_FORMAT_DEPTH16);
        thread_state = _gil_release(thread_safe);
        if (K4A_RESULT_SUCCEEDED == res) {
            res = k4a_image_create(
                    K4A_IMAGE_FORMAT_CUSTOM,
                    k4a_image_get_width_pixels(depth_image),
                    k4a_image_get_height_pixels(depth_image),
                    k4a_image_get_width_pixels(depth_image) * 3 * (int) sizeof(int16_t),
                    xyz_image);
        }

        if (K4A_RESULT_SUCCEEDED == res) {
            res = k4a_transformation_depth_image_to_point_cloud(
                    *transformation_handle,
                    depth_image, camera, *xyz_image);
            k4a_image_release(depth_image);
        }
        _gil_restore(thread_state);

        PyArrayObject* np_xyz_image;
        if (K4A_RESULT_SUCCEEDED == res) {
            res = k4a_image_to_numpy(xyz_image, &np_xyz_image);
        }

        if (K4A_RESULT_SUCCEEDED == res) {
            return PyArray_Return(np_xyz_image);
        } else {
            free(xyz_image);
            return Py_BuildValue("");
        }
    }

    static PyObject* transformation_color_image_to_depth_camera(
            PyObject* self, PyObject* args){
        k4a_transformation_t* transformation_handle;
        PyObject *capsule;
        int thread_safe;
        PyThreadState *thread_state;
        k4a_result_t res;
        PyArrayObject *in_depth_array;
        PyArrayObject *in_color_array;

        PyArg_ParseTuple(args, "OpO!O!", &capsule, &thread_safe, &PyArray_Type, &in_depth_array, &PyArray_Type, &in_color_array);
        transformation_handle = (k4a_transformation_t*)PyCapsule_GetPointer(capsule, CAPSULE_TRANSFORMATION_NAME);

        k4a_image_t* transformed_color_image = (k4a_image_t*) malloc(sizeof(k4a_image_t));

        k4a_image_t depth_image;
        k4a_image_t color_image;
        res = numpy_to_k4a_image(in_depth_array, &depth_image, K4A_IMAGE_FORMAT_DEPTH16);
        if (K4A_RESULT_SUCCEEDED == res) {
            res = numpy_to_k4a_image(in_color_array, &color_image, K4A_IMAGE_FORMAT_COLOR_BGRA32);
            if (K4A_RESULT_SUCCEEDED == res) {
                res = k4a_image_create(
                        K4A_IMAGE_FORMAT_COLOR_BGRA32,
                        k4a_image_get_width_pixels(depth_image),
                        k4a_image_get_height_pixels(depth_image),
                        k4a_image_get_width_pixels(depth_image) * (int) sizeof(uint32_t),
                        transformed_color_image);
            }
        }

        thread_state = _gil_release(thread_safe);
        if (K4A_RESULT_SUCCEEDED == res) {
            res = k4a_transformation_color_image_to_depth_camera(
                    *transformation_handle,
                    depth_image, color_image, *transformed_color_image);
            k4a_image_release(depth_image);
            k4a_image_release(color_image);
        }
        _gil_restore(thread_state);
        PyArrayObject* np_color_image;
        if (K4A_RESULT_SUCCEEDED == res) {
            res = k4a_image_to_numpy(transformed_color_image, &np_color_image);
        }

        if (K4A_RESULT_SUCCEEDED == res) {
            return PyArray_Return(np_color_image);
        }
        else {
            free(transformed_color_image);
            return Py_BuildValue("");
        }
    }

    static PyObject* capture_get_color_image(PyObject* self, PyObject* args){
        k4a_capture_t* capture_handle;
        PyObject *capsule;
        int thread_safe;
        PyThreadState *thread_state;
        k4a_result_t res = K4A_RESULT_FAILED;

        PyArg_ParseTuple(args, "Op", &capsule, &thread_safe);
        capture_handle = (k4a_capture_t*)PyCapsule_GetPointer(capsule, CAPSULE_CAPTURE_NAME);

        k4a_image_t* image = (k4a_image_t*) malloc(sizeof(k4a_image_t));
        if (image == NULL) {
            fprintf(stderr, "Cannot allocate memory");
            return Py_BuildValue("");
        }

        thread_state = _gil_release(thread_safe);
        *image = k4a_capture_get_color_image(*capture_handle);
        _gil_restore(thread_state);

        PyArrayObject* np_image;
        if (*image) {
            res = k4a_image_to_numpy(image, &np_image);
        }

        if (K4A_RESULT_SUCCEEDED == res) {
            return PyArray_Return(np_image);
        }
        else {
            free(image);
            return Py_BuildValue("");
        }
    }

    static PyObject* capture_get_depth_image(PyObject* self, PyObject* args){
        k4a_capture_t* capture_handle;
        PyObject *capsule;
        int thread_safe;
        PyThreadState *thread_state;
        k4a_result_t res = K4A_RESULT_FAILED;

        PyArg_ParseTuple(args, "Op", &capsule, &thread_safe);
        capture_handle = (k4a_capture_t*)PyCapsule_GetPointer(capsule, CAPSULE_CAPTURE_NAME);

        k4a_image_t* image = (k4a_image_t*) malloc(sizeof(k4a_image_t));
        if (image == NULL) {
            fprintf(stderr, "Cannot allocate memory");
            return Py_BuildValue("");
        }

        thread_state = _gil_release(thread_safe);
        *image = k4a_capture_get_depth_image(*capture_handle);
        _gil_restore(thread_state);

        PyArrayObject* np_image;
        if (*image) {
            res = k4a_image_to_numpy(image, &np_image);
        }

        if (K4A_RESULT_SUCCEEDED == res) {
            return PyArray_Return(np_image);
        }
        else {
            free(image);
            return Py_BuildValue("");
        }
    }

    static PyObject* capture_get_ir_image(PyObject* self, PyObject* args){
        k4a_capture_t* capture_handle;
        PyObject *capsule;
        int thread_safe;
        PyThreadState *thread_state;
        k4a_result_t res = K4A_RESULT_FAILED;

        PyArg_ParseTuple(args, "Op", &capsule, &thread_safe);
        capture_handle = (k4a_capture_t*)PyCapsule_GetPointer(capsule, CAPSULE_CAPTURE_NAME);

        k4a_image_t* image = (k4a_image_t*) malloc(sizeof(k4a_image_t));
        if (image == NULL) {
            fprintf(stderr, "Cannot allocate memory");
            return Py_BuildValue("");
        }

        thread_state = _gil_release(thread_safe);
        *image = k4a_capture_get_ir_image(*capture_handle);
        _gil_restore(thread_state);

        PyArrayObject* np_image;
        if (*image) {
            res = k4a_image_to_numpy(image, &np_image);
        }

        if (K4A_RESULT_SUCCEEDED == res) {
            return PyArray_Return(np_image);
        }
        else {
            free(image);
            return Py_BuildValue("");
        }
    }

    static PyObject* calibration_3d_to_3d(PyObject* self, PyObject *args){
        k4a_calibration_t* calibration_handle;
        PyObject *capsule;
        int thread_safe;
        PyThreadState *thread_state;
        k4a_result_t res;
        k4a_float3_t source_point3d_mm;
        k4a_float3_t target_point3d_mm;
        k4a_calibration_type_t source_camera;
        k4a_calibration_type_t target_camera;
        float source_point_x;
        float source_point_y;
        float source_point_z;
        PyArg_ParseTuple(args, "Op(fff)II",
                &capsule,
                &thread_safe,
                &source_point_x,
                &source_point_y,
                &source_point_z,
                &source_camera,
                &target_camera);
        calibration_handle = (k4a_calibration_t*)PyCapsule_GetPointer(capsule, CAPSULE_CALIBRATION_NAME);

        thread_state = _gil_release(thread_safe);
        source_point3d_mm.xyz.x = source_point_x;
        source_point3d_mm.xyz.y = source_point_y;
        source_point3d_mm.xyz.z = source_point_z;

        res = k4a_calibration_3d_to_3d (calibration_handle,
                                        &source_point3d_mm,
                                        source_camera,
                                        target_camera,
                                        &target_point3d_mm);
       _gil_restore(thread_state);
        if (res == K4A_RESULT_FAILED ) {
            return Py_BuildValue("IN", res, Py_None);
        }
        return Py_BuildValue("I(fff)", res, target_point3d_mm.xyz.x, target_point3d_mm.xyz.y, target_point3d_mm.xyz.z);
    }

    static PyObject* calibration_2d_to_3d(PyObject* self, PyObject *args){
        k4a_calibration_t* calibration_handle;
        PyObject *capsule;
        int thread_safe;
        PyThreadState *thread_state;
        float source_point_x;
        float source_point_y;
        float source_depth_mm;
        int valid;
        k4a_calibration_type_t source_camera;
        k4a_calibration_type_t target_camera;
        k4a_result_t res;
        k4a_float2_t source_point2d;
        k4a_float3_t target_point3d_mm;

        PyArg_ParseTuple(args, "Op(ff)fII",
                &capsule,
                &thread_safe,
                &source_point_x,
                &source_point_y,
                &source_depth_mm,
                &source_camera,
                &target_camera);
        calibration_handle = (k4a_calibration_t*)PyCapsule_GetPointer(capsule, CAPSULE_CALIBRATION_NAME);

        thread_state = _gil_release(thread_safe);
        source_point2d.xy.x = source_point_x;
        source_point2d.xy.y = source_point_y;

        res = k4a_calibration_2d_to_3d (calibration_handle,
                                        &source_point2d,
                                        source_depth_mm,
                                        source_camera,
                                        target_camera,
                                        &target_point3d_mm,
                                        &valid);
        _gil_restore(thread_state);
        if (res == K4A_RESULT_FAILED ) {
            return Py_BuildValue("IIN", res, valid, Py_None);
        }
        // Return object...
        return Py_BuildValue("II(fff)", res, valid, target_point3d_mm.xyz.x, target_point3d_mm.xyz.y, target_point3d_mm.xyz.z);
    }

    static PyObject* playback_open(PyObject* self, PyObject *args) {
        int thread_safe;
        PyThreadState *thread_state;
        k4a_result_t result;
        const char* file_name;
        k4a_playback_t* playback_handle = (k4a_playback_t*) malloc(sizeof(k4a_playback_t));

        if (playback_handle == NULL) {
            fprintf(stderr, "Cannot allocate memory");
            return Py_BuildValue("IN", K4A_RESULT_FAILED, Py_None);
        }

        PyArg_ParseTuple(args, "sp", &file_name, &thread_safe);

        thread_state = _gil_release(thread_safe);
        result = k4a_playback_open(file_name, playback_handle);
        _gil_restore(thread_state);

        if (result == K4A_RESULT_FAILED ) {
            free(playback_handle);
            return Py_BuildValue("IN", result, Py_None);
        }

        PyObject *capsule = PyCapsule_New(playback_handle, CAPSULE_PLAYBACK_NAME, capsule_cleanup_playback);
        return Py_BuildValue("IN", result, capsule);
    }

    static PyObject* playback_close(PyObject* self, PyObject *args) {
        int thread_safe;
        PyThreadState *thread_state;
        PyObject *capsule;
        k4a_playback_t* playback_handle;

        PyArg_ParseTuple(args, "Op", &capsule, &thread_safe);
        playback_handle = (k4a_playback_t*)PyCapsule_GetPointer(capsule, CAPSULE_PLAYBACK_NAME);

        thread_state = _gil_release(thread_safe);
        k4a_playback_close(*playback_handle);
        _gil_restore(thread_state);

        return Py_BuildValue("I", K4A_RESULT_SUCCEEDED);
    }

    static PyObject* playback_get_recording_length_usec(PyObject* self, PyObject *args) {

        int thread_safe;
        PyThreadState *thread_state;
        PyObject *capsule;
        k4a_playback_t* playback_handle;
        uint64_t recording_length;

        PyArg_ParseTuple(args, "Op", &capsule, &thread_safe);
        playback_handle = (k4a_playback_t*)PyCapsule_GetPointer(capsule, CAPSULE_PLAYBACK_NAME);

        thread_state = _gil_release(thread_safe);
        recording_length = k4a_playback_get_recording_length_usec(*playback_handle);
        _gil_restore(thread_state);

        return Py_BuildValue("K", recording_length);
    }


    static PyObject* playback_get_raw_calibration(PyObject* self, PyObject* args){
        int thread_safe;
        PyThreadState *thread_state;
        PyObject *capsule;
        k4a_playback_t* playback_handle;
        k4a_buffer_result_t result;
        size_t data_size;

        PyArg_ParseTuple(args, "Op", &capsule, &thread_safe);
        playback_handle = (k4a_playback_t*)PyCapsule_GetPointer(capsule, CAPSULE_PLAYBACK_NAME);

        thread_state = _gil_release(thread_safe);
        result = k4a_playback_get_raw_calibration(*playback_handle, NULL, &data_size);
        if (result == K4A_BUFFER_RESULT_FAILED) {
            _gil_restore(thread_state);
            return Py_BuildValue("IN", result, Py_None);
        }
        uint8_t* data = (uint8_t*) malloc(data_size);
        result = k4a_playback_get_raw_calibration(*playback_handle, data, &data_size);
        _gil_restore(thread_state);
        if (result != K4A_BUFFER_RESULT_SUCCEEDED) {
            free(data);
            return Py_BuildValue("IN", result, Py_None);
        }
        PyObject* res = Py_BuildValue("Is", result, data);
        free(data);
        return res;
    }

    static PyObject* playback_seek_timestamp(PyObject* self, PyObject *args) {
        int thread_safe;
        PyThreadState *thread_state;
        PyObject *capsule;
        k4a_playback_t* playback_handle;
        uint64_t offset;
        k4a_playback_seek_origin_t origin;
        k4a_result_t result;

        PyArg_ParseTuple(args, "OpKI", &capsule, &thread_safe, &offset, &origin);
        playback_handle = (k4a_playback_t*)PyCapsule_GetPointer(capsule, CAPSULE_PLAYBACK_NAME);

        thread_state = _gil_release(thread_safe);
        result = k4a_playback_seek_timestamp(*playback_handle, offset, origin);
        _gil_restore(thread_state);

        return Py_BuildValue("I", result);
    }

    static PyObject* playback_get_calibration(PyObject* self, PyObject *args) {
        int thread_safe;
        PyThreadState *thread_state;
        PyObject *capsule;
        k4a_playback_t* playback_handle;
        k4a_calibration_t* calibration_handle;
        k4a_result_t result;

        PyArg_ParseTuple(args, "Op", &capsule, &thread_safe);
        playback_handle = (k4a_playback_t*)PyCapsule_GetPointer(capsule, CAPSULE_PLAYBACK_NAME);

        calibration_handle = (k4a_calibration_t*) malloc(sizeof(k4a_calibration_t));
        if (calibration_handle == NULL) {
            fprintf(stderr, "Cannot allocate memory");
            return Py_BuildValue("IN", K4A_RESULT_FAILED, Py_None);
        }

        thread_state = _gil_release(thread_safe);
        result = k4a_playback_get_calibration(*playback_handle, calibration_handle);
        _gil_restore(thread_state);

        if (result == K4A_RESULT_FAILED ) {
            free(calibration_handle);
            return Py_BuildValue("IN", result, Py_None);
        }

        PyObject *calibration_capsule = PyCapsule_New(calibration_handle, CAPSULE_CALIBRATION_NAME, capsule_cleanup_calibration);

        return Py_BuildValue("IN", result, calibration_capsule);
    }

    static PyObject* playback_get_record_configuration(PyObject* self, PyObject *args) {
        int thread_safe;
        PyThreadState *thread_state;
        PyObject *capsule;
        k4a_playback_t* playback_handle;
        k4a_result_t result;
        k4a_record_configuration_t config;

        PyArg_ParseTuple(args, "Op", &capsule, &thread_safe);
        playback_handle = (k4a_playback_t*)PyCapsule_GetPointer(capsule, CAPSULE_PLAYBACK_NAME);

        thread_state = _gil_release(thread_safe);
        result = k4a_playback_get_record_configuration(*playback_handle, &config);
        _gil_restore(thread_state);

        if (result == K4A_RESULT_FAILED ) {
            return Py_BuildValue("IN", result, Py_None);
        }

        return Py_BuildValue("I(IIIIiiiiIIII)",
            result,
            config.color_format,
            config.color_resolution,
            config.depth_mode,
            config.camera_fps,
            config.color_track_enabled,
            config.depth_track_enabled,
            config.ir_track_enabled,
            config.imu_track_enabled,
            config.depth_delay_off_color_usec,
            config.wired_sync_mode,
            config.subordinate_delay_off_master_usec,
            config.start_timestamp_offset_usec
            );
    }

  static PyObject* playback_get_next_capture(PyObject* self, PyObject *args) {
        int thread_safe;
        PyThreadState *thread_state;
        PyObject *capsule;
        k4a_playback_t* playback_handle;
        k4a_stream_result_t result;

        PyArg_ParseTuple(args, "Op", &capsule, &thread_safe);
        playback_handle = (k4a_playback_t*)PyCapsule_GetPointer(capsule, CAPSULE_PLAYBACK_NAME);

        k4a_capture_t* capture = (k4a_capture_t*) malloc(sizeof(k4a_capture_t));
        if (capture == NULL) {
            fprintf(stderr, "Cannot allocate memory");
            return Py_BuildValue("IN", K4A_RESULT_FAILED, Py_None);
        }

        thread_state = _gil_release(thread_safe);
        result = k4a_playback_get_next_capture(*playback_handle, capture);
        _gil_restore(thread_state);

        if (result != K4A_STREAM_RESULT_SUCCEEDED) {
            free(capture);
            return Py_BuildValue("IN", result, Py_None);
        }
        PyObject* capsule_capture = PyCapsule_New(capture, CAPSULE_CAPTURE_NAME, capsule_cleanup_capture);
        return Py_BuildValue("IN", result, capsule_capture);
    }

      static PyObject* playback_get_previous_capture(PyObject* self, PyObject *args) {
        int thread_safe;
        PyThreadState *thread_state;
        PyObject *capsule;
        k4a_playback_t* playback_handle;
        k4a_stream_result_t result;

        PyArg_ParseTuple(args, "Op", &capsule, &thread_safe);
        playback_handle = (k4a_playback_t*)PyCapsule_GetPointer(capsule, CAPSULE_PLAYBACK_NAME);

        k4a_capture_t* capture = (k4a_capture_t*) malloc(sizeof(k4a_capture_t));
        if (capture == NULL) {
            fprintf(stderr, "Cannot allocate memory");
            return Py_BuildValue("IN", K4A_RESULT_FAILED, Py_None);
        }

        thread_state = _gil_release(thread_safe);
        result = k4a_playback_get_previous_capture(*playback_handle, capture);
        _gil_restore(thread_state);

        if (result != K4A_STREAM_RESULT_SUCCEEDED) {
            free(capture);
            return Py_BuildValue("IN", result, Py_None);
        }
        PyObject* capsule_capture = PyCapsule_New(capture, CAPSULE_CAPTURE_NAME, capsule_cleanup_capture);
        return Py_BuildValue("IN", result, capsule_capture);
    }

    struct module_state
    {
        PyObject *error;
    };

#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))
    static PyMethodDef Pyk4aMethods[] = {
        {"device_open", device_open, METH_VARARGS, "Open an Azure Kinect device"},
        {"device_start_cameras", device_start_cameras, METH_VARARGS, "Starts color and depth camera capture"},
        {"device_stop_cameras", device_stop_cameras, METH_VARARGS, "Stops the color and depth camera capture"},
        {"device_start_imu", device_start_imu, METH_VARARGS, "Starts imu sernsors"},
        {"device_stop_imu", device_stop_imu, METH_VARARGS, "Stops imu sernsors"},
        {"device_get_capture", device_get_capture, METH_VARARGS, "Reads a sensor capture"},
        {"capture_get_color_image", capture_get_color_image, METH_VARARGS, "Get the color image associated with the given capture"},
        {"capture_get_depth_image", capture_get_depth_image, METH_VARARGS, "Set or add a depth image to the associated capture"},
        {"capture_get_ir_image", capture_get_ir_image, METH_VARARGS, "Set or add a IR image to the associated capture"},
        {"device_get_imu_sample", device_get_imu_sample, METH_VARARGS, "Reads an imu sample"},
        {"device_close", device_close, METH_VARARGS, "Close an Azure Kinect device"},
        {"device_get_sync_jack", device_get_sync_jack, METH_VARARGS, "Get the device jack status for the synchronization in and synchronization out connectors."},
        {"device_get_color_control", device_get_color_control, METH_VARARGS, "Get device color control."},
        {"device_set_color_control", device_set_color_control, METH_VARARGS, "Set device color control."},
        {"device_get_color_control_capabilities", device_get_color_control_capabilities, METH_VARARGS, "Get device color control capabilities."},
        {"device_get_calibration", device_get_calibration, METH_VARARGS, "Get device calibration handle."},
        {"device_get_raw_calibration", device_get_raw_calibration, METH_VARARGS, "Get device calibration in text/json format."},
        {"calibration_get_from_raw", calibration_get_from_raw, METH_VARARGS, "Create new calibration handle from raw json."},
        {"transformation_create", transformation_create, METH_VARARGS, "Create transformation handle from calibration"},
        {"transformation_depth_image_to_color_camera", transformation_depth_image_to_color_camera, METH_VARARGS, "Transforms the depth map into the geometry of the color camera."},
        {"transformation_color_image_to_depth_camera", transformation_color_image_to_depth_camera, METH_VARARGS, "Transforms the color image into the geometry of the depth camera."},
        {"transformation_depth_image_to_point_cloud", transformation_depth_image_to_point_cloud, METH_VARARGS, "Transforms the depth map to a point cloud."},
        {"calibration_3d_to_3d", calibration_3d_to_3d, METH_VARARGS, "Transforms the coordinates between 2 3D systems"},
        {"calibration_2d_to_3d", calibration_2d_to_3d, METH_VARARGS, "Transforms the coordinates between a pixel and a 3D system"},
        {"playback_open", playback_open, METH_VARARGS, "Open file for playback"},
        {"playback_close", playback_close, METH_VARARGS, "Close opened playback"},
        {"playback_get_recording_length_usec", playback_get_recording_length_usec, METH_VARARGS, "Return recording length"},
        {"playback_get_calibration", playback_get_calibration, METH_VARARGS, "Extract calibration and create handle from recording"},
        {"playback_get_raw_calibration", playback_get_raw_calibration, METH_VARARGS, "Extract calibration json from recording"},
        {"playback_seek_timestamp", playback_seek_timestamp, METH_VARARGS, "Seek playback file to specified position"},
        {"playback_get_record_configuration", playback_get_record_configuration, METH_VARARGS, "Extract record configuration"},
        {"playback_get_next_capture", playback_get_next_capture, METH_VARARGS, "Get next capture from playback"},
        {"playback_get_previous_capture", playback_get_previous_capture, METH_VARARGS, "Get previous capture from playback"},

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
