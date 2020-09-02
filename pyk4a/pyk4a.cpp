#include <Python.h>
#include <numpy/arrayobject.h>

#include <k4a/k4a.h>
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
    typedef struct device_container {
        k4a_transformation_t transformation_handle;
        k4a_calibration_t calibration_handle;
        k4a_device_t device;
    } device_container;
    #define MAX_DEVICES 32
    device_container devices[MAX_DEVICES];

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

    static void capsule_cleanup_image(PyObject *capsule) {
        k4a_image_t *image = (k4a_image_t*)PyCapsule_GetContext(capsule);
        k4a_image_release(*image);
        free(image);
    }

    static void capsule_cleanup_capture(PyObject *capsule) {
        k4a_capture_t *capture = (k4a_capture_t*)PyCapsule_GetPointer(capsule, NULL);
        k4a_capture_release(*capture);
    }

    static PyObject* device_open(PyObject* self, PyObject* args){
        uint32_t device_id;
        int thread_safe;
        PyThreadState *thread_state;
        PyArg_ParseTuple(args, "Ip", &device_id, &thread_safe);

        thread_state = _gil_release(thread_safe);
        k4a_result_t result = k4a_device_open(device_id, &devices[device_id].device);
        _gil_restore(thread_state);
        return Py_BuildValue("I", result);
    }

    static PyObject* device_close(PyObject* self, PyObject* args){
        uint32_t device_id;
        int thread_safe;
        PyThreadState *thread_state;
        PyArg_ParseTuple(args, "Ip", &device_id, &thread_safe);
        
        thread_state = _gil_release(thread_safe);
        k4a_device_close(devices[device_id].device);
        _gil_restore(thread_state);
        
        return Py_BuildValue("I", K4A_RESULT_SUCCEEDED);
    }

    static PyObject* device_get_sync_jack(PyObject* self, PyObject* args){
        uint32_t device_id;
        int thread_safe;
        PyThreadState *thread_state;
        bool in_jack = 0;
        bool out_jack = 0;
        PyArg_ParseTuple(args, "Ip", &device_id, &thread_safe);
        
        thread_state = _gil_release(thread_safe);
        k4a_result_t result = k4a_device_get_sync_jack(devices[device_id].device, &in_jack, &out_jack);
        _gil_restore(thread_state);

        return Py_BuildValue("III", result, in_jack, out_jack);
    }

    static PyObject* device_get_color_control(PyObject* self, PyObject* args){
        uint32_t device_id;
        int thread_safe;
        PyThreadState *thread_state;
        k4a_color_control_command_t command;
        k4a_color_control_mode_t mode;
        int32_t value = 0;
        PyArg_ParseTuple(args, "IpI", &device_id, &thread_safe, &command);
        thread_state = _gil_release(thread_safe);
        k4a_result_t result = k4a_device_get_color_control(devices[device_id].device, command, &mode, &value);
        _gil_restore(thread_state);
        if (result == K4A_RESULT_FAILED) {
            return Py_BuildValue("IIi", 0, 0, 0);
        }
        return Py_BuildValue("IIi", result, mode, value);
    }

    static PyObject* device_set_color_control(PyObject* self, PyObject* args){
        uint32_t device_id;
        int thread_safe;
        PyThreadState *thread_state;
        k4a_color_control_command_t command = K4A_COLOR_CONTROL_EXPOSURE_TIME_ABSOLUTE;
        k4a_color_control_mode_t mode = K4A_COLOR_CONTROL_MODE_MANUAL;
        int32_t value = 0;
        PyArg_ParseTuple(args, "IpIIi", &device_id, &thread_safe, &command, &mode, &value);

        thread_state = _gil_release(thread_safe);
        k4a_result_t result = k4a_device_set_color_control(devices[device_id].device, command, mode, value);
        _gil_restore(thread_state);
        if (result == K4A_RESULT_FAILED) {
            return Py_BuildValue("I", K4A_RESULT_FAILED);
        }
        return Py_BuildValue("I", result);
    }

    static PyObject* device_get_color_control_capabilities(PyObject* self, PyObject* args){
        uint32_t device_id;
        int thread_safe;
        PyThreadState *thread_state;
        k4a_color_control_command_t command;
        bool supports_auto;
        int min_value;
        int max_value;
        int step_value;
        int default_value;
        k4a_color_control_mode_t default_mode;
        PyArg_ParseTuple(args, "IpI", &device_id, &thread_safe, &command);

        thread_state = _gil_release(thread_safe);
        k4a_result_t result = k4a_device_get_color_control_capabilities(devices[device_id].device, command, &supports_auto, &min_value, &max_value, &step_value, &default_value, &default_mode);
        _gil_restore(thread_state);
        if (result == K4A_RESULT_FAILED) {
            return Py_BuildValue("I(0)", result, Py_None);
        }
        return Py_BuildValue("I{s:I,s:I,s:O,s:i,s:i,s:i,s:I}", result,
                "color_control_command", command,
                "supports_auto", supports_auto ? Py_True: Py_False,
                "min_value", min_value,
                "max_value", max_value,
                "step_value", step_value,
                "default_value", default_value,
                "default_mode", default_mode);
    }

    static PyObject* device_start_cameras(PyObject* self, PyObject* args){
        uint32_t device_id;
        int thread_safe;
        PyThreadState *thread_state;
        k4a_device_configuration_t config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
        PyArg_ParseTuple(args, "IpIIIIpiIIp", &device_id, &thread_safe,
                &config.color_format,
                &config.color_resolution, &config.depth_mode,
                &config.camera_fps, &config.synchronized_images_only,
                &config.depth_delay_off_color_usec, &config.wired_sync_mode,
                &config.subordinate_delay_off_master_usec,
                &config.disable_streaming_indicator);

        k4a_result_t result;
        thread_state = _gil_release(thread_safe);
        result = k4a_device_get_calibration(devices[device_id].device, config.depth_mode, config.color_resolution, &devices[device_id].calibration_handle);
        if (result == K4A_RESULT_FAILED) {
            _gil_restore(thread_state);
            return Py_BuildValue("I", K4A_RESULT_FAILED);
        }
        devices[device_id].transformation_handle = k4a_transformation_create(&devices[device_id].calibration_handle);
        if (devices[device_id].transformation_handle == NULL) {
            _gil_restore(thread_state);
            return Py_BuildValue("I", K4A_RESULT_FAILED);
        }
        result = k4a_device_start_cameras(devices[device_id].device, &config);
        _gil_restore(thread_state);
        return Py_BuildValue("I", result);
    }
    
    static PyObject* device_start_imu(PyObject* self, PyObject* args){
        uint32_t device_id;
        int thread_safe;
        PyThreadState *thread_state;
        k4a_result_t result;
        PyArg_ParseTuple(args, "Ip", &device_id, &thread_safe);
        thread_state = _gil_release(thread_safe);
        result = k4a_device_start_imu(devices[device_id].device);
        _gil_restore(thread_state);
        return Py_BuildValue("I", result);
    }

    static PyObject* device_stop_cameras(PyObject* self, PyObject* args){
        uint32_t device_id;
        int thread_safe;
        PyThreadState *thread_state;
        PyArg_ParseTuple(args, "Ip", &device_id, &thread_safe);
        thread_state = _gil_release(thread_safe);
        if (devices[device_id].transformation_handle) {
            k4a_transformation_destroy(devices[device_id].transformation_handle);
        }
        k4a_device_stop_cameras(devices[device_id].device);

        _gil_restore(thread_state);
        return Py_BuildValue("I", K4A_RESULT_SUCCEEDED);
    }
    
    static PyObject* device_stop_imu(PyObject* self, PyObject* args){
        uint32_t device_id;
        int thread_safe;
        PyThreadState *thread_state;
        PyArg_ParseTuple(args, "Ip", &device_id, &thread_safe);
        thread_state = _gil_release(thread_safe);
        k4a_device_stop_imu(devices[device_id].device);

        _gil_restore(thread_state);
        return Py_BuildValue("I", K4A_RESULT_SUCCEEDED);
    }

    static PyObject* device_get_capture(PyObject* self, PyObject* args){
        uint32_t device_id;
        int thread_safe;
        PyThreadState *thread_state;
        long long timeout;
        PyArg_ParseTuple(args, "IpL", &device_id, &thread_safe, &timeout);
        k4a_capture_t* capture = (k4a_capture_t*) malloc(sizeof(k4a_capture_t));
        k4a_capture_create(capture);
        PyObject* capsule_capture = PyCapsule_New(capture, NULL, capsule_cleanup_capture);
        k4a_wait_result_t result;
        thread_state = _gil_release(thread_safe);
        result = k4a_device_get_capture(devices[device_id].device, capture, timeout);
        _gil_restore(thread_state);

        return Py_BuildValue("IN", result, capsule_capture);
    }
    
    static PyObject* device_get_imu_sample(PyObject* self, PyObject* args){
        uint32_t device_id;
        int thread_safe;
        PyThreadState *thread_state;
        long long timeout;
        PyArg_ParseTuple(args, "IpL", &device_id, &thread_safe, &timeout);
        
        k4a_imu_sample_t imu_sample;
        k4a_wait_result_t result;
        
        thread_state = _gil_release(thread_safe);
        result = k4a_device_get_imu_sample(devices[device_id].device, &imu_sample, timeout);
        
        _gil_restore(thread_state);
        if (K4A_WAIT_RESULT_SUCCEEDED == result) {
            return Py_BuildValue("I{s:f,s:(fff),s:L,s:(fff),s:L}", result,
                    "temperature", imu_sample.temperature,
                    "acc_sample", imu_sample.acc_sample.xyz.x, imu_sample.acc_sample.xyz.y, imu_sample.acc_sample.xyz.z,
                    "acc_timestamp", imu_sample.acc_timestamp_usec,
                    "gyro_sample", imu_sample.gyro_sample.xyz.x, imu_sample.gyro_sample.xyz.y, imu_sample.gyro_sample.xyz.z,
                    "gyro_timestamp", imu_sample.gyro_timestamp_usec);
        }

        return Py_BuildValue("I(0)", result, Py_None);
    }

    static PyObject* calibration_set_from_raw(PyObject* self, PyObject* args){
        uint32_t device_id;
        int thread_safe;
        PyThreadState *thread_state;
        char * raw_calibration;
        k4a_device_configuration_t config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
        PyArg_ParseTuple(args, "IpsIIIIpiIIp", &device_id, &thread_safe,
                &raw_calibration, &config.color_format,
                &config.color_resolution, &config.depth_mode,
                &config.camera_fps, &config.synchronized_images_only,
                &config.depth_delay_off_color_usec, &config.wired_sync_mode,
                &config.subordinate_delay_off_master_usec,
                &config.disable_streaming_indicator);
        size_t raw_calibration_size = strlen(raw_calibration) + 1;
        k4a_result_t result;

        thread_state = _gil_release(thread_safe);
        result = k4a_calibration_get_from_raw(raw_calibration,
                raw_calibration_size, config.depth_mode,
                config.color_resolution, &devices[device_id].calibration_handle);
        if (result == K4A_RESULT_FAILED) {
            _gil_restore(thread_state);
            return Py_BuildValue("I", K4A_RESULT_FAILED);
        }
        if (devices[device_id].transformation_handle) {
            k4a_transformation_destroy(devices[device_id].transformation_handle);
        }
        devices[device_id].transformation_handle = k4a_transformation_create(&devices[device_id].calibration_handle);
        _gil_restore(thread_state);
        return Py_BuildValue("I", K4A_RESULT_SUCCEEDED);
    }

    static PyObject* device_get_calibration(PyObject* self, PyObject* args){
        uint32_t device_id;
        int thread_safe;
        PyThreadState *thread_state;
        k4a_buffer_result_t result;
        size_t data_size;
        PyArg_ParseTuple(args, "Ip", &device_id, &thread_safe);
        thread_state = _gil_release(thread_safe);
        result = k4a_device_get_raw_calibration(devices[device_id].device, NULL, &data_size);
        if (result == K4A_BUFFER_RESULT_FAILED) {
            _gil_restore(thread_state);
            return Py_BuildValue("");
        }
        uint8_t* data = (uint8_t*) malloc(data_size);
        result = k4a_device_get_raw_calibration(devices[device_id].device, data, &data_size);
        _gil_restore(thread_state);
        if (result == K4A_BUFFER_RESULT_FAILED) {
            return Py_BuildValue("");
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

    static PyObject* transformation_depth_image_to_color_camera(PyObject* self, PyObject* args){
        uint32_t device_id;
        int thread_safe;
        PyThreadState *thread_state;
        k4a_result_t res;
        PyArrayObject *in_array;
        k4a_color_resolution_t color_resolution;
        PyArg_ParseTuple(args, "IpO!I", &device_id, &thread_safe, &PyArray_Type, &in_array, &color_resolution);

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
                    devices[device_id].transformation_handle,
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

    static PyObject* transformation_color_image_to_depth_camera(
            PyObject* self, PyObject* args){
        uint32_t device_id;
        int thread_safe;
        PyThreadState *thread_state;
        k4a_result_t res;
        PyArrayObject *in_depth_array;
        PyArrayObject *in_color_array;
        PyArg_ParseTuple(args, "IpO!O!", &device_id, &thread_safe, &PyArray_Type, &in_depth_array, &PyArray_Type, &in_color_array);

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
                    devices[device_id].transformation_handle,
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
        int thread_safe;
        PyThreadState *thread_state;
        PyObject *capsule_capture;
        k4a_capture_t *capture;
        k4a_result_t res;
        PyArg_ParseTuple(args, "pO", &thread_safe, &capsule_capture);


        capture = (k4a_capture_t*)PyCapsule_GetPointer(capsule_capture, NULL);
        k4a_image_t* color_image = (k4a_image_t*) malloc(sizeof(k4a_image_t));
        thread_state = _gil_release(thread_safe);
        *color_image = k4a_capture_get_color_image(*capture);
        _gil_restore(thread_state);
        PyArrayObject* np_color_image;
        if (color_image) {
            res = k4a_image_to_numpy(color_image, &np_color_image);
        }

        if (K4A_RESULT_SUCCEEDED == res) {
            return PyArray_Return(np_color_image);
        }
        else {
            free(color_image);
            return Py_BuildValue("");
        }
    }

    static PyObject* capture_get_depth_image(PyObject* self, PyObject* args){
        int thread_safe;
        PyThreadState *thread_state;
        PyObject *capsule_capture;
        k4a_capture_t *capture;
        k4a_result_t res;
        PyArg_ParseTuple(args, "pO", &thread_safe, &capsule_capture);

        capture = (k4a_capture_t*)PyCapsule_GetPointer(capsule_capture, NULL);
        thread_state = _gil_release(thread_safe);
        k4a_image_t* depth_image = (k4a_image_t*) malloc(sizeof(k4a_image_t));
        *depth_image = k4a_capture_get_depth_image(*capture);
        _gil_restore(thread_state);
        PyArrayObject* np_depth_image;
        if (depth_image) {
            res = k4a_image_to_numpy(depth_image, &np_depth_image);
        }
        if (K4A_RESULT_SUCCEEDED == res) {
            return PyArray_Return(np_depth_image);
        }
        else {
            free(depth_image);
            return Py_BuildValue("");
        }
    }

    static PyObject* capture_get_ir_image(PyObject* self, PyObject* args){
        int thread_safe;
        PyThreadState *thread_state;
        PyObject *capsule_capture;
        k4a_capture_t *capture;
        k4a_result_t res;
        PyArg_ParseTuple(args, "pO", &thread_safe, &capsule_capture);

        capture = (k4a_capture_t*)PyCapsule_GetPointer(capsule_capture, NULL);
        thread_state = _gil_release(thread_safe);
        k4a_image_t* ir_image = (k4a_image_t*) malloc(sizeof(k4a_image_t));
        *ir_image = k4a_capture_get_ir_image(*capture);
        _gil_restore(thread_state);
        PyArrayObject* np_ir_image;
        if (ir_image) {
            res = k4a_image_to_numpy(ir_image, &np_ir_image);
        }

        if (K4A_RESULT_SUCCEEDED == res) {
            return PyArray_Return(np_ir_image);
        }
        else {
            free(ir_image);
            return Py_BuildValue("");
        }
    }

    static PyObject* calibration_3d_to_3d(PyObject* self, PyObject *args){
        uint32_t device_id;
        int thread_safe;
        PyThreadState *thread_state;
        k4a_result_t res;
        k4a_float3_t source_point3d_mm;
        k4a_float3_t target_point3d_mm;
        k4a_calibration_type_t source_camera;
        k4a_calibration_type_t target_camera;
        int source_point_x;
        int source_point_y;
        int source_point_z;

        PyArg_ParseTuple(args, "IpiiiII",
                &device_id,
                &thread_safe,
                &source_point_x,
                &source_point_y,
                &source_point_z,
                &source_camera,
                &target_camera);

        thread_state = _gil_release(thread_safe);
        source_point3d_mm.xyz.x = source_point_x;
        source_point3d_mm.xyz.y = source_point_y;
        source_point3d_mm.xyz.z = source_point_z;


        res = k4a_calibration_3d_to_3d (&devices[device_id].calibration_handle,
                                        &source_point3d_mm,
                                        source_camera, 
                                        target_camera,
                                        &target_point3d_mm);
       _gil_restore(thread_state);
        if (res == K4A_RESULT_FAILED ) {
            return Py_BuildValue("I", K4A_RESULT_FAILED);
        }
        return Py_BuildValue("Ifff", res, target_point3d_mm.xyz.x, target_point3d_mm.xyz.y, target_point3d_mm.xyz.z);
    }

    static PyObject* calibration_2d_to_3d(PyObject* self, PyObject *args){
        uint32_t device_id;
        int thread_safe;
        PyThreadState *thread_state;
        int source_point_x;
        int source_point_y;
        float source_depth_mm;
        int valid;
        k4a_calibration_type_t source_camera;
        k4a_calibration_type_t target_camera;
        k4a_result_t res;
        k4a_float2_t source_point2d;
        k4a_float3_t target_point3d_mm;
        
        PyArg_ParseTuple(args, "IpiifII",
                &device_id,
                &thread_safe,
                &source_point_x,
                &source_point_y,
                &source_depth_mm,
                &source_camera,
                &target_camera);

        thread_state = _gil_release(thread_safe);
        source_point2d.xy.x = source_point_x;
        source_point2d.xy.y = source_point_y;


        res = k4a_calibration_2d_to_3d (&devices[device_id].calibration_handle,
                                        &source_point2d,
                                        source_depth_mm,
                                        source_camera, 
                                        target_camera,
                                        &target_point3d_mm,
                                        &valid);
        _gil_restore(thread_state);
        if (res == K4A_RESULT_FAILED ) {
            return Py_BuildValue("I", K4A_RESULT_FAILED);
        }
        // Return object...
        return Py_BuildValue("IIfff", res, valid, target_point3d_mm.xyz.x, target_point3d_mm.xyz.y, target_point3d_mm.xyz.z);
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
        {"device_get_calibration", device_get_calibration, METH_VARARGS, "Get device calibration in json format."},
        {"calibration_set_from_raw", calibration_set_from_raw, METH_VARARGS, "Temporary set the calibration from a json format. Must be called after device_start_cameras."},
        {"transformation_depth_image_to_color_camera", transformation_depth_image_to_color_camera, METH_VARARGS, "Transforms the depth map into the geometry of the color camera."},
        {"transformation_color_image_to_depth_camera", transformation_color_image_to_depth_camera, METH_VARARGS, "Transforms the color image into the geometry of the depth camera."},
        {"calibration_3d_to_3d", calibration_3d_to_3d, METH_VARARGS, "Transforms the coordinates between 2 3D systems"},
        {"calibration_2d_to_3d", calibration_2d_to_3d, METH_VARARGS, "Transforms the coordinates between a pixel and a 3D system"},
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
