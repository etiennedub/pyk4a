# pyk4a

![CI](https://github.com/etiennedub/pyk4a/workflows/CI/badge.svg)
[![codecov](https://codecov.io/gh/etiennedub/pyk4a/branch/master/graph/badge.svg)](https://codecov.io/gh/etiennedub/pyk4a)

![pyk4a](https://github.com/etiennedub/pyk4a/raw/master/figs/pyk4a_logo.png) 


This library is a simple and pythonic wrapper in Python 3 for the Azure-Kinect-Sensor-SDK.

Images are returned as numpy arrays and behave like python objects.

This approach incurs almost no overhead in terms of CPU, memory or other resources.
It also simplifies usage. Kinect C api image buffers are directly reused and image releases are performed automatically by the python garbage collector.

Homepage: https://github.com/etiennedub/pyk4a/

## Prerequisites
The [Azure-Kinect-Sensor-SDK](https://github.com/microsoft/Azure-Kinect-Sensor-SDK) is required to build this library.
To use the SDK, refer to the installation instructions [here](https://github.com/microsoft/Azure-Kinect-Sensor-SDK).


## Install

### Linux

Make sure your `LD_LIBRARY_PATH` contains the directory of k4a.lib

```shell
pip install pyk4a
```

### Windows

In most cases `pip install pyk4a` is enough to install this package.

When using an anaconda environment, you need to set the environment variable `CONDA_DLL_SEARCH_MODIFICATION_ENABLE=1` https://github.com/conda/conda/issues/10897

Because of the numerous issues received from Windows users, the installer ([setup.py](setup.py)) automatically detects the kinect SDK path.

When the installer is not able to find the path, the following snippet can help.
Make sure you replace the paths in these instructions with your own kinect SDK path. It is important to replace 1.4.1 with your installed version of the SDK.
```shell
pip install pyk4a --no-use-pep517 --global-option=build_ext --global-option="-IC:\Program Files\Azure Kinect SDK v1.4.1\sdk\include" --global-option="-LC:\Program Files\Azure Kinect SDK v1.4.1\sdk\windows-desktop\amd64\release\lib"
```

During execution, `k4a.dll` is required. The automatic detection should be able to find this file.
It is also possible to specify the DLL's directory with the environment variable `K4A_DLL_DIR`.
If `K4A_DLL_DIR` is used, the automatic DLL search is not performed.

## Example

For a basic example displaying the first frame, you can run this code:

```
from pyk4a import PyK4A

# Load camera with the default config
k4a = PyK4A()
k4a.start()

# Get the next capture (blocking function)
capture = k4a.get_capture()
img_color = capture.color

# Display with pyplot
from matplotlib import pyplot as plt
plt.imshow(img_color[:, :, 2::-1]) # BGRA to RGB
plt.show()
```

Otherwise, a more avanced example is available in the [example](https://github.com/etiennedub/pyk4a/tree/master/example) folder.
To execute it [opencv-python](https://github.com/skvark/opencv-python) is required.
```
git clone https://github.com/etiennedub/pyk4a.git
cd pyk4a/example
python viewer.py
```

## Documentation

No documentation is available but all functinos are properly [type hinted](https://docs.python.org/3/library/typing.html).
The code of the main class is a good place to start[PyK4A](https://github.com/etiennedub/pyk4a/blob/master/pyk4a/pyk4a.py).

You can also follow the various [example folder](example) scripts as reference.


## Bug Reports
Submit an issue and please include as much details as possible.

Make sure to use the search function on closed issues, especially if your problem is related to installing on [windows](https://github.com/etiennedub/pyk4a/issues?q=windows+).


## Module Development

1) Install required packages: `make setup`

2) Install local pyk4a version (compiles pyk4a.cpp): `make build`

## Contribution

Feel free to send pull requests. The develop branch should be used.

Please rebuild, format, check code quality and run tests before submitting a pull request:
```shell script
make build
make fmt lint
make test
```

Note: you need `clang-format` tool(v 11.0+) for formatting CPP code. 
