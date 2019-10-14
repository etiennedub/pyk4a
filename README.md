# pyk4a
This library is a simple wrapper in Python 3 for Azure-Kinect-Sensor-SDK. 

## Prerequisites
The [Azure-Kinect-Sensor-SDK](https://github.com/microsoft/Azure-Kinect-Sensor-SDK) is required to build this library.
To use the SDK, refer to the installation instructions [here](https://github.com/microsoft/Azure-Kinect-Sensor-SDK).


## Install

### Linux

Make sure your `LD_LIBRARY_PATH` contains the directory of k4a.lib

```
pip install pyk4a
```

### Windows

Make sure you replace the paths in the following instructions with your own k4a sdk path.

```
pip install pyk4a --global-option=build_ext --global-option="-IC:\Program Files\Azure Kinect SDK v1.2.0\sdk\include" --global-option="-LC:\Program Files\Azure Kinect SDK v1.2.0\sdk\windows-desktop\amd64\release\lib"
```

Don't forget to add the folder containing the release `k4a.dll` to your Path env variable `C:\Program Files\Azure Kinect SDK v1.2.0\sdk\windows-desktop\amd64\release\bin`

## Example

For a basic example displaying the first frame, you can run this code:

```
from pyk4a import PyK4A

# Load camera with the default config
k4a = PyK4A()
k4a.connect()

# Get the next color frame without the depth (blocking function)
img_color = k4a.get_capture(color_only=True)

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

No documentation is available but most methods are used in the example. You can follow it as reference.
You can also check directly the code of the main class [PyK4A](https://github.com/etiennedub/pyk4a/blob/master/pyk4a/pyk4a.py).

## Contribution
If a methods is not implemented, feel free to send a pull request.
