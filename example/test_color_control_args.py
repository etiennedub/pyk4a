import pyk4a
from pyk4a.config import ColorControlCommand

k4a = pyk4a.PyK4A(debug_color_control_setter_args=True)
k4a.connect()

for cmd in ColorControlCommand:
    print(cmd)
    capabilities = k4a._get_color_control_capabilities(cmd=cmd)
    print(capabilities)


# test setting a bunch of config values
k4a.gain =


k4a.disconnect()

