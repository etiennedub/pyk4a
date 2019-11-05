import unittest
import pyk4a
from pyk4a.config import ColorControlCommand
from pyk4a.pyk4a import K4AValueException


class TestColorControlArgs(unittest.TestCase):
    _k4a = None

    @property
    def k4a(self):
        if self._k4a is None:
            self._k4a = pyk4a.PyK4A(debug_color_control_setter_args=True)
            self._k4a.connect()
        return self._k4a

    def test_gain_positive(self):
        self.k4a.gain = 12
        self.assertEqual(self.k4a.gain, 12)

    def test_gain_negative(self):
        with self.assertRaises(K4AValueException):
            self.k4a.gain = -1

    def test_gain_too_high(self):
        with self.assertRaises(K4AValueException):
            self.k4a.gain = 500


if __name__ == '__main__':
    k4a = pyk4a.PyK4A(debug_color_control_setter_args=True)
    # k4a = pyk4a.PyK4A()
    k4a.connect()
    from pprint import pprint
    for cmd in ColorControlCommand:
        capability = k4a._get_color_control_capabilities(cmd)
        pprint(capability)

    k4a.gain = 500
    k4a.disconnect()

    # unittest.main()
