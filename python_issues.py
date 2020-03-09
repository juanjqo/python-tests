import unittest

from dqrobotics import *
from math import *
import numpy as np


class DQTestCase(unittest.TestCase):

    # DQ Multiplication having wrong precision
    def test_python_issue_17(self):
        phi_1 = pi / 4.0  # Set the rotation angle
        n_1 = DQ(0, 1, 0, 0, 0, 0, 0, 0)  # Set the rotation axis
        t_1 = DQ(0, 1, 2, 3, 0, 0, 0, 0)  # Set the translation

        r = cos(phi_1 / 2.0) + n_1 * sin(phi_1 / 2.0)
        x = r + 0.5 * E_ * t_1 * r

        np.set_printoptions(precision=12)
        print("x = {}".format(x.q))
        print("is_unit(x) = {}".format(is_unit(x)))
        print("translation(x) = {}".format(translation(x)))


if __name__ == '__main__':
    unittest.main()
