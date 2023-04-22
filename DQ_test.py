"""(C) Copyright 2019 DQ Robotics Developers

This file is part of DQ Robotics.

    DQ Robotics is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    DQ Robotics is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with DQ Robotics.  If not, see <http://www.gnu.org/licenses/>.

Contributors:
- Murilo M. Marinho (murilo@nml.t.u-tokyo.ac.jp)
  Responsible for the original implementation.

- Juan Jose Quiroz Omana (juanjqo@g.ecc.u-tokyo.ac.jp)
  Added some modifications discussed at #6 (https://github.com/dqrobotics/python-tests/pull/6)
        - Added tests for Q4 and Q8. 
        - Replaced 12 by a threshold related to DQ_threshold in all tests.
        - Added a relaxed threshold to ad and adsharp tests.
"""

import math
import unittest
import scipy.io
import numpy
from dqrobotics import *
from DQ_test_facilities import get_list_of_dq_from_mat
from DQ_test_facilities import get_list_of_matrices_from_mat

mat = scipy.io.loadmat('DQ_test_06_04_2023.mat')

dq_a_list = get_list_of_dq_from_mat('random_dq_a', mat)
dq_b_list = get_list_of_dq_from_mat('random_dq_b', mat)

# Threshold related to DQ_threshold
threshold = -math.log10(DQ_threshold)

# Relaxed threshold used in some specific tests
relaxed_threshold = threshold-1

class DQTestCase(unittest.TestCase):
    global mat
    global dq_a_list
    global dq_b_list

    # Constructors
    def test_constructor_valid(self):
        # Eight
        self.assertEqual(DQ([1, 2, 3, 4, 5, 6, 7, 8]),
                         DQ(1, 2, 3, 4, 5, 6, 7, 8),
                         "Incorrect 8 elements constructor")
        # Six
        self.assertEqual(DQ([1, 2, 3, 4, 5, 6]),
                         DQ(0, 1, 2, 3, 0, 4, 5, 6),
                         "Incorrect 6 elements constructor")

        # Three
        self.assertEqual(DQ([1, 2, 3]),
                         DQ(0, 1, 2, 3, 0, 0, 0, 0),
                         "Incorrect 3 elements constructor")

        # Four
        self.assertEqual(DQ([1, 2, 3, 4]),
                         DQ(1, 2, 3, 4, 0, 0, 0, 0),
                         "Incorrect 4 elements constructor")

        # One
        self.assertEqual(DQ([1]),
                         DQ(1, 0, 0, 0, 0, 0, 0, 0),
                         "Incorrect 1 element constructor")

    def test_constructor_invalid(self):
        # Two
        with self.assertRaises(ValueError):
            DQ([1, 2])
        # Five
        with self.assertRaises(ValueError):
            DQ([1, 2, 3, 4, 5])
        # Seven
        with self.assertRaises(ValueError):
            DQ([1, 2, 3, 4, 5, 6, 7])

    
    def test_cross(self):
        result_of_cross = get_list_of_dq_from_mat('result_of_cross', mat)
        for a, b, c in zip(dq_a_list, dq_b_list, result_of_cross):
            self.assertEqual(cross(a, b), c, "Error in cross")

    def test_ad(self):
        result_of_Ad = get_list_of_dq_from_mat('result_of_Ad', mat)
        for a, b, c in zip(dq_a_list, dq_b_list, result_of_Ad):
            numpy.testing.assert_almost_equal(vec8(Ad(a, b)), vec8(c), relaxed_threshold, "Error in Ad")

    def test_adsharp(self):
        result_of_Adsharp = get_list_of_dq_from_mat('result_of_Adsharp', mat)
        for a, b, c in zip(dq_a_list, dq_b_list, result_of_Adsharp):
            numpy.testing.assert_almost_equal(vec8(Adsharp(a, b)), vec8(c), relaxed_threshold, "Error in Adsharp")

    # Unary operators
    def test_conj(self):
        result_of_conj = get_list_of_dq_from_mat('result_of_conj', mat)
        for a, c in zip(dq_a_list, result_of_conj):
            self.assertEqual(conj(a), c, "Error in conj")

    def test_sharp(self):
        result_of_sharp = get_list_of_dq_from_mat('result_of_sharp', mat)
        for a, c in zip(dq_a_list, result_of_sharp):
            self.assertEqual(sharp(a), c, "Error in sharp")

    def test_normalize(self):
        result_of_normalize = get_list_of_dq_from_mat('result_of_normalize', mat)
        for a, c in zip(dq_a_list, result_of_normalize):
            self.assertEqual(normalize(a), c, "Error in normalize")

    def test_of_translation(self):
        result_of_translation = get_list_of_dq_from_mat('result_of_translation', mat)
        for a, c in zip(dq_a_list, result_of_translation):
            self.assertEqual(translation(normalize(a)), c, "Error in translation")

    def test_of_rotation(self):
        result_of_rotation = get_list_of_dq_from_mat('result_of_rotation', mat)
        for a, c in zip(dq_a_list, result_of_rotation):
            self.assertEqual(rotation(normalize(a)), c, "Error in rotation")

    def test_of_log(self):
        result_of_log = get_list_of_dq_from_mat('result_of_log', mat)
        for a, c in zip(dq_a_list, result_of_log):
            self.assertEqual(log(normalize(a)), c, "Error in log")

    def test_of_exp(self):
        result_of_exp = get_list_of_dq_from_mat('result_of_exp', mat)
        for a, c in zip(dq_a_list, result_of_exp):
            self.assertEqual(exp(DQ(vec6(a))), c, "Error in exp")

    def test_of_rotation_axis(self):
        result_of_rotation_axis = get_list_of_dq_from_mat('result_of_rotation_axis', mat)
        for a, c in zip(dq_a_list, result_of_rotation_axis):
            self.assertEqual(rotation_axis(normalize(a)), c, "Error in rotation_axis")

    def test_of_rotation_angle(self):
        result_of_rotation_angle = get_list_of_dq_from_mat('result_of_rotation_angle', mat)
        for a, c in zip(dq_a_list, result_of_rotation_angle):
            self.assertEqual(DQ([rotation_angle(normalize(a))]), c, "Error in rotation_angle")

    def test_of_Q4(self):
        result_of_Q4 = get_list_of_matrices_from_mat('result_of_Q4', mat)
        dq_a_list_Q4 = get_list_of_dq_from_mat('random_dq_a', mat)
        for a, c in zip(dq_a_list_Q4, result_of_Q4):
            numpy.testing.assert_almost_equal(Q4(normalize(P(a))), c, threshold, "Error in Q8")

    def test_of_Q8(self):
        result_of_Q8 = get_list_of_matrices_from_mat('result_of_Q8', mat)
        dq_a_list_Q8 = get_list_of_dq_from_mat('random_dq_a', mat)
        for a, c in zip(dq_a_list_Q8, result_of_Q8):
            numpy.testing.assert_almost_equal(Q8(normalize(a)), c, threshold, "Error in Q8")

if __name__ == '__main__':
    unittest.main()
