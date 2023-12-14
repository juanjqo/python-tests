"""(C) Copyright 2019-2023 DQ Robotics Developers

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
1. Murilo M. Marinho (murilomarinho@ieee.org)
   - Responsible for the original implementation.
2. Juan Jose Quiroz Omana (juanjqogm@gmail.com)
   - Added tests for the DQ_SerialWholeBody class.
"""

import unittest
import scipy.io
import numpy
from dqrobotics.robots import KukaLw4Robot, KukaYoubotRobot, FrankaEmikaPandaRobot
from dqrobotics import *
from DQ_test_facilities import *
from dqrobotics.robot_modeling import DQ_SerialWholeBody

# The data from MATLAB
mat = scipy.io.loadmat('DQ_Kinematics_test.mat')
# A list of random DQs
h_list = get_list_of_dq_from_mat('random_dq_a', mat)
# A list of random joint values
serial_manipulator_q_list = get_list_of_vector_from_mat('random_q', mat)

# lists for the DQ_SerialWholeBody class
serial_whole_body_q_list    = get_list_of_vector_from_mat('random_q_serial_whole_body', mat)
serial_whole_body_pose_list = get_list_of_dq_from_mat('result_of_serial_whole_body_fkm', mat)
serial_whole_body_raw_pose_list = get_list_of_dq_from_mat('result_of_serial_whole_body_raw_fkm', mat)
serial_whole_body_pose_jacobian_list = get_list_of_matrices_from_mat('result_of_serial_whole_body_jacobian', mat)

# A list of the result of fkm() for the list of random joint values
serial_manipulator_pose_list = get_list_of_dq_from_mat('result_of_fkm', mat)

# A list of the result of pose_jacobian() for the list of random joint values
serial_manipulator_pose_jacobian_list = get_list_of_matrices_from_mat('result_of_pose_jacobian', mat)

# A list of translation_jacobian() for the list of random joint values
translation_jacobian_list = get_list_of_matrices_from_mat('result_of_translation_jacobian', mat)

line_jacobian_list = get_list_of_matrices_from_mat('result_of_line_jacobian', mat)
plane_jacobian_list = get_list_of_matrices_from_mat('result_of_plane_jacobian', mat)

# The DQ_SerialManipulator used to calculate everything for DQ_Kinematics as well
serial_manipulator_robot = KukaLw4Robot.kinematics()

# The DQ_SerialWholeBody robot used to perform the tests
serial_whole_body_robot = DQ_SerialWholeBody(serial_manipulator_robot)
serial_whole_body_robot.set_reference_frame(1 + 0.5*E_*(0.5*k_))
serial_whole_body_robot.add(FrankaEmikaPandaRobot.kinematics())

class DQTestCase(unittest.TestCase):
    global mat
    global h_list
    global serial_manipulator_q_list
    global serial_manipulator_pose_list
    global serial_manipulator_pose_jacobian_list
    global serial_whole_body_q_list
    global serial_whole_body_raw_pose_list
    global serial_whole_body_pose_list
    global serial_whole_body_pose_jacobian_list
    global translation_jacobian_list
    global line_jacobian_list
    global plane_jacobian_list
    global serial_manipulator_robot
    global serial_whole_body_robot

    # DQ_SerialManipulator.fkm
    def test_serial_manipulator_fkm(self):
        for q, x in zip(serial_manipulator_q_list, serial_manipulator_pose_list):
            self.assertEqual(serial_manipulator_robot.fkm(q), x, "Error in DQ_SerialManipulator.fkm")

    # DQ_SerialManipulator.pose_jacobian
    def test_serial_manipulator_pose_jacobian(self):
        for q, J in zip(serial_manipulator_q_list, serial_manipulator_pose_jacobian_list):
            numpy.testing.assert_almost_equal(serial_manipulator_robot.pose_jacobian(q), J, 12, "Error in DQ_SerialManipultor.pose_jacobian")


    def test_serial_manipulator_pose_jacobian_derivative(self):
        serial_manipulator_q_dot_list = get_list_of_vector_from_mat('random_q_dot', mat)
        serial_manipulator_pose_jacobian_derivative_list = get_list_of_matrices_from_mat('result_of_pose_jacobian_derivative', mat)
        for q, q_dot, J_dot in zip(serial_manipulator_q_list, serial_manipulator_q_dot_list, serial_manipulator_pose_jacobian_derivative_list):
            numpy.testing.assert_almost_equal(serial_manipulator_robot.pose_jacobian_derivative(q, q_dot), J_dot, 12,
                                              "Error in DQ_SerialManipultor.pose_jacobian_derivative")

    # DQ_WholeBody.fkm
    def test_whole_body_fkm(self):
        whole_body_robot = KukaYoubotRobot.kinematics()
        whole_body_q_list = get_list_of_vector_from_mat('random_q_whole_body', mat)
        whole_body_pose_list = get_list_of_dq_from_mat('result_of_whole_body_fkm', mat)
        for q, x in zip(whole_body_q_list, whole_body_pose_list):
            self.assertEqual(whole_body_robot.fkm(q), x, "Error in DQ_WholeBody.fkm")

    # DQ_WholeBody.pose_jacobian
    def test_whole_body_pose_jacobian(self):
        whole_body_robot = KukaYoubotRobot.kinematics()
        whole_body_q_list = get_list_of_vector_from_mat('random_q_whole_body', mat)
        whole_body_pose_jacobian_list = get_list_of_matrices_from_mat('result_of_whole_body_jacobian', mat)
        for q, J in zip(whole_body_q_list, whole_body_pose_jacobian_list):
            numpy.testing.assert_almost_equal(whole_body_robot.pose_jacobian(q), J, 12, "Error in DQ_WholeBody.pose_jacobian")

    # DQ_SerialWholeBody.fkm
    def test_serial_whole_body_fkm(self):
        for q, x in zip(serial_whole_body_q_list, serial_whole_body_pose_list):
            self.assertEqual(serial_whole_body_robot.fkm(q), x, "Error in DQ_SerialWholeBody.fkm")

    # DQ_SerialWholeBody.raw_fkm
    def test_serial_whole_body_raw_fkm(self):
        for q, x in zip(serial_whole_body_q_list, serial_whole_body_raw_pose_list):
            self.assertEqual(serial_whole_body_robot.raw_fkm(q), x, "Error in DQ_SerialWholeBody.raw_fkm")

    def test_serial_whole_body_pose_jacobian(self):
        for q, J in zip(serial_whole_body_q_list, serial_whole_body_pose_jacobian_list):
            numpy.testing.assert_almost_equal(serial_whole_body_robot.pose_jacobian(q), J, 12, "Error in DQ_SerialWholeBody.pose_jacobian")


    def test_distance_jacobian(self):
        distance_jacobian_list = get_list_of_matrices_from_mat('result_of_distance_jacobian', mat)
        for J, x, distance_jacobian in zip(serial_manipulator_pose_jacobian_list, serial_manipulator_pose_list, distance_jacobian_list):
            numpy.testing.assert_almost_equal(serial_manipulator_robot.distance_jacobian(J, x), distance_jacobian, 12,
                                              "Error in distance_jacobian")

    def test_rotation_jacobian(self):
        rotation_jacobian_list = get_list_of_matrices_from_mat('result_of_rotation_jacobian', mat)
        for J, rotation_jacobian in zip(serial_manipulator_pose_jacobian_list, rotation_jacobian_list):
            numpy.testing.assert_almost_equal(serial_manipulator_robot.rotation_jacobian(J), rotation_jacobian, 12,
                                              "Error in rotation_jacobian")

    def test_translation_jacobian(self):
        for J, x, Jt in zip(serial_manipulator_pose_jacobian_list, serial_manipulator_pose_list, translation_jacobian_list):
            numpy.testing.assert_almost_equal(serial_manipulator_robot.translation_jacobian(J, x), Jt, 12, "Error in translation_jacobian")

    def test_line_jacobian(self):
        for J, x, Jl in zip(serial_manipulator_pose_jacobian_list, serial_manipulator_pose_list, line_jacobian_list):
            numpy.testing.assert_almost_equal(serial_manipulator_robot.line_jacobian(J, x, k_), Jl, 12, "Error in line_jacobian")

    def test_plane_jacobian(self):
        for J, x, Jpi in zip(serial_manipulator_pose_jacobian_list, serial_manipulator_pose_list, plane_jacobian_list):
            numpy.testing.assert_almost_equal(serial_manipulator_robot.plane_jacobian(J, x, k_), Jpi, 12, "Error in plane_jacobian")

    def test_line_to_line_distance_jacobian(self):
        result_of_line_to_line_distance_jacobian = get_list_of_matrices_from_mat(
            'result_of_line_to_line_distance_jacobian', mat)
        for Jl, x, h, result in zip(line_jacobian_list, serial_manipulator_pose_list, h_list, result_of_line_to_line_distance_jacobian):
            numpy.testing.assert_almost_equal(
                serial_manipulator_robot.line_to_line_distance_jacobian(Jl, get_line_from_dq(x, k_), get_line_from_dq(h, k_)),
                result,
                12, "Error in line_to_line_distance_jacobian")

    def test_line_to_point_distance_jacobian(self):
        result_of_line_to_point_distance_jacobian = get_list_of_matrices_from_mat(
            'result_of_line_to_point_distance_jacobian', mat)
        for Jl, x, h, result in zip(line_jacobian_list, serial_manipulator_pose_list, h_list, result_of_line_to_point_distance_jacobian):
            numpy.testing.assert_almost_equal(
                serial_manipulator_robot.line_to_point_distance_jacobian(Jl, get_line_from_dq(x, k_), get_point_from_dq(h)),
                result,
                12, "Error in line_to_point_distance_jacobian")

    def test_plane_to_point_distance_jacobian(self):
        result_of_plane_to_point_distance_jacobian = get_list_of_matrices_from_mat(
            'result_of_plane_to_point_distance_jacobian', mat)
        for Jpi, x, h, result in zip(plane_jacobian_list, serial_manipulator_pose_list, h_list,
                                     result_of_plane_to_point_distance_jacobian):
            numpy.testing.assert_almost_equal(serial_manipulator_robot.plane_to_point_distance_jacobian(Jpi, get_point_from_dq(h)),
                                              result,
                                              12, "Error in plane_to_point_distance_jacobian")

    def test_point_to_line_distance_jacobian(self):
        result_of_point_to_line_distance_jacobian = get_list_of_matrices_from_mat(
            'result_of_point_to_line_distance_jacobian', mat)
        for Jt, x, h, result in zip(translation_jacobian_list, serial_manipulator_pose_list, h_list,
                                    result_of_point_to_line_distance_jacobian):
            numpy.testing.assert_almost_equal(
                serial_manipulator_robot.point_to_line_distance_jacobian(Jt, translation(x), get_line_from_dq(h, k_)),
                result,
                12, "Error in point_to_line_distance_jacobian")

    def test_point_to_plane_distance_jacobian(self):
        result_of_point_to_plane_distance_jacobian = get_list_of_matrices_from_mat(
            'result_of_point_to_plane_distance_jacobian', mat)
        for Jt, x, h, result in zip(translation_jacobian_list, serial_manipulator_pose_list, h_list,
                                    result_of_point_to_plane_distance_jacobian):
            numpy.testing.assert_almost_equal(
                serial_manipulator_robot.point_to_plane_distance_jacobian(Jt, translation(x), get_plane_from_dq(h, k_)),
                result,
                12, "Error in point_to_plane_distance_jacobian")

    def test_point_to_point_distance_jacobian(self):
        result_of_point_to_point_distance_jacobian = get_list_of_matrices_from_mat(
            'result_of_point_to_point_distance_jacobian', mat)
        for Jt, x, h, result in zip(translation_jacobian_list, serial_manipulator_pose_list, h_list,
                                    result_of_point_to_point_distance_jacobian):
            numpy.testing.assert_almost_equal(
                serial_manipulator_robot.point_to_point_distance_jacobian(Jt, translation(x), get_point_from_dq(h)),
                result,
                12, "Error in point_to_point_distance_jacobian")


    def test_line_to_line_angle_jacobian(self):
        result_of_line_to_line_angle_jacobian = get_list_of_matrices_from_mat(
            'result_of_line_to_line_angle_jacobian', mat)
        for Jl, x, h, result in zip(line_jacobian_list, serial_manipulator_pose_list, h_list,
                                    result_of_line_to_line_angle_jacobian):
            numpy.testing.assert_almost_equal(
                serial_manipulator_robot.line_to_line_angle_jacobian(Jl, get_line_from_dq(x, k_), get_line_from_dq(h, k_)),
                result,
                12, "Error in test_line_to_line_angle_jacobian")


if __name__ == '__main__':
    unittest.main()
