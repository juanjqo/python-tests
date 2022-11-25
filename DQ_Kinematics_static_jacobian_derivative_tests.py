"""(C) Copyright 2022 DQ Robotics Developers

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
 - Juan Jose Quiroz Omana -  juanjqo@g.ecc.u-tokyo.ac.jp
"""

import unittest
from dqrobotics import*
from dqrobotics.robots import FrankaEmikaPandaRobot
from dqrobotics.robot_modeling import DQ_Kinematics
import numpy as np
from math import pi, sin, cos
from DQ_Kinematics_pose_jacobian_derivative_tests import numerical_differentiation


#####################################
iterations = 5000
T = 1e-3

######################################
robot = FrankaEmikaPandaRobot.kinematics()
njoints = robot.get_dim_configuration_space()

q = np.zeros(njoints)
q_dot = np.zeros(njoints)

Jt     = np.zeros((iterations, 4, njoints))
Jt_dot = np.zeros((iterations, 4, njoints))

Jr     = np.zeros((iterations, 4, njoints))
Jr_dot = np.zeros((iterations, 4, njoints))

Jd     = np.zeros((iterations, 1, njoints))
Jd_dot = np.zeros((iterations, 1, njoints))

Jl     = np.zeros((iterations, 8, njoints))
Jl_dot = np.zeros((iterations, 8, njoints))

Jpi     = np.zeros((iterations, 8, njoints))
Jpi_dot = np.zeros((iterations, 8, njoints))

t = 0
w = 2 * pi
for i in range(0, iterations):
    t = i * T
    theta = sin(w * t)
    theta_dot = w * cos(w * t)
    q = np.array([theta, theta, theta, theta, theta, theta, theta])
    q_dot = np.array([theta_dot, theta_dot, theta_dot, theta_dot, theta_dot, theta_dot, theta_dot])

    x = robot.fkm(q)   # pose
    J = robot.pose_jacobian(q)  # pose Jacobian
    J_dot = robot.pose_jacobian_derivative(q, q_dot) # pose Jacobian derivative

    Jt[i, :, :] = DQ_Kinematics.translation_jacobian(J, x)
    Jt_dot[i, :, :] = DQ_Kinematics.translation_jacobian_derivative(J, J_dot, x, q_dot)

    Jr[i, :, :]     = DQ_Kinematics.rotation_jacobian(J)
    Jr_dot[i, :, :] = DQ_Kinematics.rotation_jacobian_derivative(J_dot)

    Jd[i, :, :] = DQ_Kinematics.distance_jacobian(J, x)
    Jd_dot[i, :, :] = DQ_Kinematics.distance_jacobian_derivative(J, J_dot, x, q_dot)

    Jl[i, :, :] = DQ_Kinematics.line_jacobian(J, x, k_)
    Jl_dot[i, :, :] = DQ_Kinematics.line_jacobian_derivative(J, J_dot, x, k_, q_dot)

    Jpi[i, :, :] = DQ_Kinematics.plane_jacobian(J, x, k_)
    Jpi_dot[i, :, :] = DQ_Kinematics.plane_jacobian_derivative(J, J_dot, x, k_, q_dot)



## DQTestCase class.
#  This class performs the unit tests of the following static methods
#  DQ_Kinematics::translation_jacobian_derivative.
#  DQ_Kinematics::rotation_jacobian_derivative.
#  DQ_Kinematics::line_jacobian_derivative.
#  DQ_Kinematics::plane_jacobian_derivative.
class DQTestCase(unittest.TestCase):
    global Jt
    global Jt_dot
    global Jr
    global Jr_dot
    global Jd
    global Jd_dot
    global Jl
    global Jl_dot

    decimal = 10

    @staticmethod
    def check_results(A, B, decimal, msg):
        """
        checks if matrix A and B are almost equals using np.assert_almost_equal
        Args:
            A: np array of size (iterations, m, n). The object to check
            B: np array of size (iterations, m, n). The expected object
            decimal: Desired precision
            msg: The error message to be printed in case of failure.
        """
        iterations = A.shape[0]
        for i in range(0, iterations):
            if (i > 4 and i < iterations - 4):
                np.testing.assert_almost_equal(A[i, :, :], B[i, :, :],  decimal, msg)

    def test_translation_jacobian_derivative(self):
        """
        tests the method DQ_Kinematics.translation_jacobian_derivative()
        """
        self.check_results(numerical_differentiation(Jt, T), Jt_dot, self.decimal,
                           "Error in DQ_Kinematics.translation_jacobian_derivative()")

    def test_rotation_jacobian_derivative(self):
        """
        tests the method DQ_Kinematics.rotation_jacobian_derivative()
        """
        self.check_results(numerical_differentiation(Jr, T), Jr_dot, self.decimal,
                           "Error in DQ_Kinematics.rotation_jacobian_derivative()")

    def test_distance_jacobian_derivative(self):
        """
        tests the method DQ_Kinematics.distance_jacobian_derivative()
        """
        self.check_results(numerical_differentiation(Jd, T), Jd_dot, self.decimal,
                           "Error in DQ_Kinematics.distance_jacobian_derivative()")

    def test_line_jacobian_derivative(self):
        """
        tests the method DQ_Kinematics.line_jacobian_derivative()
        """
        self.check_results(numerical_differentiation(Jl, T), Jl_dot, self.decimal,
                           "Error in DQ_Kinematics.line_jacobian_derivative()")

    def test_plane_jacobian_derivative(self):
        """
        tests the method DQ_Kinematics.plane_jacobian_derivative()
        """
        self.check_results(numerical_differentiation(Jpi, T), Jpi_dot, self.decimal,
                           "Error in DQ_Kinematics.plane_jacobian_derivative()")


if __name__ == '__main__':
    unittest.main()