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
from dqrobotics.robot_modeling import DQ_SerialManipulatorDH
from dqrobotics.robot_modeling import DQ_SerialManipulatorMDH
import numpy as np
from math import pi

## @package DQ_SerialManipulatorMDH_test
#  This module implements a class to perform the unit tests of the DQ_SerialManipulatorMDH class.
#  The unit tests compare the new DQ_SerialManipulatorMDH class with the DQ_SerialManipulatorDH class,
#  which is used as the baseline.



# Robot Declaration using the Standard DH Convention
# (See Table 3.4 from Robot Modeling and Control Second Edition, Spong, Mark W.
#  Hutchinson, Seth M., Vidyasagar)

d2 = 0.4
d3 = 0.1
d6 = 0.3
robot_DH_theta = np.array([0, 0, 0, 0, 0, 0])
robot_DH_d = np.array([0,d2,d3,0,0,d6])
robot_DH_a = np.array([0, 0, 0, 0 ,0 ,0])
robot_DH_alpha = np.array([-pi/2, pi/2, 0,-pi/2,pi/2,0])
robot_type = np.array([0,0,1,0,0,0])
robot_DH_matrix = np.array([robot_DH_theta, robot_DH_d, robot_DH_a, robot_DH_alpha, robot_type])
robotDH = DQ_SerialManipulatorDH(robot_DH_matrix)


# Robot Declaration using the Modified DH Convention
# (See Table 2.1 from Foundations of Robotics, Tsuneo Yoshikawa)
robot_MDH_theta = np.array([0, 0, 0, 0, 0, 0])
robot_MDH_d = np.array([0, d2, d3, 0, 0, 0])
robot_MDH_a = np.array([0, 0, 0, 0, 0, 0])
robot_MDH_alpha = np.array([0, -pi / 2, pi / 2, 0, -pi / 2, pi / 2])
robot_MDH_matrix = np.array([robot_MDH_theta, robot_MDH_d, robot_MDH_a, robot_MDH_alpha, robot_type])
robotMDH = DQ_SerialManipulatorMDH(robot_MDH_matrix)
robotMDH.set_effector(1 + DQ.E * 0.5 * DQ.k * d6)


number_of_trials = 1000
q_list = np.random.rand(6, number_of_trials)
q_dot_list = np.random.rand(6, number_of_trials)

StanfordDHRobot = robotDH
StanfordMDHRobot = robotMDH

## DQTestCase class.
#  This class performs the unit tests of the DQ_SerialManipulatorMDH class.
class DQTestCase(unittest.TestCase):

    global StanfordDHRobot
    global StanfordMDHRobot

    ## test_get_thetas
    # Performs the unit tests of the get_thetas() methods
    def test_get_thetas(self):
        np.testing.assert_equal(StanfordDHRobot.get_thetas(), robot_DH_theta,
                                "Error in DQ_SerialManipulatorDH.get_thetas()")
        np.testing.assert_equal(StanfordMDHRobot.get_thetas(), robot_MDH_theta,
                                "Error in DQ_SerialManipulatorMDH.get_thetas()")

    ## test_get_ds
    # Performs the unit tests of the get_ds() methods
    def test_get_ds(self):
        np.testing.assert_equal(StanfordDHRobot.get_ds(), robot_DH_d,
                                "Error in DQ_SerialManipulatorDH.get_ds()")
        np.testing.assert_equal(StanfordMDHRobot.get_ds(), robot_MDH_d,
                                "Error in DQ_SerialManipulatorMDH.get_ds()")

    ## test_get_as
    # Performs the unit tests of the get_as() methods
    def test_get_as(self):
        np.testing.assert_equal(StanfordDHRobot.get_as(), robot_DH_a,
                                "Error in DQ_SerialManipulatorDH.get_as()")
        np.testing.assert_equal(StanfordMDHRobot.get_as(), robot_MDH_a,
                                "Error in DQ_SerialManipulatorMDH.get_as()")

    ## test_get_alphas
    # Performs the unit tests of the get_alphas() methods
    def test_get_alphas(self):
        np.testing.assert_equal(StanfordDHRobot.get_alphas(), robot_DH_alpha,
                                "Error in DQ_SerialManipulatorDH.get_alphas()")
        np.testing.assert_equal(StanfordMDHRobot.get_alphas(), robot_MDH_alpha,
                                "Error in DQ_SerialManipulatorMDH.get_alphas()")

    ## test_get_types
    # Performs the unit tests of the get_types() methods
    def test_get_types(self):
        np.testing.assert_equal(StanfordDHRobot.get_types(), robot_type,
                                "Error in DQ_SerialManipulatorDH.get_types()")
        np.testing.assert_equal(StanfordMDHRobot.get_types(), robot_type,
                                "Error in DQ_SerialManipulatorMDH.get_types()")

    ## test_serial_manipulator_fkm
    # Performs the unit tests of the fkm() methods
    def test_serial_manipulator_fkm(self):
        for i in range(number_of_trials):
            q = q_list [:,i]
            x1 = StanfordDHRobot.fkm(q)
            x2 = StanfordMDHRobot.fkm(q)
            np.testing.assert_almost_equal(vec8(x1), vec8(x2), 12,
                                           "Error in DQ_SerialManipulatorMDH.fkm()")

    ## test_serial_manipulator_pose_jacobian
    # Performs the unit tests of the pose_jacobian() methods
    def test_serial_manipulator_pose_jacobian(self):
        for i in range(number_of_trials):
            q = q_list [:,i]
            J1 = StanfordDHRobot.pose_jacobian(q)
            J2 = StanfordMDHRobot.pose_jacobian(q)
            np.testing.assert_almost_equal(J1, J2, 12,
                                           "Error in DQ_SerialManipulatorMDH.pose_jacobian()")

    ## test_serial_manipulator_pose_jacobian_derivative
    # Performs the unit tests of the pose_jacobian_derivative() methods
    def test_serial_manipulator_pose_jacobian_derivative(self):
        for i in range(number_of_trials):
            q = q_list [:,i]
            q_dot = q_dot_list[:, i]
            J1_dot = StanfordDHRobot.pose_jacobian_derivative(q, q_dot)
            J2_dot = haminus8(StanfordMDHRobot.get_effector())@StanfordMDHRobot.pose_jacobian_derivative(q, q_dot)
            np.testing.assert_almost_equal(J1_dot, J2_dot , 12,
                                           "Error in DQ_SerialManipulatorMDH.pose_jacobian_derivative()")


if __name__ == '__main__':
    unittest.main()
