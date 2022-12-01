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
from dqrobotics.robot_modeling import DQ_HolonomicBase
from dqrobotics.robot_modeling import DQ_DifferentialDriveRobot
from dqrobotics.robots import FrankaEmikaPandaRobot
from dqrobotics.robots import KukaLw4Robot
from dqrobotics.robots import KukaYoubotRobot
from dqrobotics.robot_modeling import DQ_SerialManipulatorDenso
from dqrobotics.robot_modeling import DQ_WholeBody
import numpy as np
from math import pi, sin , cos


def numerical_differentiation(J, T):
    """
    returns the numerical differentiation of a vector of matrices J using
    the eight-point central difference method.
    Args:
        J: A np array of size (iterations, 8, n).
        T: The step size.
    Returns:
        J_dot: A np array of size (iterations, 8, n) containing the numerical.
        differentiation of J.
    """
    s = J.shape[0]
    rows = J.shape[1]
    cols = J.shape[2]
    J_dot = np.zeros((s, rows, cols))
    for i in range(4, s-4):
        J_dot[i] = (3 * J[i - 4] - 32 * J[i - 3] + 168 * J[i - 2] - 672 * J[i - 1]
                    + 672 * J[i+1]-168 * J[i+2]+32 * J[i+3]-3 * J[i+4]) / (840 * T)
    return J_dot


def compute_jacobian_derivatives(robot, iterations, T):
    """
    returns the pose_jacobian_derivative using the analytical and
    numerical approach of a DQ_Kinematics robot.

    Args:
        robot:   A DQ_Kinematics robot. Example: FrankaEmikaPandaRobot.kinematics()
        iterations:  Number of trials. Example: 5000
        T: Step size. Example: 1e-4
    Returns:
        J_dot: A np array of size (iterations, 8, n) containing the pose jacobian derivatives of all trials
               using the DQ_Kinematics.pose_jacobian_derivative method.
        numerical_J_dot: A np array of size (iterations, 8, n) containing the pose jacobian derivatives of all trials
               using numerical differentiation
    """
    njoints = robot.get_dim_configuration_space()
    q = np.zeros(njoints)
    q_dot = np.zeros(njoints)
    J_ = robot.pose_jacobian(q)
    cols = J_.shape[1]
    J = np.zeros((iterations, 8, cols))
    J_dot = np.zeros((iterations, 8, cols))
    t = 0
    w = 2 * pi
    for i in range(0, iterations):
        t = i * T
        theta = sin(w * t)
        theta_dot = w * cos(w * t)
        for j in range(0, njoints):
            q[j] = theta
            q_dot[j] = theta_dot
        J[i, :, :] = robot.pose_jacobian(q)
        J_dot[i, :, :] = robot.pose_jacobian_derivative(q, q_dot)
    numerical_J_dot = numerical_differentiation(J, T)
    return J_dot, numerical_J_dot

#######################################################################################
#######################################################################################
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

#######################################################################################
#######################################################################################
# Robot Declaration using the Modified DH Convention
# (See Table 2.1 from Foundations of Robotics, Tsuneo Yoshikawa)
robot_MDH_theta = np.array([0, 0, 0, 0, 0, 0])
robot_MDH_d = np.array([0, d2, d3, 0, 0, 0])
robot_MDH_a = np.array([0, 0, 0, 0, 0, 0])
robot_MDH_alpha = np.array([0, -pi / 2, pi / 2, 0, -pi / 2, pi / 2])
robot_MDH_matrix = np.array([robot_MDH_theta, robot_MDH_d, robot_MDH_a, robot_MDH_alpha, robot_type])
robotMDH = DQ_SerialManipulatorMDH(robot_MDH_matrix)
robotMDH.set_effector(1 + DQ.E * 0.5 * DQ.k * d6)

#######################################################################################
#######################################################################################
## Denso Robot Manipulator definition
#
d2 = 0.4
d3 = 0.1
d6 = 0.3
robot_Denso_a = np.array([0, 0, 0, 0, 0, 0])
robot_Denso_b = np.array([0,d2,d3,0,0,d6])
robot_Denso_d = np.array([0, 0, 0, 0 ,0 ,0])
robot_Denso_alpha = np.array([-pi/2, pi/2, 0,-pi/2,pi/2,0])
robot_Denso_beta = np.array([-pi/2, pi/2, 0,-pi/2,pi/2,0])
robot_Denso_gamma = np.array([-pi/2, pi/2, 0,-pi/2,pi/2,0])

robot_Denso_matrix = np.array([robot_Denso_a, robot_Denso_b, robot_Denso_d, robot_Denso_alpha,
                            robot_Denso_beta, robot_Denso_gamma])

robotDenso= DQ_SerialManipulatorDenso(robot_Denso_matrix)

#######################################################################################
#######################################################################################
robot1 = KukaLw4Robot.kinematics()
robot2 = KukaLw4Robot.kinematics()
whole_body_robot = DQ_WholeBody(robot1)
whole_body_robot.add(robot2)



#######################################################################################
#######################################################################################

kuka = KukaLw4Robot.kinematics()
franka = FrankaEmikaPandaRobot.kinematics()
hol_base = DQ_HolonomicBase()
diff_base = DQ_DifferentialDriveRobot(0.3, 0.01)
stanfordDHRobot = robotDH
stanfordMDHRobot = robotMDH
youbot = KukaYoubotRobot.kinematics()

## DQTestCase class.
#  This class performs the unit tests of the DQ_Kinematics::pose_jacobian_derivative class.
class DQTestCase(unittest.TestCase):
    global hol_base
    global diff_base
    global franka
    global StanfordDHRobot
    global StanfordMDHRobot
    global robotDenso
    global youbot
    global whole_body_robot

    ## test_holonomic_base_pose_jacobian_derivative
    # Performs the unit tests of the DQ_HolonomicBase.pose_jacobian_derivative() method
    def test_holonomic_base_pose_jacobian_derivative(self):
        iterations = 5000
        J_dot, Numerical_J_dot_ = compute_jacobian_derivatives(hol_base, iterations, 1e-4)
        for i in range(0, iterations):
            if (i > 4 and i < iterations - 4):
                np.testing.assert_almost_equal(J_dot[i, :, :] , Numerical_J_dot_[i, :, :],  10,
                                               "Error in DQ_HolonomicBase.pose_jacobian_derivative()")

    ## test_differential_base_pose_jacobian_derivative
    # Performs the unit tests of the DQ_DifferentialDriveRobot.pose_jacobian_derivative() method
    def test_differential_base_pose_jacobian_derivative(self):
        iterations = 5000
        J_dot, Numerical_J_dot_ = compute_jacobian_derivatives(diff_base, iterations, 1e-4)
        for i in range(0, iterations):
            if (i > 4 and i < iterations - 4):
                np.testing.assert_almost_equal(J_dot[i, :, :], Numerical_J_dot_[i, :, :],  10,
                                               "Error in DQ_DifferentialDriveRobot.pose_jacobian_derivative()")

    ## test_serial_manipulator_pose_jacobian_derivative
    # Performs the unit tests of the DQ_DifferentialDriveRobot.pose_jacobian_derivative() method
    def test_serial_manipulator_pose_jacobian_derivative(self):
        iterations = 5000
        robots = [kuka, stanfordDHRobot, franka, stanfordMDHRobot]
        for robot in robots:
            J_dot, Numerical_J_dot_ = compute_jacobian_derivatives(robot, iterations, 1e-4)
            for i in range(0, iterations):
                if (i > 4 and i < iterations - 4):
                    np.testing.assert_almost_equal(J_dot[i, :, :], Numerical_J_dot_[i, :, :],  10,
                                                   "Error in DQ_SerialManipulator.pose_jacobian_derivative()")


    ##########################################################################################################
    ##########################################################################################################
    ####     Testing classes where the pose jacobian derivative method is not available for the user

    ## test_serial_manipulator_denso_pose_jacobian_derivative
    # Performs the unit tests of the DQ_SerialManipulatorDenso.pose_jacobian_derivative() method
    def test_serial_manipulator_denso_pose_jacobian_derivative(self):
        iterations = 5000
        J_dot, Numerical_J_dot_ = compute_jacobian_derivatives(robotDenso, iterations, 1e-4)
        for i in range(0, iterations):
            if (i > 4 and i < iterations - 4):
                np.testing.assert_almost_equal(J_dot[i, :, :], Numerical_J_dot_[i, :, :],  10,
                                               "Error in DQ_SerialManipulatorDenso.pose_jacobian_derivative()")


    ## test_serial_whole_body_pose_jacobian_derivative
    # Performs the unit tests of the DQ_SerialWholeBody.pose_jacobian_derivative() method
    def test_serial_whole_body_pose_jacobian_derivative(self):
        njoints = youbot.get_dim_configuration_space()
        with self.assertRaises(Exception):
          J_dot =youbot.pose_jacobian_derivative(np.zeros(njoints), np.zeros(njoints))

    ## test_whole_body_pose_jacobian_derivative
    # Performs the unit tests of the DQ_WholeBody.pose_jacobian_derivative() method
    def test_whole_body_pose_jacobian_derivative(self):
        njoints = whole_body_robot.get_dim_configuration_space()
        with self.assertRaises(Exception):
          J_dot = whole_body_robot.pose_jacobian_derivative(np.zeros(njoints), np.zeros(njoints))

if __name__ == '__main__':
    unittest.main()

