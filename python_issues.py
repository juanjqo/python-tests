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

    # Print function malfunctioning
    def test_python_issue_18(self):
        r = DQ([0.5068154, -0.8591303, 0.0555768, -0.0440969])
        r = normalize(r)
        axis = r.rotation_axis()
        angle = r.rotation_angle()
        self.assertEqual(' - 0.996608i + 0.06447j - 0.051153k', str(axis))
        self.assertEqual('2.0786195489067434', str(angle))

    def test_python_issue_22(self):
        from dqrobotics.robot_modeling import DQ_DifferentialDriveRobot
        q = np.array([0.12, 0.0, 0.0])
        diff_robot = DQ_DifferentialDriveRobot(1, 1)
        J = diff_robot.pose_jacobian(q)
        print(J)

    def test_python_issue26(self):
        import numpy as np
        from dqrobotics.solvers import DQ_QuadprogSolver
        from dqrobotics.robot_control import DQ_ClassicQPController
        from dqrobotics.robot_control import ControlObjective
        from dqrobotics.robots import KukaLw4Robot
        import math

        robot = KukaLw4Robot.kinematics()
        qp_solver = DQ_QuadprogSolver()
        controller = DQ_ClassicQPController(robot, qp_solver)
        controller.set_control_objective(ControlObjective.Pose)
        controller.set_gain(10.0)
        controller.set_damping(0.001)
        q = np.array([math.pi, math.pi, math.pi / 2, 0, math.pi, math.pi / 2, 0])
        effector_p = 0.0 * k_
        effector_t = 1 + E_ * 0.5 * effector_p
        effector_phi = -math.pi
        effector_n = j_
        effector_r = math.cos(effector_phi / 2) + effector_n * math.sin(effector_phi / 2)
        effector_pose = effector_t * effector_r
        print("The control signal is {}.".format(controller.compute_setpoint_control_signal(q, vec8(effector_pose))))

if __name__ == '__main__':
    unittest.main()
