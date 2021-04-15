import unittest
import numpy

from dqrobotics import *
from dqrobotics.robots import *
from dqrobotics.robot_control import *
from dqrobotics.solvers import *


class DQTestCase(unittest.TestCase):

    def test_cpp_issue_30(self):
        # Define the plane
        plane = k_;

        # Define the robot's kinematics
        robot_kinematics = KukaLw4Robot.kinematics()

        # Constrained QP controller
        solver = DQ_QuadprogSolver()
        controller = DQ_ClassicQPController(robot_kinematics, solver)
        controller.set_control_objective(ControlObjective.DistanceToPlane)
        controller.set_gain(10.0)
        controller.set_damping(0.01)
        controller.set_stability_counter_max(100)
        controller.set_target_primitive(plane)

    def test_cpp_issue_32(self):
        # Define the plane
        plane = k_;

        # Define the robot's kinematics
        robot_kinematics = KukaLw4Robot.kinematics()

        # Constrained QP controller
        solver = DQ_QuadprogSolver()
        controller = DQ_ClassicQPController(robot_kinematics, solver)
        controller.set_control_objective(ControlObjective.DistanceToPlane)
        controller.set_gain(10.0)
        controller.set_damping(0.01)
        controller.set_stability_counter_max(100)
        controller.set_target_primitive(plane)

        # Arbitrary joint values
        q0 = np.array([0.0, 1.7453e-01, 0.0, 1.5708, 0.0, 2.6273e-01, 0.0])

        numpy.testing.assert_almost_equal(controller.get_task_variable(q0), [0.7716435593916681767723275697790086269378662109375])


if __name__ == '__main__':
    unittest.main()
