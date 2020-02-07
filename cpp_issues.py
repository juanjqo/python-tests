import unittest
import quadprog

from dqrobotics import *
from dqrobotics.robot_modeling import *
from dqrobotics.robot_control import *
from dqrobotics.robots import *
from dqrobotics.solvers import *

class DQTestCase(unittest.TestCase):

    # DQ_SerialManipulator.fkm
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

if __name__ == '__main__':
    unittest.main()