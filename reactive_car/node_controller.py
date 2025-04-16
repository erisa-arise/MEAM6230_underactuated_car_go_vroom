import rclpy
import torch
import math
import casadi
import numpy as np

from casadi import MX, DM, SX
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.subscription import Subscription
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, Vector3
from ackermann_msgs.msg import AckermannDrive
from mocap4r2_msgs.msg import RigidBodies
from misc_scripts.n_ode import N_ODE
from typing import Tuple


class OdomToAckermann(Node):
    def __init__(self) -> None:
        super().__init__('Neural_ODE_Controller')
        self.vicon_subscriber: Subscription = self.create_subscription(RigidBodies, '/odom_topic', self.odom_callback, 10)
        self.ackermann_publisher: Publisher = self.create_publisher(AckermannDrive, '/ackermann_cmd', 10)
        self.get_logger().info('Initialized NODE Controller')

        # load the neural ODE model here
        self.model: N_ODE = N_ODE()
        self.model.load_state_dict(torch.load('model_weights.pth'))
        self.model.eval()

        # car parameters
        self.L = 1.0

    def odom_callback(self, msg: RigidBodies):
        # extract orientation and position from the VICON ROS2 message
        orientation: Quaternion = msg.rigidbodies.pose.orientation
        position: Vector3 = msg.rigidbodies.pose.position
        yaw: float = self.get_yaw_from_quaternion(orientation)

        # create the state tensor and query the Neural ODE
        state = np.array([position.x, position.y, yaw])
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            n_ode_output = self.model(state_tensor) # Waiting on josh's n_ode input script  

        # convert Neural ODE output to x_dot and then compute the safe control
        x_dot: np.ndarray[np.float32] = self.map_n_ode_to_x_dot(n_ode_output)
        v_safe, delta_safe = self.compute_safe_control(x_dot) 
        
        # publish the safe velocitya and steering angle
        ackermann_msg: AckermannDrive = AckermannDrive()
        ackermann_msg.speed = v_safe
        ackermann_msg.steering_angle = delta_safe
        self.ackermann_publisher.publish(ackermann_msg)    

    def get_yaw_from_quaternion(self, q: Quaternion) -> float:
        # Convert quaternion to yaw
        siny_cosp: float = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp: float = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)
    
    def map_n_ode_to_x_dot(self, n_ode_output: torch.Tensor[torch.float32]) -> np.ndarray[np.float32]:
        # Converts using the car dynamics model 
        v: float = n_ode_output[0][0].item()
        delta: float = n_ode_output[0][1].item()
        x_dot: np.ndarray[np.float32] = np.array([v * math.cos(delta), v * math.sin(delta), v/self.L * math.tan(delta)])
        return x_dot

    def compute_safe_control(self, state: np.ndarray[np.float32], x_dot: np.ndarray[np.float32]) -> Tuple[float]:
        # Computes the safe control using the control barrier function, control lyapunov function, and solves a QP using Casadi
        x: MX = casadi.MX.sym('x')
        y: MX = casadi.MX.sym('y')
        theta: MX = casadi.MX.sym('theta')

        gamma: float = 1.0  # Weight TODO: Tune
        epsilon: MX = casadi.MX.sym('epsilon')  # Slack variable

        v: MX = casadi.MX.sym('v')
        delta: MX = casadi.MX.sym('delta')
        delta_upper_bound: float = math.pi / 4  # 45 degrees
        delta_lower_bound: float = -math.pi / 4

        cost = v**2 + delta**2 + gamma*epsilon

        constraints = []
        constraint_lower_bound = []
        constraint_upper_bound = []

        constraints.append(delta)
        constraint_lower_bound.append(delta_lower_bound)
        constraint_upper_bound.append(delta_upper_bound)

        casadi.fabs
        
        return v, delta
    
    def control_boundary_function(self, state: np.ndarray[np.float32]):
        # B(x)
        pass
    
    def control_lyapunov_function(self, state: np.ndarray[np.float32]):
        # V(x)
        pass

def main(args=None):
    rclpy.init(args=args)
    node = OdomToAckermann()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
