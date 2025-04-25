#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import numpy as np

class TrajectoryPublisher(Node):
    def __init__(self, path_to_npy):
        super().__init__('trajectory_publisher')
        
        self.publisher_ = self.create_publisher(Path, 'trajectory_path', 10)
        self.timer = self.create_timer(1.0, self.timer_callback)  # publish at 1Hz

        self.path_msg = Path()
        self.path_msg.header.frame_id = "map"  # or "odom" depending on your setup

        state_history = np.load(path_to_npy)  # (N, 3)

        for state in state_history:
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.pose.position.x = float(state[0])  # First column
            pose.pose.position.y = float(state[1])  # Second column
            pose.pose.position.z = 0.0

            # Assume no orientation (identity quaternion)
            pose.pose.orientation.w = 1.0
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = 0.0

            self.path_msg.poses.append(pose)

    def timer_callback(self):
        self.path_msg.header.stamp = self.get_clock().now().to_msg()
        for pose in self.path_msg.poses:
            pose.header.stamp = self.get_clock().now().to_msg()
        self.publisher_.publish(self.path_msg)

def main(args=None):
    rclpy.init(args=args)

    path_to_npy = "/home/ubuntu/MEAM6230/final_proj/src/MEAM6230_Underactuated_ODE/data/latest_sim_teleop_data/state_history.npy"
    trajectory_publisher = TrajectoryPublisher(path_to_npy)

    rclpy.spin(trajectory_publisher)

    trajectory_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
