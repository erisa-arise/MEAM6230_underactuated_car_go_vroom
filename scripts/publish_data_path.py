#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import numpy as np

class TrajectoryMarkerPublisher(Node):
    def __init__(self, path_to_npy):
        super().__init__('trajectory_marker_publisher')

        self.publisher_ = self.create_publisher(Marker, '/trajectory_marker', 10)
        self.timer = self.create_timer(1.0, self.timer_callback)  # 1Hz

        # Load the trajectory data
        state_history = np.load(path_to_npy)  # shape: (N, 3)

        # Initialize Marker
        self.marker = Marker()
        self.marker.header.frame_id = "map"
        self.marker.ns = "trajectory"
        self.marker.id = 0
        self.marker.type = Marker.LINE_STRIP
        self.marker.action = Marker.ADD

        self.marker.scale.x = 0.1  # Line width

        # Set color to blue
        self.marker.color.r = 0.0
        self.marker.color.g = 0.0
        self.marker.color.b = 1.0
        self.marker.color.a = 1.0

        # Set orientation and lifetime
        self.marker.pose.orientation.w = 1.0
        self.marker.lifetime.sec = 0  # 0 = forever

        # Populate points
        for state in state_history:
            point = Point()
            point.x = float(state[0])
            point.y = float(state[1])
            point.z = 0.0
            self.marker.points.append(point)

    def timer_callback(self):
        self.marker.header.stamp = self.get_clock().now().to_msg()
        self.publisher_.publish(self.marker)

def main(args=None):
    rclpy.init(args=args)

    path_to_npy = "/home/frankgon/ros2_ws/src/reactive_car/data/latest_sim_teleop_data/state_history.npy"
    trajectory_marker_publisher = TrajectoryMarkerPublisher(path_to_npy)

    rclpy.spin(trajectory_marker_publisher)

    trajectory_marker_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
