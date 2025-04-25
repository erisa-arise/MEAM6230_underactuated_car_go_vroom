#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import pygame
import sys
import math
import numpy as np
import os

class TeleopNode(Node):
    def __init__(self):
        super().__init__('teleop_pygame_node')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)

        pygame.init()
        self.screen = pygame.display.set_mode((100, 100))  # Dummy window
        pygame.display.set_caption("ROS2 Teleop")

        self.linear_velocity = 0.0
        self.angular_velocity = 0.0

        self.subscription = self.create_subscription(
            Odometry,
            '/ego_racecar/odom',  # Change this if your topic name is different
            self.odom_callback,
            10
        )

        self.pose_data = []
        self.control_data = []
    
    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        qx, qy, qz, qw = q.x, q.y, q.z, q.w
        yaw = math.atan2(2.0*(qw*qz + qx*qy), 1.0 - 2.0*(qy*qy + qz*qz))
        self.latest_yaw = yaw
        self.latest_x = x
        self.latext_y = y
    
    def save_traj_data(self, save_dir = "/home/ubuntu/MEAM6230/final_proj/src/MEAM6230_Underactuated_ODE/data/latest_sim_teleop_data"):
        pose_data_arr = np.array(self.pose_data)
        control_data_arr = np.array(self.control_data)

        pose_data_path = os.path.join(save_dir, "state_history.npy")
        control_data_path = os.path.join(save_dir, "control_history.npy")

        np.save(pose_data_path, pose_data_arr)
        np.save(control_data_path, control_data_arr)

    def timer_callback(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.save_traj_data()
                pygame.quit()
                rclpy.shutdown()
                sys.exit()

        keys = pygame.key.get_pressed()

        # Linear velocity (forward/back)
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            self.linear_velocity = 1.0
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            self.linear_velocity = -0.4
        else:
            self.linear_velocity = 0.0

        # Angular velocity (steering)
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.angular_velocity = 1.0
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.angular_velocity = -1.0
        else:
            self.angular_velocity = 0.0

        twist = Twist()
        twist.linear.x = self.linear_velocity
        twist.angular.z = self.angular_velocity

        if self.linear_velocity > 0.5:
            controls = [self.linear_velocity, self.angular_velocity]
            pose = [self.latest_x, self.latext_y, self.latest_yaw]
            self.control_data.append(controls)
            self.pose_data.append(pose)

        self.publisher_.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = TeleopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        pygame.quit()


if __name__ == '__main__':
    main()
