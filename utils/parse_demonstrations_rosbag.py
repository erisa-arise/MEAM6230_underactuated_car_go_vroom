from rclpy.serialization import deserialize_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rosidl_runtime_py.utilities import get_message
import numpy as np
import math
import matplotlib.pyplot as plt
import os

def load_messages(bag_path: str, topic_name: str, msg_type_str: str):
    reader: SequentialReader = SequentialReader()
    storage_options: StorageOptions = StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options: ConverterOptions = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
    reader.open(storage_options, converter_options)

    msg_type = get_message(msg_type_str)

    topic_msgs = []

    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic == topic_name:
            msg = deserialize_message(data, msg_type)
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            topic_msgs.append((stamp, msg))
    return topic_msgs

def reduce_data(ackermann_messages, rigid_body_messages, rigid_body_name):
    # extract the relevant data from the messages
    ackermann_data = []
    rigid_body_data = []

    for stamp, msg in ackermann_messages:
        ackermann_data.append((stamp, (msg.drive.speed, msg.drive.steering_angle)))

    for stamp, msg in rigid_body_messages:
        for rigid_body in msg.rigidbodies:
            if rigid_body.rigid_body_name == rigid_body_name:
                quaternion = rigid_body.pose.orientation
                yaw = yaw_from_quaternion(quaternion)
                rigid_body_data.append((stamp, 
                                        (rigid_body.pose.position.x, rigid_body.pose.position.y, yaw)))
    return ackermann_data, rigid_body_data

def yaw_from_quaternion(quat) -> float:
    """
    Converts a quaternion to yaw angle.
    
    Args:
        q (Quaternion): The quaternion to convert
    Returns:
        yaw (float): The yaw angle in radians
    """
    siny_cosp: float = 2.0 * (quat.w * quat.z + quat.x * quat.y)
    cosy_cosp: float = 1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z)
    return math.atan2(siny_cosp, cosy_cosp) + np.pi / 2 # Adjust for ROS coordinate system

def generate_dataset(ackermann_data, rigid_body_data):
    state_history = []
    control_history = []
    for ackermann_stamp, ackermann in ackermann_data:
        # find the closest pose to this timestamp
        closest_rigid_body = None
        min_diff = float('inf')
        for t, data in rigid_body_data:
            diff = abs(t - ackermann_stamp)
            if diff < min_diff:
                min_diff = diff
                closest_rigid_body = (t, data)

        if closest_rigid_body:
            rigid_body_stamp, rigid_body = closest_rigid_body
            state_history.append(rigid_body)
            control_history.append(ackermann)
    return np.array(state_history), np.array(control_history)

def visualize_bev(state_history):
    fig, ax = plt.subplots()
    
    # Add ellipse to represent boundary function
    theta = np.linspace(0, 2 * np.pi, 100)
    a, b = 3.5, 2.5
    x0, y0 = 0.5, 0.0
    x = x0 + a * np.cos(theta)
    y = y0 + b * np.sin(theta)
    ax.plot(x, y, 'b--', label='Ellipse')

    state_history_size = state_history.shape[0]
    ax.plot(state_history[:state_history_size//reduction_factor, 0],
            state_history[:state_history_size//reduction_factor, 1], 'ro', label='State History')

    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('State History')
    ax.legend()
    return fig, ax

def visualize_state_history(state_history):
    state_history_size = state_history.shape[0]
    
    fig, ax = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    ax[0].plot(state_history[:state_history_size//reduction_factor, 0], 'r-')
    ax[0].set_title('X Position History')
    ax[0].set_ylabel('X Position')
    ax[1].plot(state_history[:state_history_size//reduction_factor, 1], 'b-')
    ax[1].set_title('Y Position History')
    ax[1].set_ylabel('Y Position')
    ax[2].plot(np.cos(state_history[:state_history_size//reduction_factor, 2]), 'g-')
    ax[2].set_title('Cosine Yaw History')
    ax[2].set_xlabel('Time Step')
    ax[2].set_ylabel('Yaw')
    fig.tight_layout()
    return fig, ax

def visualize_control_history(control_history):
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    control_history_size = control_history.shape[0]
    
    ax[0].plot(control_history[:control_history_size//reduction_factor, 0], 'b-')
    ax[0].set_title('Velocity')
    ax[0].set_ylabel('Velocity')

    ax[1].plot(control_history[:control_history_size//reduction_factor, 1], 'r-')
    ax[1].set_title('Steering Angle')
    ax[1].set_xlabel('Time Step')
    ax[1].set_ylabel('Steering Angle')

    fig.tight_layout()
    return fig, ax

if __name__ == "__main__":
    filepath = os.path.dirname(os.path.abspath(__file__))
    rosbag_path = f"{filepath}/../data/real_data/teleop_demonstrations"
    out_dir = f"{filepath}/../data/real_data/"
    rigid_body_name = "racecar_vroom.racecar_vroom"
    ackermann_messages = load_messages(rosbag_path, "/ackermann_cmd", "ackermann_msgs/msg/AckermannDriveStamped")
    rigid_body_messages = load_messages(rosbag_path, "/rigid_bodies", "mocap4r2_msgs/msg/RigidBodies")

    ackermann_data, pose_data = reduce_data(ackermann_messages, rigid_body_messages, rigid_body_name)
    state_history, control_history = generate_dataset(ackermann_data, pose_data)

    # drop the first 40 samples of each
    state_history = state_history[40:, :]
    control_history = control_history[40:, :]

    out_state_path = out_dir + "state_history"
    out_control_path = out_dir + "control_history"

    np.save(out_state_path, state_history)
    np.save(out_control_path, control_history)

    reduction_factor = 30
    visualize_bev(state_history)
    visualize_state_history(state_history)
    visualize_control_history(control_history)
    plt.show()