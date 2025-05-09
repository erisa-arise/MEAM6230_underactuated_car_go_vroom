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

def reduce_data(rigid_body_messages, rigid_body_name, closest_point_messages):
    # extract the relevant data from the messages
    rigid_body_data = []
    closest_point_data = []

    for stamp, msg in rigid_body_messages:
        for rigid_body in msg.rigidbodies:
            if rigid_body.rigid_body_name == rigid_body_name:
                quaternion = rigid_body.pose.orientation
                yaw = yaw_from_quaternion(quaternion)
                rigid_body_data.append((stamp, 
                                        (rigid_body.pose.position.x, rigid_body.pose.position.y, yaw)))
    
    for stamp, msg in closest_point_messages:
        closest_point_data.append((stamp, (msg.point.x, msg.point.y)))

    return rigid_body_data, closest_point_data



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

def generate_dataset(rigid_body_data, closest_point_data):
    state_history = []
    closest_point_history = []
    time_history = []
    for closest_point_stamp, closest_point in closest_point_data:
        # find the closest pose to this timestamp
        closest_rigid_body = None
        min_diff = float('inf')
        for t, data in rigid_body_data:
            diff = abs(t - closest_point_stamp)
            if diff < min_diff:
                min_diff = diff
                closest_rigid_body = (t, data)

        if closest_rigid_body:
            rigid_body_stamp, rigid_body = closest_rigid_body
            state_history.append(rigid_body)
            closest_point_history.append(closest_point)
            time_history.append(closest_point_stamp)   
    return np.array(state_history), np.array(closest_point_history), np.array(time_history)

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
    ax.plot(state_history[:state_history_size, 0],
            state_history[:state_history_size, 1], 'ro', label='State History')

    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('State History')
    ax.legend()
    return fig, ax

def visualize_state_history(time_history, state_history):
    state_history_size = state_history.shape[0]
    time_history = np.array(time_history)
    time_history = (time_history - time_history[0])
    
    fig, ax = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    ax[0].plot(time_history, state_history[:state_history_size, 0], 'r-')
    ax[0].set_title('X Position History')
    ax[0].set_ylabel('X Position')
    ax[1].plot(time_history, state_history[:state_history_size, 1], 'b-')
    ax[1].set_title('Y Position History')
    ax[1].set_ylabel('Y Position')
    ax[2].plot(time_history, np.cos(state_history[:state_history_size, 2]), 'g-')
    ax[2].set_title('Cosine Yaw History')
    ax[2].set_xlabel('Time Step')
    ax[2].set_ylabel('Yaw')
    fig.tight_layout()
    return fig, ax

def visualize_se(state_history, closest_point_history, time_history):
    fig, ax = plt.subplots()
    state_history_size = state_history.shape[0]
    se_clean = []
    time_history_clean = []
    for i in range(state_history_size):
        se_i= np.linalg.norm(state_history[i, :2] - closest_point_history[i, :2])**2
        if se_i < 2:
            se_clean.append(se_i)
            time_history_clean.append(time_history[i])
    time_history_clean = np.array(time_history_clean)
    time_history_clean = (time_history_clean - time_history_clean[0])
    ax.scatter(time_history_clean, se_clean, label='SE')
    ax.set_xlabel('Time')
    ax.set_ylabel('SE')
    ax.set_title('Squared Error')
    ax.legend()
    return se_clean


if __name__ == "__main__":
    filepath = os.path.dirname(os.path.abspath(__file__))
    rosbag_path = f"{filepath}/../data/real_data/hardware_tests/nominal5"
    rigid_body_name = "f1tenth_car.f1tenth_car"
    rigid_body_messages = load_messages(rosbag_path, "/rigid_bodies", "mocap4r2_msgs/msg/RigidBodies")
    closest_point_messages = load_messages(rosbag_path, "/closest_point", "geometry_msgs/msg/PointStamped")

    pose_data, closest_point_data = reduce_data(rigid_body_messages, rigid_body_name, closest_point_messages)
    state_history, closest_point_history, time_history = generate_dataset(pose_data, closest_point_data)

    # drop the first 40 samples of each
    state_history = state_history[:320, :]
    closest_point_history = closest_point_history[:320, :]
    time_history = time_history[:320]

    # trim last 50 for nominal3
    # trim last 50 for nominal4
    # trim last 61 for nominal5 both loops
    # keep first 320 for nominal5 first loop 
    # trim first 320 and last 61 for nominal5 second loop 


    
    visualize_bev(state_history)
    visualize_state_history(time_history, state_history)
    squared_error = visualize_se(state_history, closest_point_history, time_history)
    print(f"Mean Squared Error: {np.mean(squared_error)}")
    plt.show()