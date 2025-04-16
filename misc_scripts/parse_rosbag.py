from rclpy.serialization import deserialize_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rosidl_runtime_py.utilities import get_message
import numpy as np
import math
import matplotlib.pyplot as plt

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

def yaw_from_quaternion(quat):
    yaw = math.atan2(2.0*(quat.y*quat.z + quat.w*quat.x), quat.w*quat.w - quat.x*quat.x - quat.y*quat.y + quat.z*quat.z)
    return yaw

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

def visualize_state_history(state_history):
    plt.figure()

    # Add ellipse to represent boundary function
    theta = np.linspace(0, 2 * np.pi, 100)
    a = 3.5 
    b = 2.5
    x0, y0 = (0.5, 0.0)
    x = x0 + a * np.cos(theta)
    y = y0 + b * np.sin(theta)
    plt.plot(x, y, 'b--', label='Ellipse')

    # plot the state history
    plt.plot(state_history[:, 0], state_history[:, 1], 'ro', label='State History')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('State History')
    plt.legend()
    plt.show()

def visualize_control_history(control_history):
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(control_history[:, 0], 'b-')
    ax[0].set_title('Velocity')
    ax[1].plot(control_history[:, 1], 'r-')
    ax[1].set_title('Steering Angle')
    plt.show()


if __name__ == "__main__":
    rosbag_path = "/home/frankgon/ros2_ws/src/reactive_car/data/rosbag2_2025_04_11-23_21_31"
    rigid_body_name = "racecar_vroom.racecar_vroom"
    ackermann_messages = load_messages(rosbag_path, "/ackermann_cmd", "ackermann_msgs/msg/AckermannDriveStamped")
    rigid_body_messages = load_messages(rosbag_path, "/rigid_bodies", "mocap4r2_msgs/msg/RigidBodies")

    ackermann_data, pose_data = reduce_data(ackermann_messages, rigid_body_messages, rigid_body_name)
    state_history, control_history = generate_dataset(ackermann_data, pose_data)
    visualize_state_history(state_history)