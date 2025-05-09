import rclpy
import os
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from std_msgs.msg import Float32
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions

import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime
import numpy as np

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
            stamp = stamp = t * 1e-9
            topic_msgs.append((stamp, msg))
    return topic_msgs

def reduce_data(barrier_messages):
    # extract the relevant data from the messages
    barrier_times = []
    barrier_data = []

    for stamp, msg in barrier_messages:
        barrier_times.append(stamp)
        barrier_data.append(msg.data)

    return np.array(barrier_times), np.array(barrier_data)

def main():
    filepath = os.path.dirname(os.path.abspath(__file__))
    rosbag_path = f"{filepath}/../data/node_cbf_validation/barrier"
    lyapunov_messages = load_messages(rosbag_path, '/barrer', 'std_msgs/msg/Float32')
    barrier_times, barrier_data= reduce_data(lyapunov_messages)

    # Normalize timestamps relative to start time
    barrier_times = barrier_times - barrier_times[0]

    # --- Plot Settings ---
    plt.style.use('bmh')  # or 'ggplot', 'bmh', 'fivethirtyeight'
    mpl.rcParams['figure.dpi'] = 150
    mpl.rcParams['font.size'] = 12
    mpl.rcParams['axes.labelsize'] = 14
    mpl.rcParams['axes.titlesize'] = 16
    mpl.rcParams['legend.fontsize'] = 12
    mpl.rcParams['grid.alpha'] = 0.3

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(barrier_times[50:-50], barrier_data[50:-50], label='Barrier Value', color='#007acc', linewidth=2)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Barrier Value")
    ax.set_title('Barrier Function')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend(loc='best')

    # Optional: dark background
    # plt.style.use('dark_background')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
