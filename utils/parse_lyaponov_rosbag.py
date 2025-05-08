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

def reduce_data(lyaponov_messages):
    # extract the relevant data from the messages
    lyaponov_times = []
    lyaponov_data = []

    for stamp, msg in lyaponov_messages:
        lyaponov_times.append(stamp)
        lyaponov_data.append(msg.data)

    return np.array(lyaponov_times), np.array(lyaponov_data)

def main():
    filepath = os.path.dirname(os.path.abspath(__file__))
    # rosbag_path = f"{filepath}/../data/node_clf_validation/no_translation_rotation"
    # rosbag_path = f"{filepath}/../data/node_clf_validation/translation_no_rotation"
    rosbag_path = f"{filepath}/../data/node_clf_validation/translation_rotation_away"
    # rosbag_path = f"{filepath}/../data/node_clf_validation/translation_rotation_towards"
    lyapunov_messages = load_messages(rosbag_path, '/lyapunov', 'std_msgs/msg/Float32')
    lyaponov_times, lyaponov_data= reduce_data(lyapunov_messages)

    # Normalize timestamps relative to start time
    lyaponov_times = lyaponov_times - lyaponov_times[0]

    # --- Plot Settings ---
    plt.style.use('bmh')  # or 'ggplot', 'bmh', 'fivethirtyeight'
    mpl.rcParams['figure.dpi'] = 150
    mpl.rcParams['font.size'] = 12
    mpl.rcParams['axes.labelsize'] = 14
    mpl.rcParams['axes.titlesize'] = 16
    mpl.rcParams['legend.fontsize'] = 12
    mpl.rcParams['grid.alpha'] = 0.3

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(lyaponov_times[30:-100], lyaponov_data[30:-100], label='Lyapunov Value', color='#007acc', linewidth=2)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Lyapunov Value")
    # ax.set_title('Rotation "Step" Response')
    # ax.set_title('Translation "Step" Response')
    ax.set_title('Adversarial "Step" Response')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend(loc='best')

    # Optional: dark background
    # plt.style.use('dark_background')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
