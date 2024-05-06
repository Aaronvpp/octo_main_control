import os
import json
import pickle
from datetime import datetime
import glob

def load_uwb_data_from_pickle(pickle_path):
    data_dicts = []
    with open(pickle_path, 'rb') as f:
        while True:
            try:
                data = pickle.load(f)
                # Convert the timestamp from string to datetime.datetime
                # data['timestamp'] = datetime.strptime(data['timestamp'], '%Y-%m-%d %H:%M:%S.%f')
                data_dicts.append(data)
            except EOFError:
                break
    return data_dicts

def find_closest_timestamp(target, data_dicts):
    timestamps = [data['timestamp'] for data in data_dicts]
    closest_timestamp = min(timestamps, key=lambda x: abs(x - target))
    return closest_timestamp

def extract_and_save_segments(movements, uwb_data, output_directory, uwb_pickle_path):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    pickle_suffix = os.path.basename(uwb_pickle_path).split('.')[0]
    for i, movement in enumerate(movements):
        start_closest = find_closest_timestamp(movement['start'], uwb_data)
        end_closest = find_closest_timestamp(movement['end'], uwb_data)
        print(start_closest)
        
        start_index = next(i for i, data_dict in enumerate(uwb_data) 
                           if data_dict['timestamp'] == start_closest)
        end_index = next(i for i, data_dict in enumerate(uwb_data) 
                         if data_dict['timestamp'] == end_closest)
        
        segment_data = uwb_data[start_index:end_index + 1]
        output_filename = f"segment_{i+1}_{start_closest.strftime('%Y%m%d%H%M%S')}_to_{end_closest.strftime('%Y%m%d%H%M%S')}_{pickle_suffix}.pickle"
        output_path = os.path.join(output_directory, output_filename)
        
        with open(output_path, 'wb') as f:
            for data in segment_data:
                pickle.dump(data, f)
# Paths
uwb_pickle_path = '/home/aiot/Desktop/Dongsheng/Octonet_Server/node_1/uwb/data/20240412125611_node_1_modality_uwb_subject_1_activity_2node,allmodalities, sysn and segment test_trial_4.pickle'
movements_json_path = '/home/aiot/Desktop/Dongsheng/Octonet_Server/node_1/depthCamera/data/20240412125611_node_1_modality_depthcam_subject_1_activity_2node,allmodalities, sysn and segment test_trial_4/20240412125611_node_1_modality_rgbcam_subject_1_activity_2node,allmodalities, sysn and segment test_trial_4_datetime.json'
output_directory = os.path.dirname(uwb_pickle_path)

# Load uwb data
uwb_data = load_uwb_data_from_pickle(uwb_pickle_path)
print(uwb_data)
# Load movements from JSON
with open(movements_json_path, 'r') as f:
    movements = json.load(f)

# Convert JSON timestamps to datetime
for movement in movements:
    movement['start'] = datetime.strptime(movement['start'], '%Y-%m-%d %H:%M:%S.%f')
    movement['end'] = datetime.strptime(movement['end'], '%Y-%m-%d %H:%M:%S.%f')

extract_and_save_segments(movements, uwb_data, output_directory, uwb_pickle_path)