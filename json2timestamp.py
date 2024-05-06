from datetime import datetime, timedelta
import json
import os

def read_first_timestamp(timestamp_path):
    with open(timestamp_path, 'r') as f:
        first_line = f.readline().strip()
    return first_line

def convert_json_to_datetime(json_path, reference_datetime):
    with open(json_path, 'r') as f:
        movements = json.load(f)

    converted_movements = []
    for start_seconds, end_seconds in movements:
        start_datetime = reference_datetime + timedelta(seconds=start_seconds)
        end_datetime = reference_datetime + timedelta(seconds=end_seconds)
        
        converted_movements.append({
            "start": start_datetime.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],  # Truncate microseconds to milliseconds
            "end": end_datetime.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        })

    return converted_movements

def save_converted_movements(converted_movements, output_json_path):
    with open(output_json_path, 'w') as f:
        json.dump(converted_movements, f, indent=4)

# Paths setup
timestamp_path = '/home/aiot/Desktop/Dongsheng/Octonet_Server/node_1/depthCamera/data/20240321124852_node_1_modality_depthcam_subject_2_activity_comboforseg2_trial_4/20240321124852_node_1_modality_depthcam_subject_2_activity_comboforseg2_trial_4.txt'
video_path = '/home/aiot/Desktop/Dongsheng/Octonet_Server/node_1/depthCamera/data/20240321124852_node_1_modality_depthcam_subject_2_activity_comboforseg2_trial_4/20240321124852_node_1_modality_rgbcam_subject_2_activity_comboforseg2_trial_4.mp4'
json_path = os.path.splitext(video_path)[0] + '.json'
output_json_path = os.path.splitext(video_path)[0] + '_datetime.json'

first_timestamp_str = read_first_timestamp(timestamp_path)
first_timestamp_dt = datetime.strptime(first_timestamp_str, '%Y-%m-%d %H:%M:%S.%f')

converted_movements = convert_json_to_datetime(json_path, first_timestamp_dt)
save_converted_movements(converted_movements, output_json_path)
