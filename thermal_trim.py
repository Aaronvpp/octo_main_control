from datetime import datetime
import cv2
import json
import os

def read_first_timestamp(timestamp_path):
    with open(timestamp_path, 'r') as f:
        first_line = f.readline().strip()
    return datetime.strptime(first_line, '%Y-%m-%d %H:%M:%S.%f')

def read_timestamps(timestamp_path):
    timestamps = []
    with open(timestamp_path, 'r') as f:
        for line in f:
            timestamps.append(datetime.strptime(line.strip(), '%Y-%m-%d %H:%M:%S.%f'))
    return timestamps

def find_closest_timestamp(target, timestamps):
    closest = min(timestamps, key=lambda x: abs(x - target))
    return closest

def cut_video_segments(video_path, movements, thermal_timestamps, output_path):
    # Ensure output directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Extract the desired suffix from the original video filename
    video_suffix = os.path.basename(video_path).split('.')[0]  # Removes the extension and keeps the filename

    for i, movement in enumerate(movements):
        start_closest = find_closest_timestamp(movement['start'], thermal_timestamps)
        end_closest = find_closest_timestamp(movement['end'], thermal_timestamps)
        
        start_frame = thermal_timestamps.index(start_closest)  # Use index in the timestamp list as frame number
        end_frame = thermal_timestamps.index(end_closest)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        base_filename = f"segment_{i+1}_{start_closest.strftime('%Y%m%d%H%M%S')}_to_{end_closest.strftime('%Y%m%d%H%M%S')}_{video_suffix}"
        video_filename = f"{base_filename}.mp4"
        timestamp_filename = f"{base_filename}.txt"

        video_path = os.path.join(output_path, video_filename)
        timestamp_path = os.path.join(output_path, timestamp_filename)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

        with open(timestamp_path, 'w') as timestamp_file:
            for frame_number in range(start_frame, end_frame + 1):
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
                # Write the timestamp for the current frame
                timestamp_file.write(thermal_timestamps[frame_number].strftime('%Y-%m-%d %H:%M:%S.%f') + '\n')

        out.release()
    cap.release()

# Paths
video_path = '/home/aiot/Desktop/Dongsheng/Octonet_Server/node_1/seekThermal/data/20240418210708_node_1_modality_seekthermal_subject_1_activity_shout1-6_trial_10.mp4'
json_path = '/home/aiot/Desktop/Dongsheng/Octonet_Server/node_1/depthCamera/data/20240418210708_node_1_modality_depthcam_subject_1_activity_shout1-6_trial_10/20240418210708_node_1_modality_rgbcam_subject_1_activity_shout1-6_trial_10_datetime.json'
timestamp_path = '/home/aiot/Desktop/Dongsheng/Octonet_Server/node_1/seekThermal/data/20240418210708_node_1_modality_seekthermal_subject_1_activity_shout1-6_trial_10.txt'
output_path = os.path.dirname(video_path)

# Process
thermal_timestamps = read_timestamps(timestamp_path)

with open(json_path, 'r') as f:
    movements = json.load(f)

# Convert JSON timestamps to datetime and adjust to find closest thermal camera timestamps
for movement in movements:
    movement['start'] = datetime.strptime(movement['start'], '%Y-%m-%d %H:%M:%S.%f')
    movement['end'] = datetime.strptime(movement['end'], '%Y-%m-%d %H:%M:%S.%f')
    # Adjust start and end times using the closest thermal timestamps
    movement['start'] = find_closest_timestamp(movement['start'], thermal_timestamps)
    movement['end'] = find_closest_timestamp(movement['end'], thermal_timestamps)

cut_video_segments(video_path, movements, thermal_timestamps, output_path)
