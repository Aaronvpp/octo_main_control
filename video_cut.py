from datetime import datetime
import cv2
import json
import os


def read_timestamps(timestamp_path):
    timestamps = []
    with open(timestamp_path, 'r') as f:
        for line in f:
            timestamps.append(datetime.strptime(line.strip(), '%Y-%m-%d %H:%M:%S.%f'))
    return timestamps

def find_closest_timestamp(target, timestamps):
    closest = min(timestamps, key=lambda x: abs(x - target))
    return closest

def get_video_properties(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return None, None, None
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, frame_width, frame_height

def cut_video_segments(video_path, movements, original_timestamp, output_path):
    fps, frame_width, frame_height = get_video_properties(video_path)
    if not all([fps, frame_width, frame_height]):
        return  # Exit if video properties could not be determined
    
    
    # Extract the desired suffix from the original video filename
    video_suffix = os.path.basename(video_path).split('.')[0]  # Removes the extension and keeps the filename

    for i, movement in enumerate(movements):
        start_closest = find_closest_timestamp(movement['start'], original_timestamp)
        end_closest = find_closest_timestamp(movement['end'], original_timestamp)
        
        start_frame = original_timestamp.index(start_closest)  # Use index in the timestamp list as frame number
        end_frame = original_timestamp.index(end_closest)

        
        base_filename = f"segment_{i+1}_{start_closest.strftime('%Y%m%d%H%M%S')}_to_{end_closest.strftime('%Y%m%d%H%M%S')}_{video_suffix}"
        video_filename = f"{base_filename}.mp4"
        timestamp_filename = f"{base_filename}.txt"

        output_video_path = os.path.join(output_path, video_filename)
        timestamp_path = os.path.join(output_path, timestamp_filename)
        
        video_format = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, video_format, fps, (frame_width, frame_height))
        
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        # for _ in range(start_frame, end_frame + 1):
        #     ret, frame = cap.read()
        #     if not ret:
        #         break
        #     video_writer.write(frame)
        with open(timestamp_path, 'w') as timestamp_file:
            for frame_number in range(start_frame, end_frame + 1):
                ret, frame = cap.read()
                if not ret:
                    break
                video_writer.write(frame)
                # Write the timestamp for the current frame
                timestamp_file.write(original_timestamp[frame_number].strftime('%Y-%m-%d %H:%M:%S.%f') + '\n')
        
        video_writer.release()
        cap.release()

video_path = '/home/aiot/Desktop/Dongsheng/Octonet_Server/node_5/depthCamera/data/20240430173245_node_5_modality_depthcam_subject_1_activity_one2ten_trial_22/20240430173245_node_5_modality_rgbcam_subject_1_activity_one2ten_trial_22.mp4'
depth_path = '/home/aiot/Desktop/Dongsheng/Octonet_Server/node_1/depthCamera/data/20240419140333_node_1_modality_depthcam_subject_1_activity_shout1-6_trial_10/20240419140333_node_1_modality_depthcam_subject_1_activity_shout1-6_trial_10.mp4'
output_path = os.path.dirname(video_path)
timestamp_path = '/home/aiot/Desktop/Dongsheng/Octonet_Server/node_5/depthCamera/data/20240430173245_node_5_modality_depthcam_subject_1_activity_one2ten_trial_22/20240430173245_node_5_modality_depthcam_subject_1_activity_one2ten_trial_22.txt' 
jason_path = '/home/aiot/Desktop/Dongsheng/Octonet_Server/node_1/depthCamera/data/20240430173245_node_1_modality_depthcam_subject_1_activity_one2ten_trial_22/20240430173245_node_1_modality_rgbcam_subject_1_activity_one2ten_trial_22_datetime.json'
# Ensure the output directory exists
if not os.path.exists(output_path):
    os.makedirs(output_path)
# Process
original_timestamps = read_timestamps(timestamp_path)
# Load the movements from the JSON file
try:
    with open(jason_path, 'r') as f:
        movements = json.load(f)
except Exception as e:
    print(f"Failed to load movements from JSON: {e}")
    exit(1)
    

# Convert JSON timestamps to datetime and adjust to find closest thermal camera timestamps
for movement in movements:
    movement['start'] = datetime.strptime(movement['start'], '%Y-%m-%d %H:%M:%S.%f')
    movement['end'] = datetime.strptime(movement['end'], '%Y-%m-%d %H:%M:%S.%f')
    # Adjust start and end times using the closest thermal timestamps
    movement['start'] = find_closest_timestamp(movement['start'], original_timestamps)
    movement['end'] = find_closest_timestamp(movement['end'], original_timestamps)
# Process the RGB video
cut_video_segments(video_path, movements, original_timestamps, output_path)

# Process the depth video
# cut_video_segments(depth_path, movements, original_timestamps, output_path)

# import cv2
# import json
# import os
# from datetime import datetime

# def read_timestamps(timestamp_path):
#     timestamps = []
#     with open(timestamp_path, 'r') as f:
#         for line in f:
#             timestamps.append(datetime.strptime(line.strip(), '%Y-%m-%d %H:%M:%S.%f'))
#     return timestamps

# def find_closest_timestamp(target, timestamps):
#     closest = min(timestamps, key=lambda x: abs(x - target))
#     return closest

# def cut_video_segments_with_timestamps(path, timestamp_path, movements, output_path):
#     fps, frame_width, frame_height = get_video_properties(path)
#     if not all([fps, frame_width, frame_height]):
#         return  # Exit if video properties could not be determined
    
#     video_timestamps = read_timestamps(timestamp_path)

#     for i, movement in enumerate(movements):
#         start_closest = find_closest_timestamp(movement['start'], video_timestamps)
#         end_closest = find_closest_timestamp(movement['end'], video_timestamps)
        
#         start_frame_index = video_timestamps.index(start_closest)
#         end_frame_index = video_timestamps.index(end_closest)

#         base_filename = f"segment_{i+1}_{start_closest.strftime('%Y%m%d%H%M%S')}_to_{end_closest.strftime('%Y%m%d%H%M%S')}_{os.path.basename(path).split('.')[0]}"
#         video_filename = f"{base_filename}.mp4"
#         timestamp_filename = f"{base_filename}.txt"

#         path = os.path.join(output_path, video_filename)
#         print("output_path",output_path)
#         print("timestamp_file", timestamp_filename)
#         timestamp_path = os.path.join(output_path, timestamp_filename)
        
#         video_writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
        
#         cap = cv2.VideoCapture(path)
#         cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_index)
        
#         with open(timestamp_path, 'w') as timestamp_file:
#             for _ in range(start_frame_index, end_frame_index + 1):
#                 ret, frame = cap.read()
#                 if not ret:
#                     break
#                 video_writer.write(frame)
#                 timestamp_file.write(video_timestamps[start_frame_index].strftime('%Y-%m-%d %H:%M:%S.%f') + '\n')
#                 start_frame_index += 1  # Increment to get the next timestamp
        
#         video_writer.release()
#         cap.release()

# # Convert JSON timestamps to datetime
# def adjust_movements_with_json(movements_json_path):
#     with open(movements_json_path, 'r') as f:
#         movements = json.load(f)
    
#     for movement in movements:
#         movement['start'] = datetime.strptime(movement['start'], '%Y-%m-%d %H:%M:%S.%f')
#         movement['end'] = datetime.strptime(movement['end'], '%Y-%m-%d %H:%M:%S.%f')
#     return movements


# # Assuming paths and prefixes are defined
# movements = adjust_movements_with_json(jason_path)  # Correct the variable name if necessary

# timestamp_path_for_modality = '/home/aiot/Desktop/Dongsheng/Octonet_Server/node_1/depthCamera/data/20240412125611_node_1_modality_depthcam_subject_1_activity_2node,allmodalities, sysn and segment test_trial_4/20240412125611_node_1_modality_rgbcam_subject_1_activity_2node,allmodalities, sysn and segment test_trial_4.txt' # Update this path for each modality accordingly

# # Now, call cut_video_segments_with_timestamps for each modality
# cut_video_segments_with_timestamps(video_path, timestamp_path_for_modality, movements, output_path)
# cut_video_segments_with_timestamps(depth_path, timestamp_path_for_modality, movements, output_path)