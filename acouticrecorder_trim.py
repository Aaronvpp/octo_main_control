import re
from datetime import datetime, timedelta
import json
from pydub import AudioSegment
import os

def extract_start_timestamp(log_file_path):
    with open(log_file_path, 'r') as file:
        for line in file:
            if "Start Recording" in line:
                timestamp_str = re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}", line).group()
                return datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f')
    return None

def load_segments(json_file_path):
    with open(json_file_path, 'r') as file:
        return json.load(file)

def calculate_differences(segments, recording_start_timestamp):
    # Convert recording start timestamp to milliseconds
    recording_start_ms = recording_start_timestamp.timestamp() * 1000
    for segment in segments:
        # Parse segment start and end timestamps
        segment_start_timestamp = datetime.strptime(segment['start'], '%Y-%m-%d %H:%M:%S.%f')
        segment_end_timestamp = datetime.strptime(segment['end'], '%Y-%m-%d %H:%M:%S.%f')
        # Calculate differences in milliseconds
        segment['start_diff_ms'] = (segment_start_timestamp.timestamp() * 1000) - recording_start_ms
        segment['end_diff_ms'] = (segment_end_timestamp.timestamp() * 1000) - recording_start_ms
    return segments

def trim_audio(file_path, start_diff_ms, end_diff_ms, output_file_path):
    audio = AudioSegment.from_wav(file_path)
    # Trim using the calculated differences as start and end points
    trimmed_audio = audio[start_diff_ms:end_diff_ms]
    trimmed_audio.export(output_file_path, format="wav")

log_file_path = '/home/aiot/Desktop/Dongsheng/Octonet_Server/node_4/acoustic/data/20240418210708_node_4_modality_acoustic_subject_1_activity_shout1-6_trial_10/20240418210708_node_4_modality_acoustic_subject_1_activity_shout1-6_trial_10.log' 
json_file_path = '/home/aiot/Desktop/Dongsheng/Octonet_Server/node_1/depthCamera/data/20240418210708_node_1_modality_depthcam_subject_1_activity_shout1-6_trial_10/20240418210708_node_1_modality_rgbcam_subject_1_activity_shout1-6_trial_10_datetime.json'
wav_file_path = '/home/aiot/Desktop/Dongsheng/Octonet_Server/node_4/acoustic/data/20240418210708_node_4_modality_acoustic_subject_1_activity_shout1-6_trial_10/20240418210708_node_4_modality_acoustic_subject_1_activity_shout1-6_trial_10.wav'
output_directory = os.path.dirname(wav_file_path)
wav_suffix = os.path.basename(wav_file_path).split('.')[0]
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Execution
recording_start_timestamp = extract_start_timestamp(log_file_path)
segments = load_segments(json_file_path)
segments_with_diffs = calculate_differences(segments, recording_start_timestamp)

for i, segment in enumerate(segments_with_diffs):
    output_file_name = f"segment_{i+1}_{datetime.strptime(segment['start'], '%Y-%m-%d %H:%M:%S.%f').strftime('%Y%m%d%H%M%S')}_to_{datetime.strptime(segment['end'], '%Y-%m-%d %H:%M:%S.%f').strftime('%Y%m%d%H%M%S')}_{wav_suffix}.wav"
    output_file_path = os.path.join(output_directory, output_file_name)
    trim_audio(wav_file_path, int(segment['start_diff_ms']), int(segment['end_diff_ms']), output_file_path)