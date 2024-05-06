import json
import pandas as pd
from datetime import datetime, timedelta
import os

# Function to load segments from JSON file
def load_segments(json_file_path):
    with open(json_file_path, 'r') as file:
        return json.load(file)

# Function to parse the start timestamp from the CSV file
def parse_csv_start_timestamp(csv_file_path):
    with open(csv_file_path, 'r') as file:
        for line in file:
            if "Capture Start Time" in line:
                line_parts = line.split(',')
                timestamp_str = line_parts[11].strip()
                return datetime.strptime(timestamp_str, '%Y-%m-%d %I.%M.%S.%f %p')
    return None

# Load segments and start time
json_file_path = 'node_1/depthCamera/data/20240418210708_node_1_modality_depthcam_subject_1_activity_shout1-6_trial_10/20240418210708_node_1_modality_rgbcam_subject_1_activity_shout1-6_trial_10_datetime.json'
csv_file_path = '/home/aiot/Desktop/Dongsheng/Octonet_Server/Take 2024-04-12 12.56.10 PM2.csv'
segments = load_segments(json_file_path)
start_time_csv = parse_csv_start_timestamp(csv_file_path)

# Load CSV data
df = pd.read_csv(csv_file_path, skiprows=6)

# Calculate absolute timestamps for each row
df['Absolute Time'] = df['Time (Seconds)'].apply(lambda x: start_time_csv + timedelta(seconds=x))

# Convert JSON segment times to datetime objects
segments_datetime = [{'start_time': datetime.strptime(seg['start'], '%Y-%m-%d %H:%M:%S.%f'),
                      'end_time': datetime.strptime(seg['end'], '%Y-%m-%d %H:%M:%S.%f')} 
                      for seg in segments]

# Filter rows within the start and end times of each segment
filtered_segments = []
for segment in segments_datetime:
    segment_df = df[(df['Absolute Time'] >= segment['start_time']) & (df['Absolute Time'] <= segment['end_time'])]
    filtered_segments.append(segment_df)

# Save the filtered segments to new CSV files
output_directory = os.path.dirname(csv_file_path)

# Read the header rows line by line and store them in a list
header_rows = []
with open(csv_file_path, 'r') as file:
    for _ in range(7):  
        header_rows.append(file.readline())

for i, segment_df in enumerate(filtered_segments):
    output_file_path = f"{output_directory}/segment_{i+1}.csv"
    with open(output_file_path, 'w') as f:
        # Write header rows to the output file
        for header_line in header_rows:
            f.write(header_line)
        # Append the segment data to the output file
        segment_df.to_csv(f, header=False, index=False)
