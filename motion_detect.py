import cv2
import json
import os
from datetime import datetime, timedelta

def detect_motion(video_path, buffer_time=2.0, sensitivity=500):
    """
    Detects motion in a video and saves the start and end times of movements to a JSON file.

    Parameters:
    - video_path: Path to the video file.
    - buffer_time: Time in seconds to buffer the end of a movement detection.
    - sensitivity: The minimum area for a contour to be considered motion.

    Returns:
    - A JSON file path where movements are saved.
    """
    camera = cv2.VideoCapture(video_path)
    if not camera.isOpened():
        print('Failed to open video')
        return

    pre_frame = None
    movement_detected = False
    movements = []
    last_movement_time = 0

    while True:
        grabbed, frame = camera.read()
        if not grabbed:
            if movement_detected and (camera.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 - last_movement_time >= buffer_time):
                end_time = last_movement_time
                movements.append((start_time, end_time))
                movement_detected = False
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

        if pre_frame is None:
            pre_frame = gray_frame
            continue

        img_delta = cv2.absdiff(pre_frame, gray_frame)
        thresh = cv2.threshold(img_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        current_time = camera.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        movement_now = False
        for c in contours:
            if cv2.contourArea(c) < sensitivity:
                continue
            movement_now = True
            last_movement_time = current_time
            if not movement_detected:
                start_time = current_time
                movement_detected = True
            break

        if movement_detected and not movement_now and (current_time - last_movement_time >= buffer_time):
            end_time = last_movement_time
            movements.append((start_time, end_time))
            movement_detected = False

        pre_frame = gray_frame

    camera.release()

    json_path = os.path.splitext(video_path)[0] + '.json'
    with open(json_path, 'w') as f:
        json.dump(movements, f, indent=4)

    return json_path

def convert_to_real_timestamps(json_path, timestamp_path):
    """
    Converts the motion detection timestamps to real-world timestamps based on a reference timestamp.

    Parameters:
    - json_path: Path to the JSON file with motion detection timestamps.
    - timestamp_path: Path to the file containing the reference timestamp.

    Returns:
    - A JSON file path where the converted timestamps are saved.
    """
    with open(timestamp_path, 'r') as f:
        first_timestamp_str = f.readline().strip()
    first_timestamp_dt = datetime.strptime(first_timestamp_str, '%Y-%m-%d %H:%M:%S.%f')

    with open(json_path, 'r') as f:
        movements = json.load(f)

    converted_movements = []
    for start_seconds, end_seconds in movements:
        start_datetime = first_timestamp_dt + timedelta(seconds=start_seconds)
        end_datetime = first_timestamp_dt + timedelta(seconds=end_seconds)
        converted_movements.append({
            "start": start_datetime.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            "end": end_datetime.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        })

    output_json_path = os.path.splitext(json_path)[0] + '_datetime.json'
    with open(output_json_path, 'w') as f:
        json.dump(converted_movements, f, indent=4)

    return output_json_path
