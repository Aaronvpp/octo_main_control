import cv2
import json
import os
# Load the video
video_path = '/home/aiot/Desktop/Dongsheng/Octonet_Server/node_1/depthCamera/data/20240321124852_node_1_modality_depthcam_subject_2_activity_comboforseg2_trial_4/20240321124852_node_1_modality_rgbcam_subject_2_activity_comboforseg2_trial_4.mp4'


camera = cv2.VideoCapture(video_path)

if camera.isOpened():
    print('Video Opened')
else:
    print('Failed to open video')

pre_frame = None
movement_detected = False
movements = []  # List to store start and end times of movements
buffer_time = 2.0  # Buffer time in seconds
last_movement_time = 0  # Time of the last detected movement

while True:
    grabbed, frame = camera.read()
    if not grabbed:
        # Check if movement was detected but not concluded due to buffer time
        if movement_detected and (camera.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 - last_movement_time >= buffer_time):
            end_time = last_movement_time  # Use the last movement time as end time
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
        if cv2.contourArea(c) < 500:  # Adjust sensitivity
            continue
        movement_now = True
        last_movement_time = current_time  # Update last movement time
        if not movement_detected:
            # Movement starts
            start_time = current_time
            movement_detected = True
        break

    # Check if movement ended considering buffer time
    if movement_detected and not movement_now and (current_time - last_movement_time >= buffer_time):
        # Movement ends
        end_time = last_movement_time
        movements.append((start_time, end_time))
        movement_detected = False

    pre_frame = gray_frame

    # Visualize movement status and times on the video
    text = "Movement: Yes" if movement_detected else "Movement: No"
    cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Optionally, display start and end times
    if movement_detected:
        cv2.putText(frame, f"Start Time: {start_time:.2f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    if len(movements) > 0 and not movement_detected:
        # Display the end time of the last recorded movement
        cv2.putText(frame, f"End Time: {movements[-1][1]:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

# Print all recorded movements
for start, end in movements:
    print(f"Movement from {start:.2f} to {end:.2f} seconds")

# Construct the JSON file path in the same directory as the video
json_path = os.path.splitext(video_path)[0] + '.json'
# Save the movements to a JSON file
with open(json_path, 'w') as f:
    json.dump(movements, f, indent=4)
