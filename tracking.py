import cv2
import numpy as np
import random
import os
import time
import csv
from datetime import datetime
from sort.sort import Sort  # SORT tracker
from ultralytics import YOLO  # YOLOv8 model

# 🚀 Initialize the YOLOv8 model
print("\033[94m[INFO]\033[0m Initializing YOLOv8 model...")
yolo_model = YOLO('yolov8n.pt')
print("\033[92m[SUCCESS]\033[0m YOLOv8 model loaded successfully!")

# 🎨 Generate a random color for each tracked object
def get_random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

# ✏️ Update object paths and draw bounding boxes, ID labels, and speed
def update_paths_and_draw(frame, tracked_objects, object_paths, object_colors, object_speeds, frame_count):
    for obj in tracked_objects:
        track_id = int(obj[4])
        x1, y1, x2, y2 = map(int, obj[:4])
        x_center = int((x1 + x2) / 2)
        y_center = int((y1 + y2) / 2)

        # Assign a unique color for each object ID
        if track_id not in object_colors:
            object_colors[track_id] = get_random_color()

        # Update the path (trail) for each object
        if track_id not in object_paths:
            object_paths[track_id] = []
        object_paths[track_id].append((x_center, y_center))

        # Draw the motion path of the object
        for i in range(1, len(object_paths[track_id])):
            cv2.line(frame, object_paths[track_id][i - 1], object_paths[track_id][i], object_colors[track_id], 2)

        speed = object_speeds.get(track_id, 0)

        # Draw bounding box and speed info
        cv2.rectangle(frame, (x1, y1), (x2, y2), object_colors[track_id], 2)
        text_y = y1 - 10
        cv2.putText(frame, f'ID {track_id}', (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, object_colors[track_id], 2)
        text_y -= 15
        cv2.putText(frame, f'Speed: {speed:.2f} m/s', (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, object_colors[track_id], 2)

    return frame

# 🧠 Main tracking loop — detects and tracks objects with YOLO and SORT
def track_objects(yolo_model, source='webcam', video_path=None, confidence_threshold=0.3, show_fps=True):
    tracker = Sort()  # Initialize SORT tracker
    object_paths = {}  # For path drawing
    object_colors = {}  # For object ID color mapping
    object_speeds = {}  # For speed tracking

    # Calibration values (adjustable)
    pixel_to_meter_ratio = 1 / 50  # 1 meter = 50 pixels
    real_time_fps = 30  # Assumed real-time frame rate
    record_fps = 20  # Output video frame rate

    # 🔄 Create output directories for video and CSV logs
    base_output_dir = os.path.join(os.getcwd(), 'detectandtrack', 'enhanced_outputs')
    os.makedirs(base_output_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    video_output_path = os.path.join(base_output_dir, f'sample_output_{timestamp}.mp4')
    csv_output_path = os.path.join(base_output_dir, f'sample_output_{timestamp}.csv')

    # 🎥 Set up input source: webcam or video
    if source == 'webcam':
        cap = cv2.VideoCapture(0)
    elif source == 'video' and video_path:
        cap = cv2.VideoCapture(video_path)
    else:
        print("\033[91m[ERROR]\033[0m Invalid video source.")
        return

    if not cap.isOpened():
        print("\033[91m[ERROR]\033[0m Could not open video source.")
        return

    # 📝 Set up video writer and CSV logger
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_output_path, fourcc, record_fps, (frame_width, frame_height))

    csv_file = open(csv_output_path, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Frame', 'Track ID', 'X1', 'Y1', 'X2', 'Y2', 'Speed (m/s)'])

    frame_count = 0
    prev_time = time.time()

    print("\033[96m[INFO]\033[0m Starting object tracking... Press 'q' to quit.")

    # 🌀 Frame-by-frame loop
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 📦 Object Detection
        detections, frame = detect_objects(frame, yolo_model, confidence_threshold)
        detections_for_tracker = np.array([[d[0], d[1], d[2], d[3]] for d in detections]) if detections else np.empty((0, 4))

        # 📍 Object Tracking
        tracked_objects = tracker.update(detections_for_tracker)

        # ⏱ FPS Display
        current_time = time.time()
        fps_display = 1 / (current_time - prev_time)
        prev_time = current_time

        if show_fps:
            cv2.putText(frame, f"Real-Time FPS: {fps_display:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # ⚡️ Speed Calculation + CSV Logging
        for obj in tracked_objects:
            track_id = int(obj[4])
            x1, y1, x2, y2 = map(int, obj[:4])
            center = ((x1 + x2) // 2, (y1 + y2) // 2)

            if track_id in object_paths and len(object_paths[track_id]) > 0:
                prev_center = object_paths[track_id][-1]
                pixel_distance = np.linalg.norm(np.array(center) - np.array(prev_center))
                speed_m_per_s = (pixel_distance * pixel_to_meter_ratio) * real_time_fps
                object_speeds[track_id] = speed_m_per_s

            csv_writer.writerow([frame_count, track_id, x1, y1, x2, y2, object_speeds.get(track_id, 0)])

        # 🎨 Draw tracks and labels
        frame = update_paths_and_draw(frame, tracked_objects, object_paths, object_colors, object_speeds, frame_count)

        # 💾 Write frame to video and display
        out.write(frame)
        cv2.imshow("Object Tracking", frame)

        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 🧹 Cleanup
    cap.release()
    out.release()
    csv_file.close()
    cv2.destroyAllWindows()

    print(f"\033[92m[OUTPUT SAVED]\033[0m Video saved to: {video_output_path}")
    print(f"\033[92m[OUTPUT SAVED]\033[0m CSV log saved to: {csv_output_path}")

# 📦 YOLOv8 Detection Function
def detect_objects(frame, model, confidence_threshold=0.3):
    results = model(frame)
    detections = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            confidence = box.conf[0].cpu().numpy()

            if confidence > confidence_threshold:
                detections.append([x1, y1, x2, y2, confidence])

            # 🏷 Draw label and confidence
            label = result.names[int(box.cls)]
            label_color = (0, 255, 0)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.rectangle(frame, (x1, y1), (x2, y2), label_color, 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10), font, 0.5, label_color, 2)

    return detections, frame
