import cv2
import numpy as np
from ultralytics import YOLO

def detect_objects(frame, model, confidence_threshold=0.3):
    # Run object detection on the frame using the provided YOLO model
    results = model(frame)  # YOLOv8 performs detection on the frame
    
    detections = []
    for result in results:
        boxes = result.boxes  # Get bounding boxes
        for box in boxes:
            # Coordinates of the bounding box
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            confidence = box.conf[0].cpu().numpy()  # Confidence score
            class_id = int(box.cls[0].cpu().numpy())  # Class ID (e.g., 'person' -> 0)

            # Add to detections if confidence is high enough (adjust threshold if needed)
            if confidence > confidence_threshold:
                detections.append([x1, y1, x2, y2, confidence, class_id])

            # Customize label and color
            label = result.names[class_id]
            label_color = (0, 255, 0)
            font = cv2.FONT_HERSHEY_SIMPLEX

            # Draw bounding boxes for visualization
            cv2.rectangle(frame, (x1, y1), (x2, y2), label_color, 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10),
                        font, 0.5, label_color, 2)

    return detections, frame  # Return both detections and the annotated frame
