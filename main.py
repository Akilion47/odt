# main.py
from tracking import track_objects
from ultralytics import YOLO
from detection import detect_objects
import cv2
import time

# Styling
GREEN = '\033[92m'
RED = '\033[91m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
RESET = '\033[0m'
BOLD = '\033[1m'

def main():
    print(f"{BOLD}{CYAN}🔄👁️ Welcome to YOLOv8 Object Detection & Tracking Interface 🎯📹🚀{RESET}")
    print()

    # Let the user choose the source
    source_choice = input(f"{YELLOW}🎥 Choose input source — Enter 'w' for webcam 📷 or 'v' for video 🎞️: {RESET}").lower()

    if source_choice == 'w':
        source = 'webcam'
        video_path = None
    elif source_choice == 'v':
        video_path = input(f"{YELLOW}📂 Enter the full path to the video file (e.g., C:\\Users\\Ayush\\Downloads\\video.mp4): {RESET}")
        source = 'video'
    else:
        print(f"{RED}❌🫡 Invalid mode 😵‍💫. 💀 Exiting the program... 🏃‍♂️💨{RESET}")
        return

    # Let the user choose the mode
    mode_choice = input(f"{YELLOW}🎯 Enter 'd' for detection👀 or 't' for tracking🖲️: {RESET}").lower()

    # Load YOLO model
    print(f"{CYAN}🔍 Loading YOLO model ⚙️💻🔥🫷🏻🥶🫸🏻...{RESET}")
    start_time = time.time()
    yolo_model = YOLO('yolov8n.pt')
    elapsed = time.time() - start_time
    print(f"{GREEN}✅💻🔥 YOLO model loaded successfully🫂 🫂 in {elapsed:.2f} seconds!☺️{RESET}")
    print()

    # Run detection or tracking
    if mode_choice == 'd':
        print(f"{CYAN}🧠 Running in {BOLD}detection mode🤗...{RESET}")
        run_detection(source, video_path, yolo_model)
    elif mode_choice == 't':
        print(f"{CYAN}🧠 Running in {BOLD}tracking mode🤗...{RESET}")
        track_objects(yolo_model, source, video_path)
    else:
        print(f"{RED}❌🫡 Invalid mode 😵‍💫. 💀 Exiting the program... 🏃‍♂️💨{RESET}")
        return

def run_detection(source, video_path, yolo_model):
    if source == 'webcam':
        cap = cv2.VideoCapture(0)
    elif source == 'video' and video_path:
        cap = cv2.VideoCapture(video_path)
    else:
        print(f"{RED}❌ Invalid source. Please choose 'webcam' or provide a valid video path.{RESET}")
        return

    if not cap.isOpened():
        print(f"{RED}❌ Error: Could not open video source.{RESET}")
        return

    print(f"{GREEN}📡 Detection stream started! Press 'q' to quit.{RESET}")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections, annotated_frame = detect_objects(frame, yolo_model)
        cv2.imshow("Object Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(f"{YELLOW}🛑 Quitting detection...{RESET}")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"{GREEN}✅ Detection mode exited cleanly.{RESET}")

if __name__ == "__main__":
    main()
