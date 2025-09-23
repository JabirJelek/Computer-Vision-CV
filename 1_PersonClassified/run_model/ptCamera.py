import cv2
from ultralytics import YOLO
from pathlib import Path
import pygame
import threading
import time
from collections import Counter


# Initialize pygame mixer
pygame.mixer.init()

# Load your custom trained model
model_path = Path(r"C:\Farid\Dokumen\Object Detection\CV_model\task1.torchscript")
model = YOLO(str(model_path), task='detect')

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set resolution
cap.set(3, 640)  # width
cap.set(4, 480)  # height

# Class configuration
class_color_map = {
    0: (0, 0, 255),    # Red (class 0)
    1: (0, 255, 0),    # Green (class 1)
    # Add more classes as needed
}

# Counter configuration
CONF_THRESHOLD = 0.5
cumulative_counts = Counter()

# Alert configuration
ALERT_COOLDOWN = 5
class_cooldowns = {0: 10, 1: 5, 2: 15}  # Different cooldowns per class  # Seconds between alerts (change to 5 if preferred)
alert_timers = {}  # Track last alert time for each class

# Audio alert function
def play_audio_alert():
    try:
        sound = pygame.mixer.Sound(r"C:\Farid\Dokumen\Object Detection\usedAudio\notification-alert-269289.mp3")
        sound.play()
    except Exception as e:
        print(f"Audio error: {e}")

# Function to check if alert should be triggered
def should_trigger_alert(cls_id):
    current_time = time.time()
    cooldown = class_cooldowns.get(cls_id, ALERT_COOLDOWN) # Default if not specified
    
    # If we haven't alerted for this class before, or cooldown has passed
    if cls_id not in alert_timers or (current_time - alert_timers[cls_id]) >= ALERT_COOLDOWN:
        alert_timers[cls_id] = current_time
        return True
    return False

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Run YOLOv8 inference
        results = model(frame, stream=True)


        # Counter detections
        frame_counts = Counter() # counts for this frame

        # Process detections
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                
                if conf > CONF_THRESHOLD:
                    # Check if we should trigger alert for this class
                    if cls_id in [0, 1, 2] and should_trigger_alert(cls_id):  # Class 0
                        # Use threading for non-blocking audio
                        audio_thread = threading.Thread(target=play_audio_alert)
                        audio_thread.daemon = True
                        audio_thread.start()

                    # Visualization code
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    color = class_color_map.get(cls_id, (255, 255, 255))
                    class_name = model.names[cls_id]
                    
                    # Draw bounding box and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{class_name} {conf:.2f}"
                    
                    # Label background
                    (text_width, text_height), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                    )
                    cv2.rectangle(
                        frame, 
                        (x1, y1 - text_height - 10), 
                        (x1 + text_width, y1), 
                        color, 
                        -1
                    )
                    
                    # Label text
                    cv2.putText(
                        frame, 
                        label, 
                        (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, 
                        (255, 255, 255), 
                        2
                    )

                    # Count this detection by class name (not including confidence)
                    frame_counts[class_name] += 1

        # Update Cumulative Counts
        cumulative_counts.update(frame_counts)

        # Overlay the counts panel (top-left)
        x0, y0 = 10, 30
        line_height = 22
        padding = 8

        # Prepare text lines
        lines = []
        lines.append("Counts (frame):")
        if frame_counts:
            for name, cnt in frame_counts.items():
                lines.append(f"{name}: {cnt}")
        else:
            lines.append("None")

        lines.append("")  # blank line
        lines.append("Cumulative:")
        if cumulative_counts:
            for name, cnt in cumulative_counts.items():
                lines.append(f"{name}: {cnt}")
        else:
            lines.append("None")

        # Compute panel size
        panel_w = 250
        panel_h = line_height * len(lines) + padding

        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0 - padding//2, y0 - line_height),
                      (x0 + panel_w, y0 - line_height + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Put text lines
        y = y0
        for line in lines:
            cv2.putText(frame, line, (x0, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            y += line_height

        # Display cooldown status on screen
        current_time = time.time()
        for i, (cls_id, last_alert) in enumerate(alert_timers.items()):
            time_remaining = max(0, ALERT_COOLDOWN - (current_time - last_alert))
            status_text = f"Class {cls_id} alert: {time_remaining:.1f}s cooldown"
            cv2.putText(frame, status_text, (10, 30 + i*30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow('YOLOv8 Live Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
finally:
    cap.release()
    cv2.destroyAllWindows()