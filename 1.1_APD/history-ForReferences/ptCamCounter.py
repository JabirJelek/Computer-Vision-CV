import cv2
from ultralytics import YOLO
from pathlib import Path
from collections import Counter

# Load your custom trained model
model_path = Path(r"D:\RaihanFarid\Dokumen\Object Detection\CV_model\bestSmallS.torchscript")
model = YOLO(str(model_path), task='detect')

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Optional: set resolution
cap.set(3, 1000)  # width
cap.set(4, 500)  # height

# Counter of each object
CONF_THRESHOLD = 0.5
cumulative_counts = Counter()

# Assign fixed color map - using class IDs as keys
class_color_map = {
    0: (0, 0, 255),    # Assuming class 0 is 'person' (Red in BGR)
    1: (0, 255, 0),    # Assuming class 1 is 'glasses' (Green in BGR)
    # Add more classes as needed
}
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Run YOLOv8 inference (stream=True yields results for the frame)
        results = model(frame, stream=True)

        # Process detections
        frame_counts = Counter()  # counts for this frame

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Extract confidence and class ID                
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])

                # Get class name                
                class_name = model.names[cls_id]
                
                # Create label with class name and confidence
                label = f"{class_name} {conf:.2f}"

                # Only process detections with confidence > 0.5
                if conf > CONF_THRESHOLD:

                    # Get color for this class - default to white if class not in map
                    color = class_color_map.get(cls_id, (255, 255, 255))

                    # Draw bbox + label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2) # Start change from this

                    
                    # Draw label background
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

                    # Put text
                    cv2.putText(
                        frame, 
                        label, 
                        (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, 
                        (255, 255, 255),  # White text
                        2
                    )


                    # Count this detection by class name (not including confidence)
                    frame_counts[class_name] += 1

        # Update cumulative counts
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

        cv2.imshow('YOLOv8 Live Detection', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Release resources outside the loop
    cap.release()
    cv2.destroyAllWindows()