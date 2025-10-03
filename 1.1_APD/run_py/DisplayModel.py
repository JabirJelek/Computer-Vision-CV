import cv2
from ultralytics import YOLO
import numpy as np

# Configuration - Easy to modify
CLASS_CONFIG = {
    'person_with_helmet_forklift': {
        'display': True,
        'color': (0, 0, 255),  # Red
        'placeholder': 'NO MASK ALERT!',
        'box_thickness': 3
    },
    'person_with_mask_forklift': {
        'display': True,
        'color': (0, 0, 255),  # Red
        'placeholder': 'NO HELMET ALERT!',
        'box_thickness': 3
    },
    'person_without_mask_helmet_forklift': {
        'display': True,
        'color': (0, 0, 255),  # Red
        'placeholder': 'MISSING PROTECTION',
        'box_thickness': 3
    },
    'person_without_mask_nonForklift': {
        'display': True,
        'color': (0, 0, 255),  # Red
        'placeholder': 'NO MASK ALERT!',
        'box_thickness': 3
    },
    # Add other classes here if you want to display them with different settings
    'person_with_mask_helmet_forklift': {
        'display': False,  # Set to True if you want to display this class
        'color': (0, 255, 255),  # Yellow
        'placeholder': 'FULL PROTECTION',
        'box_thickness': 2
    }
}

def predict_and_detect_advanced(chosen_model, img, conf=0.5):
    """Advanced version with per-class configuration"""
    results = chosen_model.predict(img, conf=conf)
    
    for result in results:
        for box in result.boxes:
            # Get class name
            class_name = result.names[int(box.cls[0])]
            confidence = float(box.conf[0])
            
            # Skip if class not in config or display is False
            if class_name not in CLASS_CONFIG or not CLASS_CONFIG[class_name]['display']:
                continue
            
            # Get configuration for this class
            config = CLASS_CONFIG[class_name]
            
            # Get coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = config['color']
            placeholder = config['placeholder']
            thickness = config['box_thickness']
            
            # Create display text
            display_text = f"{placeholder} {confidence:.2f}"
            
            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            
            # Calculate text size
            (text_width, text_height), baseline = cv2.getTextSize(
                display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            
            # Draw background for text
            cv2.rectangle(img, 
                         (x1, y1 - text_height - baseline - 5),
                         (x1 + text_width, y1),
                         color, -1)
            
            # Draw text
            cv2.putText(img, display_text,
                       (x1, y1 - baseline - 3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return img

# Load model and video
model = YOLO(r"D:\RaihanFarid\Dokumen\Object Detection\CV_model\v2-apd.torchscript")
video_path = r"D:\RaihanFarid\Dokumen\Object Detection\history_dataset\3-APDDetection\apd.mp4"
cap = cv2.VideoCapture(video_path)

# Setup VideoWriter
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('output_video_advanced.mp4', fourcc, fps, (frame_width, frame_height))

print("Processing video with advanced configuration...")
print("Classes being displayed:")
for class_name, config in CLASS_CONFIG.items():
    if config['display']:
        print(f"  {class_name} -> '{config['placeholder']}'")

# Process each frame
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Run inference with advanced detection
    result_frame = predict_and_detect_advanced(model, frame, conf=0.5)
    
    # Save the frame
    out.write(result_frame)
    
    # Display the frame
    cv2.imshow("YOLO Advanced Detection", result_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Video processing completed!")
print(f"Output saved as: output_video_advanced.mp4")