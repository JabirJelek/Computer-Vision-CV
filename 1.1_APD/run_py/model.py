import cv2
from ultralytics import YOLO


def get_color(class_name):
    # Simple deterministic color generator
    hash_val = hash(class_name) % 256
    return (hash_val * 50 % 256, hash_val * 30 % 256, hash_val * 103 % 256)

def predict_and_detect(chosen_model, img, conf=0.5):
    """Runs prediction and draws bounding boxes on the image."""
    results = chosen_model.predict(img, conf=conf)
    for result in results:
        for box in result.boxes:
            # Get class name
            class_name = result.names[int(box.cls[0])]
            # Get color from map, default to white if class not found
            color = get_color(class_name)
            
            # Draw bounding box
            cv2.rectangle(img,
                          (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])),
                          color, 2)
            # Draw label
            cv2.putText(img, f"{class_name} {box.conf[0]:.2f}",
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, color, 1)
    return img

# Load model and video
model = YOLO(r"D:\RaihanFarid\Dokumen\Object Detection\CV_model\v2-apd.torchscript")
video_path = r"D:\RaihanFarid\Dokumen\Object Detection\1.1_APD\extractor\apd.mp4"
cap = cv2.VideoCapture(video_path)

# Setup VideoWriter to save the result
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (frame_width, frame_height))

# Process each frame
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Run inference and draw on the frame
    result_frame = predict_and_detect(model, frame, conf=0.5)
    
    # Save the frame to the output video
    out.write(result_frame)
    
    # Optional: Display the frame (close window to stop)
    cv2.imshow("YOLO11 Inference", result_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()