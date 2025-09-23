import cv2
from ultralytics import YOLO
from pathlib import Path
from collections import Counter
import time

class RTSPDetector:
    def __init__(self, model_path, rtsp_url, conf_threshold=0.5):
        """
        Initialize RTSP detector with YOLOv8 model
        
        Args:
            model_path (str): Path to .torchscript model
            rtsp_url (str): RTSP stream URL
            conf_threshold (float): Confidence threshold for detections
        """
        self.model_path = Path(model_path)
        self.rtsp_url = rtsp_url
        self.conf_threshold = conf_threshold
        self.cumulative_counts = Counter()
        
        # Initialize model
        self.model = YOLO(str(self.model_path), task='detect')
        
        # Initialize video capture
        self.cap = None
        self.initialize_capture()
        
    def initialize_capture(self):
        """Initialize or reinitialize RTSP connection"""
        if self.cap is not None:
            self.cap.release()
            
        self.cap = cv2.VideoCapture(self.rtsp_url)
        
        # Set RTSP parameters for better stability
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer
        self.cap.set(cv2.CAP_PROP_FPS, 30)  # Try to set FPS
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Optional: set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        return self.cap.isOpened()
    
    def process_detections(self, frame):
        """
        Process frame through YOLO model and return detections
        
        Returns:
            tuple: (processed_frame, frame_counts)
        """
        frame_counts = Counter()
        
        try:
            # Run inference
            results = self.model(frame, stream=True, verbose=False)
            
            # Process each detection
            for r in results:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    if conf > self.conf_threshold:
                        cls_id = int(box.cls[0])
                        class_name = self.model.names[cls_id]
                        
                        # Draw bounding box
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        label = f"{class_name} {conf:.2f}"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Count detection
                        frame_counts[class_name] += 1
            
            # Update cumulative counts
            self.cumulative_counts.update(frame_counts)
            
        except Exception as e:
            print(f"Error during detection: {e}")
            
        return frame, frame_counts
    
    def display_info_panel(self, frame, frame_counts):
        """Display information panel with detection counts"""
        lines = ["Real-time Counts:"]
        
        # Add frame counts
        if frame_counts:
            for name, count in frame_counts.items():
                lines.append(f"{name}: {count}")
        else:
            lines.append("None detected")
        
        lines.append("")
        lines.append("Cumulative Counts:")
        
        # Add cumulative counts
        if self.cumulative_counts:
            for name, count in self.cumulative_counts.items():
                lines.append(f"{name}: {count}")
        else:
            lines.append("None detected")
        
        # Panel dimensions
        x0, y0 = 10, 30
        line_height = 22
        padding = 8
        panel_width = 250
        panel_height = len(lines) * line_height + padding
        
        # Create semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0 - padding//2, y0 - line_height),
                     (x0 + panel_width, y0 - line_height + panel_height), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Add text
        y_offset = y0
        for line in lines:
            cv2.putText(frame, line, (x0, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            y_offset += line_height
    
    def run(self, reconnect_delay=5):
        """Main processing loop with reconnection capability"""
        consecutive_failures = 0
        max_failures = 3
        
        while True:
            # Check connection status
            if not self.cap.isOpened() or consecutive_failures >= max_failures:
                print("Attempting to reconnect to RTSP stream...")
                if self.initialize_capture():
                    consecutive_failures = 0
                    print("Reconnection successful")
                else:
                    print(f"Reconnection failed. Retrying in {reconnect_delay} seconds...")
                    time.sleep(reconnect_delay)
                    continue
            
            # Read frame
            ret, frame = self.cap.read()
            if not ret:
                consecutive_failures += 1
                print(f"Failed to read frame ({consecutive_failures}/{max_failures})")
                time.sleep(1)
                continue
            
            # Reset failure counter on successful frame read
            consecutive_failures = 0
            
            # Process frame
            processed_frame, frame_counts = self.process_detections(frame)
            
            # Display information
            self.display_info_panel(processed_frame, frame_counts)
            
            # Show output
            frame = cv2.resize(processed_frame, (1024, 576))
            cv2.imshow('RTSP YOLOv8 Detection', frame)
            
            # Check for exit command
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    # Configuration
    MODEL_PATH = r"C:\Farid\Dokumen\Object Detection\1_PersonClassified\CV_model\CCTV.torchscript"
    RTSP_URL = "rtsp://admin:CemaraMas2025!@192.168.2.190:554/Streaming/Channels/101"
    CONF_THRESHOLD = 0.5
    
    # Create and run detector
    detector = RTSPDetector(MODEL_PATH, RTSP_URL, CONF_THRESHOLD)
    detector.run()

if __name__ == "__main__":
    main()