import cv2
from ultralytics import YOLO
from pathlib import Path
import pygame
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Dict, Tuple, Any, Union

@dataclass
class AppConfig:
    """Centralized configuration for the application"""
    # Model configuration
    model_path: Path = Path(r"D:\RaihanFarid\Dokumen\Object Detection\CV_model\bestSmallS.torchscript")
    
    # Video source configuration - can be camera index (0, 1, 2) or RTSP URL
    camera_source: Union[int, str] = 0  # Can be 0 for webcam or "rtsp://..." for RTSP
    frame_width: int = 640
    frame_height: int = 480
    
    # Detection configuration
    confidence_threshold: float = 0.5
    class_color_map: Dict[int, Tuple[int, int, int]] = None
    
    # Alert configuration
    #alert_sound_path: Path = Path(r"C:\Farid\Dokumen\Object Detection\usedAudio\notification-alert-269289.mp3")
    alert_cooldown: int = 5
    class_cooldowns: Dict[int, int] = None
    
    # Counter configuration
    counter_time_window: int = 10  # Time window in seconds for the cumulative counter
    
    # RTSP configuration
    reconnect_delay: int = 5  # Seconds between reconnection attempts
    max_consecutive_failures: int = 3  # Maximum failures before attempting reconnection
    
    def __post_init__(self):
        # Initialize default values for mutable objects
        if self.class_color_map is None:
            self.class_color_map = {0: (0, 0, 255), 1: (0, 255, 0)}
        
        if self.class_cooldowns is None:
            self.class_cooldowns = {0: 10, 1: 5, 2: 15}

class AlertManager:
    """Manages audio alerts with cooldown functionality"""
    def __init__(self, config: AppConfig):
        self.config = config
        self.alert_timers = {}
        pygame.mixer.init()
        
        try:
            self.alert_sound = pygame.mixer.Sound(str(config.alert_sound_path))
        except Exception as e:
            print(f"Audio loading error: {e}")
            self.alert_sound = None
    
    def should_trigger_alert(self, cls_id: int) -> bool:
        """Check if an alert should be triggered based on cooldown"""
        current_time = time.time()
        cooldown = self.config.class_cooldowns.get(cls_id, self.config.alert_cooldown)
        
        if cls_id not in self.alert_timers or (current_time - self.alert_timers[cls_id]) >= cooldown:
            self.alert_timers[cls_id] = current_time
            return True
        return False
    
    def play_alert(self):
        """Play the alert sound in a non-blocking thread"""
        if self.alert_sound:
            threading.Thread(target=self.alert_sound.play, daemon=True).start()

class TimeWindowCounter:
    """Manages a time-based counter with a rolling window"""
    def __init__(self, time_window_seconds: int = 10):
        self.time_window = time_window_seconds
        self.detection_history = deque()  # Stores (timestamp, class_name) tuples
        self.current_counts = Counter()
    
    def add_detection(self, class_name: str):
        """Add a detection to the history and update counts"""
        current_time = time.time()
        self.detection_history.append((current_time, class_name))
        self.current_counts[class_name] += 1
        
        # Remove old detections outside the time window
        self._prune_old_detections(current_time)
    
    def _prune_old_detections(self, current_time: float):
        """Remove detections older than the time window"""
        while self.detection_history and current_time - self.detection_history[0][0] > self.time_window:
            old_time, old_class = self.detection_history.popleft()
            self.current_counts[old_class] -= 1
            
            # Remove zero counts to keep the counter clean
            if self.current_counts[old_class] <= 0:
                del self.current_counts[old_class]
    
    def get_counts(self) -> Counter:
        """Get current counts within the time window"""
        current_time = time.time()
        self._prune_old_detections(current_time)
        return self.current_counts.copy()
    
    def get_time_remaining(self) -> float:
        """Get time until the next counter update (oldest detection expires)"""
        if not self.detection_history:
            return self.time_window
        
        current_time = time.time()
        oldest_time = self.detection_history[0][0]
        time_elapsed = current_time - oldest_time
        return max(0, self.time_window - time_elapsed)

class DetectionVisualizer:
    """Handles visualization of detection results"""
    def __init__(self, config: AppConfig):
        self.config = config
        self.line_height = 22
        self.padding = 8
    
    def draw_detection(self, frame, box, class_name, conf, cls_id):
        """Draw a single detection bounding box and label"""
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = self.config.class_color_map.get(cls_id, (255, 255, 255))
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Create label
        label = f"{class_name} {conf:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        
        # Draw label background
        cv2.rectangle(
            frame, 
            (x1, y1 - text_height - 10), 
            (x1 + text_width, y1), 
            color, 
            -1
        )
        
        # Draw label text
        cv2.putText(
            frame, 
            label, 
            (x1, y1 - 5), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (255, 255, 255), 
            2
        )
        
        return class_name
    
    def draw_text_panel(self, frame, lines, x0=10, y0=30, panel_w=250, right_align=False):
        """Draw a semi-transparent text panel with information"""
        panel_h = self.line_height * len(lines) + self.padding
        
        # Adjust x position if right-aligned
        if right_align:
            x0 = frame.shape[1] - panel_w - x0
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(
            overlay, 
            (x0 - self.padding//2, y0 - self.line_height),
            (x0 + panel_w, y0 - self.line_height + panel_h), 
            (0, 0, 0), 
            -1
        )
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Draw text lines
        y = y0
        for line in lines:
            cv2.putText(
                frame, line, (x0, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1
            )
            y += self.line_height
    
    def draw_cooldown_status(self, frame, alert_manager):
        """Display cooldown status for each class in top-left"""
        current_time = time.time()
        for i, (cls_id, last_alert) in enumerate(alert_manager.alert_timers.items()):
            time_remaining = max(0, self.config.alert_cooldown - (current_time - last_alert))
            status_text = f"Class {cls_id} alert: {time_remaining:.1f}s cooldown"
            cv2.putText(
                frame, status_text, (10, 30 + i*30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
            )
    
    def draw_connection_status(self, frame, is_connected, failure_count):
        """Display connection status"""
        status_text = "Connected" if is_connected else f"Disconnected ({failure_count})"
        color = (0, 255, 0) if is_connected else (0, 0, 255)
        cv2.putText(
            frame, f"Status: {status_text}", (10, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )
    
    def draw_error_message(self, frame, message):
        """Display error message on the frame"""
        # Create a semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Add error text
        text_lines = message.split('\n')
        y_pos = frame.shape[0] // 2 - (len(text_lines) * 30) // 2
        
        for line in text_lines:
            text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            x_pos = (frame.shape[1] - text_size[0]) // 2
            cv2.putText(frame, line, (x_pos, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            y_pos += 40

class ObjectDetector:
    """Main class for object detection workflow"""
    def __init__(self, config: AppConfig):
        self.config = config
        self.model = YOLO(str(config.model_path), task='detect')
        self.cap = None
        self.alert_manager = AlertManager(config)
        self.visualizer = DetectionVisualizer(config)
        self.time_window_counter = TimeWindowCounter(config.counter_time_window)
        self.consecutive_failures = 0
        
        # Initialize video source
        self.setup_video_source()
    
    def setup_video_source(self):
        """Initialize and configure the video source (camera or RTSP)"""
        if self.cap is not None:
            self.cap.release()
            
        try:
            # Convert camera index to int if it's a numeric string
            if isinstance(self.config.camera_source, str) and self.config.camera_source.isdigit():
                camera_source = int(self.config.camera_source)
            else:
                camera_source = self.config.camera_source
                
            self.cap = cv2.VideoCapture(camera_source)
            
            # Set RTSP parameters for better stability if using RTSP
            if isinstance(camera_source, str) and "rtsp" in camera_source.lower():
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Set resolution if specified
            if self.config.frame_width > 0:
                self.cap.set(3, self.config.frame_width)
            if self.config.frame_height > 0:
                self.cap.set(4, self.config.frame_height)
            
            # Test if we can read a frame
            if not self.cap.isOpened():
                return False
                
            # Try to read a frame to verify the camera works
            ret, frame = self.cap.read()
            if not ret:
                return False
                
            return True
            
        except Exception as e:
            print(f"Error setting up video source: {e}")
            return False
    
    def process_frame(self, frame):
        """Process a single frame for object detection"""
        frame_counts = Counter()
        results = self.model(frame, stream=True)
        
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                
                if conf > self.config.confidence_threshold:
                    # Check for alert
                    if cls_id in self.config.class_cooldowns and self.alert_manager.should_trigger_alert(cls_id):
                        self.alert_manager.play_alert()
                    
                    # Visualize detection
                    class_name = self.model.names[cls_id]
                    self.visualizer.draw_detection(frame, box, class_name, conf, cls_id)
                    frame_counts[class_name] += 1
                    
                    # Add to time window counter
                    self.time_window_counter.add_detection(class_name)
        
        # Get time window counts
        time_window_counts = self.time_window_counter.get_counts()
        time_remaining = self.time_window_counter.get_time_remaining()
        
        # Prepare and display counts panel in top-right corner
        lines = ["Counts (frame):"]
        if frame_counts:
            for name, cnt in frame_counts.items():
                lines.append(f"{name}: {cnt}")
        else:
            lines.append("None")
            
        lines.append("")
        lines.append(f"Last {self.config.counter_time_window}s:")
        if time_window_counts:
            for name, cnt in time_window_counts.items():
                lines.append(f"{name}: {cnt}")
        else:
            lines.append("None")
            
        lines.append(f"Reset in: {time_remaining:.1f}s")
            
        # Draw counter panel in top-right corner
        self.visualizer.draw_text_panel(frame, lines, 10, 30, 250, right_align=True)
        
        # Display cooldown status in top-left corner
        self.visualizer.draw_cooldown_status(frame, self.alert_manager)
        
        # Display connection status
        is_connected = self.cap.isOpened() and self.consecutive_failures < self.config.max_consecutive_failures
        self.visualizer.draw_connection_status(frame, is_connected, self.consecutive_failures)

        # Resize the frame 
        # frame = cv2.resize(frame, (1024, 576)) # Change this to resize the window
        
        return frame
    
    def run(self):
        """Main application loop with reconnection capability"""
        try:
            # Create a blank frame for error display
            blank_frame = None
            
            while True:
                # Check connection status and reconnect if needed
                if not self.cap.isOpened() or self.consecutive_failures >= self.config.max_consecutive_failures:
                    print(f"Attempting to reconnect to video source... (Failures: {self.consecutive_failures})")
                    try:
                        if self.setup_video_source():
                            self.consecutive_failures = 0
                            print("Reconnection successful")
                        else:
                            print(f"Reconnection failed")
                            time.sleep(self.config.reconnect_delay)
                            continue
                    except Exception as e:
                        print(f"Reconnection error: {e}")
                        time.sleep(self.config.reconnect_delay)
                        continue
                
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    self.consecutive_failures += 1
                    print(f"Failed to read frame ({self.consecutive_failures}/{self.config.max_consecutive_failures})")
                    
                    # Create a blank frame for display if needed
                    if blank_frame is None:
                        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    
                    # Show error message
                    error_msg = f"Cannot read from video source\nFailures: {self.consecutive_failures}/{self.config.max_consecutive_failures}"
                    self.visualizer.draw_error_message(blank_frame, error_msg)
                    cv2.imshow('YOLOv8 Live Detection', blank_frame)
                    
                    time.sleep(1)
                    continue
                
                # Reset failure counter on successful frame read
                self.consecutive_failures = 0
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Display the resulting frame
                cv2.imshow('YOLOv8 Live Detection', processed_frame)
                
                # Exit on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            if self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()

def main():
    """Main function to initialize and run the application"""
    # Configuration - change only these values to modify application behavior
    
    # For webcam (default)
    camera_source = 0
    
    # For RTSP stream (uncomment and modify as needed)
    #camera_source = "rtsp://admin:CemaraMas2025!@192.168.2.190:554/Streaming/Channels/101"
    
    # Try different camera indices if 0 doesn't work
    for camera_idx in [0, 1, 2]:
        try:
            test_cap = cv2.VideoCapture(camera_idx)
            if test_cap.isOpened():
                print(f"Found camera at index {camera_idx}")
                test_cap.release()
                camera_source = camera_idx
                break
        except:
            pass
    
    config = AppConfig(
        model_path=Path(r"D:\RaihanFarid\Dokumen\Object Detection\CV_model\bestSmallS.torchscript"),
        camera_source=camera_source,  # Use 0 for webcam or RTSP URL for stream
        frame_width=640,
        frame_height=480,
        confidence_threshold=0.5,
        class_color_map={0: (0, 0, 255), 1: (0, 255, 0)},  # Red, Green
        #alert_sound_path=Path(r"C:\Farid\Dokumen\Object Detection\usedAudio\notification-alert-269289.mp3"),
        alert_cooldown=5,
        class_cooldowns={0: 10, 1: 5, 2: 15},
        counter_time_window=10,  # 10-second time window for counter
        reconnect_delay=5,  # Seconds between reconnection attempts
        max_consecutive_failures=3  # Maximum failures before attempting reconnection
    )
    
    # Create and run the detector
    detector = ObjectDetector(config)
    detector.run()

if __name__ == "__main__":
    # Add numpy import at the top if not already present
    import numpy as np
    main()