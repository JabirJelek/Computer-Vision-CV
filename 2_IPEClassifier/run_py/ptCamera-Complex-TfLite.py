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
    model_path: Path = Path(r"C:\Farid\Dokumen\Object Detection\CV_model\task1.torchscript")
    
    # Video source configuration - can be camera index (0, 1, 2) or RTSP URL
    camera_source: Union[int, str] = 0  # Can be 0 for webcam or "rtsp://..." for RTSP
    frame_width: int = 640
    frame_height: int = 480
    
    # Detection configuration
    confidence_threshold: float = 0.5
    class_color_map: Dict[int, Tuple[int, int, int]] = None
    
    # Alert configuration
    alert_sound_path: Path = Path(r"C:\Farid\Dokumen\Object Detection\usedAudio\notification-alert-269289.mp3")
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

import numpy as np
import tensorflow as tf
from typing import List

class TFLiteObjectDetector:
    """Object detector for TFLite models"""
    def __init__(self, config: AppConfig):
        self.config = config
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.input_shape = None
        self.class_names = {0: "class_0", 1: "class_1"}  # Update with your class names
        
        # Initialize video capture attribute to None
        self.cap = None
        
        # Initialize video source and TFLite model
        self.setup_video_source()
        self.load_tflite_model()
        
        # Initialize other components
        self.alert_manager = AlertManager(config)
        self.visualizer = DetectionVisualizer(config)
        self.time_window_counter = TimeWindowCounter(config.counter_time_window)
        self.consecutive_failures = 0
    
    def load_tflite_model(self):
        """Load TFLite model and allocate tensors"""
        try:
            # Load the TFLite model
            self.interpreter = tf.lite.Interpreter(model_path=str(self.config.model_path))
            self.interpreter.allocate_tensors()
            
            # Get input and output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            # Get input shape
            self.input_shape = self.input_details[0]['shape']
            print(f"TFLite model loaded successfully. Input shape: {self.input_shape}")
            
        except Exception as e:
            print(f"Error loading TFLite model: {e}")
            raise
    
    def preprocess_frame(self, frame):
        """Preprocess frame for TFLite model input"""
        # Resize frame to match model input size
        input_size = (self.input_shape[1], self.input_shape[2])  # (height, width)
        resized_frame = cv2.resize(frame, input_size)
        
        # Normalize if needed (adjust based on your model requirements)
        # Most TFLite models expect normalized input [0, 1] or [-1, 1]
        normalized_frame = resized_frame.astype(np.float32) / 255.0
        
        # Add batch dimension
        input_data = np.expand_dims(normalized_frame, axis=0)
        
        return input_data
    
    def postprocess_output(self, output_data, original_frame):
        """Postprocess TFLite model output to extract detections"""
        detections = []
        frame_height, frame_width = original_frame.shape[:2]
        
        # Adjust this based on your specific TFLite model's output format
        # Common formats:
        # 1. [boxes, classes, scores, num_detections] for SSD models
        # 2. Single tensor with format [batch, num_detections, 5 + num_classes] for YOLO models
        
        # Example for SSD-style output (4 outputs)
        if len(output_data) == 4:
            boxes = output_data[0][0]  # [y_min, x_min, y_max, x_max] normalized
            classes = output_data[1][0].astype(np.int32)
            scores = output_data[2][0]
            num_detections = int(output_data[3][0])
            
            for i in range(num_detections):
                if scores[i] > self.config.confidence_threshold:
                    y_min, x_min, y_max, x_max = boxes[i]
                    
                    # Convert normalized coordinates to pixel coordinates
                    x1 = int(x_min * frame_width)
                    y1 = int(y_min * frame_height)
                    x2 = int(x_max * frame_width)
                    y2 = int(y_max * frame_height)
                    
                    detections.append({
                        'box': [x1, y1, x2, y2],
                        'class_id': classes[i],
                        'score': scores[i],
                        'class_name': self.class_names.get(classes[i], f"class_{classes[i]}")
                    })
        
        # Example for YOLO-style single output
        elif len(output_data) == 1:
            # Assuming output format: [batch, num_detections, 5 + num_classes]
            output = output_data[0][0]  # Remove batch dimension
            
            for detection in output:
                # Extract bounding box and confidence
                x_center, y_center, width, height, confidence = detection[:5]
                
                if confidence > self.config.confidence_threshold:
                    # Extract class probabilities
                    class_probs = detection[5:]
                    class_id = np.argmax(class_probs)
                    class_confidence = class_probs[class_id]
                    
                    # Calculate box coordinates
                    x1 = int((x_center - width / 2) * frame_width)
                    y1 = int((y_center - height / 2) * frame_height)
                    x2 = int((x_center + width / 2) * frame_width)
                    y2 = int((y_center + height / 2) * frame_height)
                    
                    # Ensure coordinates are within frame boundaries
                    x1 = max(0, min(x1, frame_width))
                    y1 = max(0, min(y1, frame_height))
                    x2 = max(0, min(x2, frame_width))
                    y2 = max(0, min(y2, frame_height))
                    
                    detections.append({
                        'box': [x1, y1, x2, y2],
                        'class_id': class_id,
                        'score': confidence * class_confidence,
                        'class_name': self.class_names.get(class_id, f"class_{class_id}")
                    })
        
        return detections
    
    def process_frame(self, frame):
        """Process a single frame for object detection using TFLite"""
        frame_counts = Counter()
        
        # Preprocess frame
        input_data = self.preprocess_frame(frame)
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output tensors
        output_data = []
        for output_detail in self.output_details:
            output_data.append(self.interpreter.get_tensor(output_detail['index']))
        
        # Postprocess outputs
        detections = self.postprocess_output(output_data, frame)
        
        # Process each detection
        for detection in detections:
            # Check for alert
            if detection['class_id'] in self.config.class_cooldowns and \
               self.alert_manager.should_trigger_alert(detection['class_id']):
                self.alert_manager.play_alert()
            
            # Visualize detection
            class_name = detection['class_name']
            # Create a mock box object with xyxy attribute for compatibility
            class MockBox:
                def __init__(self, coords):
                    self.xyxy = [coords]
            
            box = MockBox(detection['box'])
            self.visualizer.draw_detection(frame, box, class_name, detection['score'], detection['class_id'])
            frame_counts[class_name] += 1
            
            # Add to time window counter
            self.time_window_counter.add_detection(class_name)
        
        # The rest of your visualization code remains the same
        time_window_counts = self.time_window_counter.get_counts()
        time_remaining = self.time_window_counter.get_time_remaining()
        
        # Prepare and display counts panel
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
            
        # Draw counter panel
        self.visualizer.draw_text_panel(frame, lines, 10, 30, 250, right_align=True)
        
        # Display cooldown status
        self.visualizer.draw_cooldown_status(frame, self.alert_manager)
        
        # Display connection status
        is_connected = self.cap.isOpened() and self.consecutive_failures < self.config.max_consecutive_failures
        self.visualizer.draw_connection_status(frame, is_connected, self.consecutive_failures)
        
        return frame
    
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
    """Process a single frame for object detection using TFLite"""
    frame_counts = Counter()
    
    # Preprocess frame for TFLite model
    input_data = self.preprocess_frame(frame)
    
    # Set input tensor
    self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
    
    # Run inference
    self.interpreter.invoke()
    
    # Get output tensors - adjust this based on your model's output format
    output_data = []
    for i, output_detail in enumerate(self.output_details):
        output_data.append(self.interpreter.get_tensor(output_detail['index']))
    
    # Process detections based on your model's output format
    detections = self.postprocess_output(output_data, frame)
    
    # Process each detection
    for detection in detections:
        cls_id = detection['class_id']
        conf = detection['score']
        class_name = detection['class_name']
        box = detection['box']
        
        if conf > self.config.confidence_threshold:
            # Check for alert
            if cls_id in self.config.class_cooldowns and self.alert_manager.should_trigger_alert(cls_id):
                self.alert_manager.play_alert()
            
            # Visualize detection - create a mock box object for compatibility
            class MockBox:
                def __init__(self, coords):
                    self.xyxy = [coords]
            
            mock_box = MockBox(box)
            self.visualizer.draw_detection(frame, mock_box, class_name, conf, cls_id)
            frame_counts[class_name] += 1
            
            # Add to time window counter
            self.time_window_counter.add_detection(class_name)
    
    # The rest of your visualization code remains the same
    time_window_counts = self.time_window_counter.get_counts()
    time_remaining = self.time_window_counter.get_time_remaining()
    
    # Prepare and display counts panel
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
        
    # Draw counter panel
    self.visualizer.draw_text_panel(frame, lines, 10, 30, 250, right_align=True)
    
    # Display cooldown status
    self.visualizer.draw_cooldown_status(frame, self.alert_manager)
    
    # Display connection status
    is_connected = self.cap.isOpened() and self.consecutive_failures < self.config.max_consecutive_failures
    self.visualizer.draw_connection_status(frame, is_connected, self.consecutive_failures)
    
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
    # Configuration
    camera_source = 0  # Use 0 for webcam
    
    # Update class names based on your model
    class_names = {0: "person", 1: "glasses", 2:"phone"}  # Example - replace with your classes
    
    config = AppConfig(
        model_path=Path(r"C:\Farid\Dokumen\Object Detection\CV_model\best_float32.tflite"),  # Update path to your TFLite model
        camera_source=camera_source,
        frame_width=640,
        frame_height=480,
        confidence_threshold=0.5,
        class_color_map={0: (0, 0, 255), 1: (0, 255, 0)},  # Red, Green
        alert_sound_path=Path(r"C:\Farid\Dokumen\Object Detection\usedAudio\notification-alert-269289.mp3"),
        alert_cooldown=5,
        class_cooldowns={0: 10, 1: 5},
        counter_time_window=10,
        reconnect_delay=5,
        max_consecutive_failures=3
    )
    
    # Create and run the detector
    detector = TFLiteObjectDetector(config)
    detector.run()

if __name__ == "__main__":
    main()