# Python 2/3 compatibility
from __future__ import print_function

import os
import sys
import time
import threading
import numpy as np
import cv2 as cv
from pathlib import Path
from ultralytics import YOLO

class SelectiveFrameProcessor:
    """
    A two-thread system for efficient frame capture with YOLO object detection:
    - Capture Thread: Continuously captures frames, keeping only the latest
    - Processing Thread: Samples frames at fixed intervals for YOLO inference
    Supports both camera devices and RTSP streams
    """
    
    def __init__(self, source=0, fps=30, processing_interval=0.5, is_rtsp=False, display_width=640, 
                model_path="path/to/your/model.pt", 
                conf_threshold=0.5, alert_classes_path=None, log_file="detection_log.csv",
                bbox_label_config_path=None):
        """
        Args:
            source: Camera device index (int) or RTSP URL (string)
            fps: Target frames per second for camera (ignored for RTSP)
            processing_interval: Time in seconds between processing frames
            is_rtsp: Boolean indicating if source is an RTSP stream
            display_width: Width for resizing display frame (maintains aspect ratio)
            model_path: Path to YOLO model weights (.pt file)
            conf_threshold: Confidence threshold for YOLO detections
            alert_classes_path: Path to alert classes configuration file
        """
        self.source = source
        self.is_rtsp = is_rtsp
        self.processing_interval = processing_interval
        self.display_width = display_width
        self.conf_threshold = conf_threshold

            # Initialize logging system
        self.log_file = log_file
        self._initialize_logging()
        
        # Initialize YOLO model
        self.model_path = model_path
        self.model = self._initialize_model()
        
        # Initialize alert system
        self.alert_classes_path = alert_classes_path
        self.alert_classes = self._initialize_alert_classes()
        self.alert_cooldown = {}  # Track last alert time per class
        self.alert_cooldown_duration = 2  # Seconds between alerts for same class
        self.active_alerts = set()  # Currently triggered alert classes
        
        # Initialize capture based on source type
        if self.is_rtsp:
            print(f"Initializing RTSP stream: {source}")
            self.capture = cv.VideoCapture(source)
            
            # Set RTSP-specific options for better performance
            self.capture.set(cv.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size
            self.capture.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'H264'))
        else:
            print(f"Initializing camera device: {source}")
            self.capture = cv.VideoCapture(source)
            self.capture.set(cv.CAP_PROP_FPS, fps)
        
        if not self.capture.isOpened():
            error_msg = f"Could not open {'RTSP stream' if is_rtsp else 'camera'}: {source}"
            raise RuntimeError(error_msg)
            
        # Get video properties
        self.frame_width = int(self.capture.get(cv.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.capture.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.actual_fps = self.capture.get(cv.CAP_PROP_FPS)
        
        # Calculate display dimensions maintaining aspect ratio
        self.display_height = int((self.display_width / self.frame_width) * self.frame_height)
        
        print(f"Video properties: {self.frame_width}x{self.frame_height} at {self.actual_fps:.2f} FPS")
        print(f"Display size: {self.display_width}x{self.display_height}")
        
        # Thread synchronization
        self.lock = threading.Lock()
        self.latest_frame = None
        self.frame_counter = 0
        self.running = False
        
        # Threads
        self.capture_thread = None
        self.processing_thread = None
        
        # Performance monitoring
        self.capture_failures = 0
        self.max_capture_failures = 10
        self.detection_count = 0

        # Initialize custom bounding box labeling system
        self.bbox_label_config_path = bbox_label_config_path
        self.bbox_label_map = self._initialize_bbox_labeling()
        
    def _initialize_bbox_labeling(self):
        """
        Initialize custom bounding box labeling system
        Returns: dict with class_id -> {label, color, confidence_threshold}
        """
        bbox_label_map = {}
        
        if self.bbox_label_config_path and os.path.exists(self.bbox_label_config_path):
            try:
                with open(self.bbox_label_config_path, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line and not line.startswith('#'):
                            parts = line.split(':')
                            if len(parts) == 4:
                                class_id = int(parts[0].strip())
                                display_label = parts[1].strip()
                                color_str = parts[2].strip()
                                conf_thresh = float(parts[3].strip())
                                
                                # Parse BGR color
                                color_parts = color_str.split(',')
                                if len(color_parts) == 3:
                                    color = tuple(int(c) for c in color_parts)
                                else:
                                    color = (0, 255, 0)  # Default green
                                
                                bbox_label_map[class_id] = {
                                    'label': display_label,
                                    'color': color,
                                    'confidence_threshold': conf_thresh
                                }
                                
                                print(f"Custom bbox: Class {class_id} -> '{display_label}' {color} @ {conf_thresh}")
                
                print(f"Loaded {len(bbox_label_map)} custom bounding box labels")
                
            except Exception as e:
                print(f"Error loading bbox label config: {e}")
                print("Continuing with default bounding box labels")
        else:
            if self.bbox_label_config_path:
                print(f"BBox label config not found: {self.bbox_label_config_path}")
            print("Using default YOLO bounding box labels")
        
        return bbox_label_map        
        
    def _initialize_logging(self):
        """Initialize logging system"""
        try:
            # Create log header if file doesn't exist
            if not os.path.exists(self.log_file):
                with open(self.log_file, 'w') as f:
                    f.write("Timestamp,Frame_Number,Class_ID,Class_Name,Confidence,Alert_Triggered\n")
            print(f"Logging enabled: {self.log_file}")
        except Exception as e:
            print(f"Log initialization warning: {e}")

    def _log_detection(self, class_id, class_name, confidence, frame_num, is_alert=False):
        """Log detection event to file"""
        try:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            alert_flag = "YES" if is_alert else "NO"
            
            log_entry = f"{timestamp},{frame_num},{class_id},{class_name},{confidence:.3f},{alert_flag}\n"
            
            with open(self.log_file, 'a') as f:
                f.write(log_entry)
                
        except Exception as e:
            print(f"Logging error: {e}")  # Silent fail - don't break main functionality
                    
    def _initialize_alert_classes(self):
        """Initialize alert classes from configuration file"""
        alert_classes = {}
        
        if self.alert_classes_path and os.path.exists(self.alert_classes_path):
            try:
                with open(self.alert_classes_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            parts = line.split(':')
                            if len(parts) == 2:
                                class_id = int(parts[0].strip())
                                class_name = parts[1].strip()
                                alert_classes[class_id] = class_name
                
                print(f"Loaded {len(alert_classes)} alert classes from {self.alert_classes_path}")
                for class_id, class_name in alert_classes.items():
                    print(f"  - Class {class_id}: {class_name}")
                    
            except Exception as e:
                print(f"Error loading alert classes: {e}")
                print("Continuing without alert classes")
        else:
            print("No alert classes configuration file provided. Alerts disabled.")
            if self.alert_classes_path:
                print(f"Alert classes path was: {self.alert_classes_path}")
        
        return alert_classes
    
    def _initialize_model(self):
        """Initialize YOLO model with error handling"""
        try:
            # Replace this path with your actual model path
            placeholder_path = Path(self.model_path)
            
            if not placeholder_path.exists():
                print(f"Warning: Model path '{self.model_path}' does not exist.")
                print("Please update the 'model_path' parameter with your actual model path.")
                print("For now, using a pretrained YOLO11n model as placeholder.")
                model = YOLO("yolo11n.pt")  # Fallback to pretrained model:cite[1]
            else:
                model = YOLO(self.model_path)  # Load your custom model:cite[1]
            
            print(f"YOLO model loaded successfully: {model.__class__.__name__}")
            return model
            
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            print("Falling back to pretrained YOLO11n model")
            return YOLO("yolo11n.pt")  # Ultimate fallback:cite[1]

    def _check_alerts(self, results, frame_num):
        current_time = time.time()
        newly_triggered = set()
        
        # Clear expired alerts
        expired_alerts = []
        for class_id, last_alert in self.alert_cooldown.items():
            if current_time - last_alert > self.alert_cooldown_duration:
                expired_alerts.append(class_id)
        
        for class_id in expired_alerts:
            if class_id in self.active_alerts:
                self.active_alerts.remove(class_id)
        
        if results and len(results) > 0 and results[0].boxes:
            for box in results[0].boxes:
                class_id = int(box.cls.item())
                confidence = box.conf.item()
                
                # Get class name from model names
                class_name = "unknown"
                if hasattr(self.model, 'names') and self.model.names:
                    class_name = self.model.names.get(class_id, f"class_{class_id}")
                else:
                    class_name = f"class_{class_id}"
                
                # Log ALL detections (not just alert classes)
                self._log_detection(class_id, class_name, confidence, frame_num, is_alert=False)
                
                # Check if this class is in our alert classes
                if class_id in self.alert_classes:
                    # Check cooldown for this class
                    last_alert = self.alert_cooldown.get(class_id, 0)
                    if current_time - last_alert >= self.alert_cooldown_duration:
                        # Trigger alert
                        alert_class_name = self.alert_classes[class_id]
                        newly_triggered.add(class_id)
                        self.alert_cooldown[class_id] = current_time
                        
                        # Log the alert
                        self._log_detection(class_id, alert_class_name, confidence, frame_num, is_alert=True)
                        
                        # Print alert message
                        alert_msg = f"🚨 ALERT: {alert_class_name} detected (Confidence: {confidence:.2f})"
                        print(alert_msg)
            
            # FIX: Update active alerts without clearing previous ones
            self.active_alerts.update(newly_triggered)
            
            # Clear alerts that are no longer active (after their cooldown)
            still_active = set()
            for class_id in self.active_alerts:
                last_alert = self.alert_cooldown.get(class_id, 0)
                if current_time - last_alert < self.alert_cooldown_duration:
                    still_active.add(class_id)
            self.active_alerts = still_active

    def _run_yolo_detection(self, frame, frame_num):
        """Run YOLO object detection with custom bounding box rendering"""
        try:
            # Run YOLO inference
            results = self.model.predict(
                source=frame,
                conf=self.conf_threshold,
                verbose=False
            )
            
            # Check for alerts
            self._check_alerts(results, frame_num)
            
            # Process results with custom bounding boxes
            if results and len(results) > 0:
                # Use custom bounding box rendering instead of default plot()
                annotated_frame = self._draw_custom_bounding_boxes(frame, results)
                
                detections_count = len(results[0].boxes) if results[0].boxes else 0
                self.detection_count += detections_count
                return annotated_frame, detections_count
            
            return frame, 0
            
        except Exception as e:
            print(f"YOLO inference error: {e}")
            return frame, 0
  
    def _add_info_overlay(self, frame, frame_num, processed_count, detections_count):
        """Add informational text overlay to the frame with YOLO-specific info"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        source_type = "RTSP" if self.is_rtsp else "Camera"
        
        # Scale font size based on display width
        font_scale = 0.5 if self.display_width <= 640 else 0.7
        thickness = 1 if self.display_width <= 640 else 2
        
        # Add different colored text for better visibility
        cv.putText(frame, f"Source: {source_type}", (10, 25), 
                  cv.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)
        cv.putText(frame, f"Frame: {frame_num}", (10, 45), 
                  cv.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
        cv.putText(frame, f"Processed: {processed_count}", (10, 65), 
                  cv.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 0), thickness)
        cv.putText(frame, f"Detections: {detections_count}", (10, 85), 
                  cv.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 255), thickness)
        cv.putText(frame, f"Total Detections: {self.detection_count}", (10, 105), 
                  cv.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 255), thickness)
        
        # Add alert status
        alert_status = f"Active Alerts: {len(self.active_alerts)}"
        alert_color = (0, 0, 255) if self.active_alerts else (0, 255, 0)
        cv.putText(frame, alert_status, (10, 125), 
                  cv.FONT_HERSHEY_SIMPLEX, font_scale, alert_color, thickness)
        
        # Add alert classes if any are active
        if self.active_alerts:
            alert_text = "Alerts: " + ", ".join([self.alert_classes[class_id] for class_id in self.active_alerts])
            cv.putText(frame, alert_text, (10, 145), 
                      cv.FONT_HERSHEY_SIMPLEX, font_scale-0.1, (0, 0, 255), 1)
        
        cv.putText(frame, f"Time: {timestamp}", (10, 160), 
                  cv.FONT_HERSHEY_SIMPLEX, font_scale-0.1, (255, 255, 255), 1)
        cv.putText(frame, f"Interval: {self.processing_interval}s", (10, 175), 
                  cv.FONT_HERSHEY_SIMPLEX, font_scale-0.1, (255, 255, 255), 1)
        cv.putText(frame, "Press ESC to exit", (10, 190), 
                  cv.FONT_HERSHEY_SIMPLEX, font_scale-0.1, (255, 255, 255), 1)
    
    def start(self):
        """Start both capture and processing threads"""
        self.running = True
        
        # Start capture thread
        self.capture_thread = threading.Thread(target=self._capture_loop, name="CaptureThread")
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        # Start processing thread  
        self.processing_thread = threading.Thread(target=self._processing_loop, name="ProcessingThread")
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        source_type = "RTSP stream" if self.is_rtsp else "camera"
        print(f"Started SelectiveFrameProcessor with YOLO:")
        print(f"  - Source: {source_type} ({self.source})")
        print(f"  - Capture: Continuous")
        print(f"  - YOLO Processing: Every {self.processing_interval} seconds")
        print(f"  - Display size: {self.display_width}x{self.display_height}")
        print(f"  - Confidence threshold: {self.conf_threshold}")
        
    def stop(self):
        """Enhanced cleanup"""
        self.running = False
        
        # Proper thread termination
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
        
        # Force cleanup if threads hang
        if self.capture_thread and self.capture_thread.is_alive():
            print("Warning: Capture thread did not terminate cleanly")
        if self.processing_thread and self.processing_thread.is_alive():
            print("Warning: Processing thread did not terminate cleanly")
        
        self.capture.release()
        cv.destroyAllWindows()
        print(f"Stopped SelectiveFrameProcessor. Total detections: {self.detection_count}")
        
    def _capture_loop(self):
        """Continuously capture frames, keeping only the latest"""
        source_type = "RTSP" if self.is_rtsp else "camera"
        print(f"Capture thread started - continuously capturing frames from {source_type}")
        frames_captured = 0
        self.capture_failures = 0
        
        while self.running:
            ret, frame = self.capture.read()
            if not ret:
                self.capture_failures += 1
                print(f"Warning: Failed to capture frame from {source_type} (failure #{self.capture_failures})")
                
                # For RTSP, try to reconnect after multiple failures
                if self.is_rtsp and self.capture_failures >= self.max_capture_failures:
                    print("Multiple RTSP capture failures - attempting reconnection...")
                    self._reconnect_rtsp()
                    self.capture_failures = 0
                else:
                    time.sleep(0.1)
                continue
                
            # Reset failure counter on successful capture
            self.capture_failures = 0
            frames_captured += 1
            
            # Store only the latest frame
            with self.lock:
                self.latest_frame = frame.copy() if frame is not None else None
                self.frame_counter = frames_captured
                
        print(f"Capture thread stopped. Total frames captured: {frames_captured}")
        
    def _reconnect_rtsp(self):
        """Attempt to reconnect to RTSP stream"""
        print("Attempting RTSP reconnection...")
        self.capture.release()
        time.sleep(2)
        
        self.capture = cv.VideoCapture(self.source)
        self.capture.set(cv.CAP_PROP_BUFFERSIZE, 1)
        self.capture.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'H264'))
        
        if self.capture.isOpened():
            print("RTSP reconnection successful")
        else:
            print("RTSP reconnection failed")
                 
    def _processing_loop(self):
        """Run YOLO detection on frames at fixed time intervals"""
        print("YOLO processing thread started - sampling frames at fixed intervals")
        frames_processed = 0
        last_processing_time = time.time()
        
        while self.running:
            current_time = time.time()
            elapsed = current_time - last_processing_time
            
            if elapsed >= self.processing_interval:
                frame_to_process = None
                frame_num = 0
                
                with self.lock:
                    if self.latest_frame is not None:
                        frame_to_process = self.latest_frame.copy()
                        frame_num = self.frame_counter
                        self.latest_frame = None  # Clear after taking to prevent stale frames
                
                if frame_to_process is not None:
                    frames_processed += 1
                    
                    # Run YOLO object detection - pass both frame and frame_num
                    processed_frame, detections = self._run_yolo_detection(frame_to_process, frame_num)
                    
                    # Rest of your existing code remains the same...
                    resized_frame = self._resize_frame(processed_frame)
                    
                    # Add informational overlay with detection info
                    self._add_info_overlay(resized_frame, frame_num, frames_processed, detections)
                    
                    # Display the resized frame with detections
                    cv.imshow("YOLO Object Detection - Selective Processing", resized_frame)
                    
                    # Handle key presses - only ESC for exit
                    key = cv.waitKey(1) & 0xFF
                    if key == 27:  # ESC key
                        self.running = False
                        break
                
                last_processing_time = current_time
                
            time.sleep(0.001)          
    
    def _resize_frame(self, frame):
        """Resize frame to display dimensions maintaining aspect ratio"""
        return cv.resize(frame, (self.display_width, self.display_height))
     
    def set_processing_interval(self, interval):
        """Dynamically change the processing interval"""
        self.processing_interval = max(0.01, interval)
        print(f"YOLO processing interval changed to {self.processing_interval} seconds")
    
    def set_display_size(self, width):
        """Dynamically change the display size"""
        self.display_width = max(160, width)  # Minimum 160px width
        self.display_height = int((self.display_width / self.frame_width) * self.frame_height)
        print(f"Display size changed to {self.display_width}x{self.display_height}")
    
    def set_confidence_threshold(self, confidence):
        """Dynamically change YOLO confidence threshold"""
        self.conf_threshold = max(0.01, min(1.0, confidence))
        print(f"YOLO confidence threshold changed to {self.conf_threshold}")
    
    def get_video_properties(self):
        """Get current video stream properties"""
        return {
            'width': self.frame_width,
            'height': self.frame_height,
            'fps': self.actual_fps,
            'source_type': 'RTSP' if self.is_rtsp else 'Camera',
            'display_size': f"{self.display_width}x{self.display_height}",
            'model': str(self.model_path),
            'confidence_threshold': self.conf_threshold,
            'total_detections': self.detection_count
        }

    def set_alert_cooldown(self, cooldown_seconds):
        """Dynamically change alert cooldown duration"""
        self.alert_cooldown_duration = max(1, cooldown_seconds)
        print(f"Alert cooldown changed to {self.alert_cooldown_duration} seconds")
    
    def reload_alert_classes(self, new_alert_classes_path=None):
        """Reload alert classes from configuration file"""
        if new_alert_classes_path:
            self.alert_classes_path = new_alert_classes_path
        
        self.alert_classes = self._initialize_alert_classes()
        self.alert_cooldown.clear()
        self.active_alerts.clear()
        print("Alert classes reloaded")

    def _draw_custom_bounding_boxes(self, frame, results):
        """
        Draw custom bounding boxes with personalized labels and colors
        NOW ONLY DISPLAYS CLASSES THAT ARE IN ALERT_CLASSES.TXT
        Returns: annotated frame
        """
        if not results or len(results) == 0 or not results[0].boxes:
            return frame
        
        annotated_frame = frame.copy()
        boxes = results[0].boxes
        
        for box in boxes:
            class_id = int(box.cls.item())
            confidence = box.conf.item()
            
            # ONLY display if this class is in alert_classes
            if class_id not in self.alert_classes:
                continue  # Skip classes not in alert_classes.txt
            
            # Get original coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Determine label, color, and whether to display
            display_label, box_color, should_display = self._get_bbox_display_properties(class_id, confidence)
            
            if should_display:
                # Draw bounding box
                thickness = 2
                cv.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, thickness)
                
                # Draw label background
                label = f"{display_label} {confidence:.2f}"
                label_size = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                
                # Label background rectangle
                cv.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                             (x1 + label_size[0], y1), box_color, -1)
                
                # Label text
                cv.putText(annotated_frame, label, (x1, y1 - 5), 
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Additional visual indicators for high-confidence detections
                if confidence > 0.8:
                    # Draw corner markers
                    marker_length = 15
                    # Top-left corner
                    cv.line(annotated_frame, (x1, y1), (x1 + marker_length, y1), box_color, 3)
                    cv.line(annotated_frame, (x1, y1), (x1, y1 + marker_length), box_color, 3)
                    # Top-right corner  
                    cv.line(annotated_frame, (x2, y1), (x2 - marker_length, y1), box_color, 3)
                    cv.line(annotated_frame, (x2, y1), (x2, y1 + marker_length), box_color, 3)
                    # Bottom-left corner
                    cv.line(annotated_frame, (x1, y2), (x1 + marker_length, y2), box_color, 3)
                    cv.line(annotated_frame, (x1, y2), (x1, y2 - marker_length), box_color, 3)
                    # Bottom-right corner
                    cv.line(annotated_frame, (x2, y2), (x2 - marker_length, y2), box_color, 3)
                    cv.line(annotated_frame, (x2, y2), (x2, y2 - marker_length), box_color, 3)
        
        return annotated_frame
    
    def _get_bbox_display_properties(self, class_id, confidence):
        """
        Determine display properties for a bounding box based on custom configuration
        NOW ONLY PROCESSES CLASSES THAT ARE IN ALERT_CLASSES.TXT
        Returns: (label, color, should_display)
        """
        # Only display if class is in alert_classes
        if class_id not in self.alert_classes:
            return "", (0, 0, 0), False
        
        # Check if we have custom configuration for this class
        if class_id in self.bbox_label_map:
            config = self.bbox_label_map[class_id]
            # Check confidence threshold
            if confidence >= config['confidence_threshold']:
                return config['label'], config['color'], True
            else:
                return "", (0, 0, 0), False  # Don't display if below threshold
        
        # Default behavior - use alert class name and default color
        alert_class_name = self.alert_classes.get(class_id, f"class_{class_id}")
        # Use color based on class_id for variety
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), 
                 (255, 0, 255), (0, 255, 255), (128, 0, 128)]
        color = colors[class_id % len(colors)]
        return alert_class_name, color, True

    def reload_bbox_labels(self, new_config_path=None):
        """Reload custom bounding box label configuration"""
        if new_config_path:
            self.bbox_label_config_path = new_config_path
        
        self.bbox_label_map = self._initialize_bbox_labeling()
        print("Bounding box labels reloaded")
    
    def add_custom_bbox_label(self, class_id, display_label, color=(0, 255, 0), confidence_threshold=0.5):
        """Dynamically add a custom bounding box label"""
        self.bbox_label_map[class_id] = {
            'label': display_label,
            'color': color,
            'confidence_threshold': confidence_threshold
        }
        print(f"Added custom bbox: Class {class_id} -> '{display_label}'")
    
    def remove_custom_bbox_label(self, class_id):
        """Remove a custom bounding box label"""
        if class_id in self.bbox_label_map:
            del self.bbox_label_map[class_id]
            print(f"Removed custom bbox label for class {class_id}")

    def _draw_styled_bounding_box(self, frame, x1, y1, x2, y2, label, color, confidence):
        """Draw bounding box with advanced styling options"""
        thickness = 3
        alpha = 0.6  # Transparency for fill
        
        # Create overlay for filled rectangle
        overlay = frame.copy()
        
        # Filled rectangle with transparency
        cv.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Outer border
        cv.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Inner border
        inner_thickness = 1
        cv.rectangle(frame, (x1 + thickness, y1 + thickness), 
                     (x2 - thickness, y2 - thickness), (255, 255, 255), inner_thickness)
        
        # Label with background
        label_text = f"{label} {confidence:.2f}"
        font_scale = 0.6
        font_thickness = 2
        
        label_size = cv.getTextSize(label_text, cv.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        label_y = y1 - 10 if y1 - 10 > 20 else y1 + 20
        
        # Label background
        cv.rectangle(frame, (x1, label_y - label_size[1] - 10),
                     (x1 + label_size[0] + 10, label_y + 5), color, -1)
        
        # Label text
        cv.putText(frame, label_text, (x1 + 5, label_y), 
                   cv.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
        
        return frame

def main():
    """
    Demonstration of the SelectiveFrameProcessor with YOLO Object Detection
    """
    print("Selective Frame Processing with YOLO Object Detection")
    print("=" * 60)
    print("Features:")
    print("- Camera & RTSP support")
    print("- Multi-threaded architecture") 
    print("- Selective frame sampling for CPU efficiency")
    print("- YOLO object detection integration")
    print("- Class-based alert system")
    print("- Resizable display output")
    print("- Alert class with file input")
    print("- Logging for alerted classes")        
    print("- Real-time performance monitoring")
    print("\nControls:")
    print("  ESC: Exit")
    print("=" * 60)
    
    # Choose source type
    while True:
        choice = input("Choose source type:\n1. Camera\n2. RTSP Stream\nEnter choice (1 or 2): ").strip()
        
        if choice == '1':
            camera_index = int(input("Enter camera index (default 0): ") or "0")
            display_width = int(input("Enter display width (default 640): ") or "640")
            processing_interval = float(input("Enter processing interval in seconds (default 0.5): ") or "0.5")
            model_path = input("Enter YOLO model path (or press Enter for pretrained model): ").strip()
            alert_classes_path = input("Enter alert classes config file path (or press Enter to skip): ").strip()
            bbox_label_config_path = input("Enter custom bbox label config file path (or press Enter to skip): ").strip()

            if not bbox_label_config_path:
                bbox_label_config_path = None
                print("Using default bounding box labels")
            else:
                print(f"Using custom bbox labels from: {bbox_label_config_path}")

            
            if not model_path:
                model_path = "yolo11n.pt"
                print("Using pretrained YOLO11n model")
            
            processor = SelectiveFrameProcessor(
                source=camera_index,
                fps=30,
                processing_interval=processing_interval,
                is_rtsp=False,
                display_width=display_width,
                model_path=model_path,
                alert_classes_path=alert_classes_path if alert_classes_path else None,
                bbox_label_config_path=bbox_label_config_path        
            )
            break
        elif choice == '2':
            rtsp_url = input("Enter RTSP URL: ").strip()
            if not rtsp_url:
                rtsp_url = "rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4"
                print(f"Using demo URL: {rtsp_url}")
            
            display_width = int(input("Enter display width (default 640): ") or "640")
            processing_interval = float(input("Enter processing interval in seconds (default 1.0): ") or "1.0")
            model_path = input("Enter YOLO model path (or press Enter for pretrained model): ").strip()
            alert_classes_path = input("Enter alert classes config file path (or press Enter to skip): ").strip()
            bbox_label_config_path = input("Enter custom bbox label config file path (or press Enter to skip): ").strip()

            if not bbox_label_config_path:
                bbox_label_config_path = None
                print("Using default bounding box labels")
            else:
                print(f"Using custom bbox labels from: {bbox_label_config_path}")            
            
            if not model_path:
                model_path = "yolo11n.pt"
                print("Using pretrained YOLO11n model")
            
            processor = SelectiveFrameProcessor(
                source=rtsp_url,
                processing_interval=processing_interval,
                is_rtsp=True,
                display_width=display_width,
                model_path=model_path,
                alert_classes_path=alert_classes_path if alert_classes_path else None,
                bbox_label_config_path=bbox_label_config_path        
            )
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")
    
    try:
        processor.start()
        
        # Keep main thread alive while threads run
        while processor.running:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        processor.stop()


if __name__ == '__main__':
    main()