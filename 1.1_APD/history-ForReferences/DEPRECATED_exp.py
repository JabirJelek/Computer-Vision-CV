# Python 2/3 compatibility



"""
This file deprecated because it consist of too many lines of code, while 
the same functionality can be achieved by using threading in a more simpler way.

Therefore, this only used if references needed further.


"""




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
                conf_threshold=0.5, alert_classes_path=None, log_file="detection_log.txt"):
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
        self.alert_cooldown_duration = 5  # Seconds between alerts for same class
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
        
    def _initialize_alert_classes(self, alert_classes_path):
        """Initialize alert classes with individual cooldown times"""
        alert_classes = {}
        class_cooldowns = {}  # New: Store per-class cooldowns
        
        if alert_classes_path and os.path.exists(alert_classes_path):
            try:
                with open(alert_classes_path, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line and not line.startswith('#'):
                            parts = line.split(':')
                            if len(parts) >= 2:
                                class_id = int(parts[0].strip())
                                class_name = parts[1].strip()
                                
                                # Get cooldown time (default to 5 seconds if not specified)
                                cooldown = int(parts[2].strip()) if len(parts) >= 3 and parts[2].strip() else 5
                                
                                alert_classes[class_id] = class_name
                                class_cooldowns[class_id] = cooldown
                                
                                print(f"  - Class {class_id}: {class_name} (cooldown: {cooldown}s)")
                
                # Store class_cooldowns as instance variable
                self.class_cooldowns = class_cooldowns
                print(f"Loaded {len(alert_classes)} alert classes with individual cooldowns")
                
            except Exception as e:
                print(f"Error loading alert classes: {e}")
                # Fallback to default cooldown
                self.class_cooldowns = {}
        else:
            print("No alert classes configuration file provided. Alerts disabled.")
            self.class_cooldowns = {}
        
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
        """Check detections against alert classes using individual cooldowns"""
        current_time = time.time()
        newly_triggered = set()
        
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
                
                # Log ALL detections
                self._log_detection(class_id, class_name, confidence, frame_num, is_alert=False)
                
                # Check if this class is in our alert classes
                if class_id in self.alert_classes:
                    # Get cooldown for this specific class (default to 5 seconds)
                    class_cooldown = self.class_cooldowns.get(class_id, 5)
                    
                    # Check cooldown for this class
                    last_alert = self.alert_cooldown.get(class_id, 0)
                    if current_time - last_alert >= class_cooldown:
                        # Trigger alert
                        alert_class_name = self.alert_classes[class_id]
                        newly_triggered.add(class_id)
                        self.alert_cooldown[class_id] = current_time
                        
                        # Log the alert
                        self._log_detection(class_id, alert_class_name, confidence, frame_num, is_alert=True)
                        
                        # Print alert message with cooldown info
                        alert_msg = f"🎬 VIDEO ALERT: {alert_class_name} detected (Confidence: {confidence:.2f}, Cooldown: {class_cooldown}s)"
                        print(alert_msg)
            
            # Update active alerts
            self.active_alerts = newly_triggered           

    def set_class_cooldown(self, class_id, cooldown_seconds):
        """Dynamically change cooldown for a specific class"""
        if class_id in self.alert_classes:
            self.class_cooldowns[class_id] = max(1, cooldown_seconds)
            print(f"Cooldown for class {class_id} ({self.alert_classes[class_id]}) changed to {cooldown_seconds} seconds")
        else:
            print(f"Warning: Class ID {class_id} not found in alert classes")

    def set_global_cooldown(self, cooldown_seconds):
        """Set default cooldown for all classes"""
        default_cooldown = max(1, cooldown_seconds)
        for class_id in self.alert_classes:
            self.class_cooldowns[class_id] = default_cooldown
        print(f"Global cooldown set to {cooldown_seconds} seconds for all classes")

    def get_cooldown_info(self):
        """Get current cooldown configuration"""
        cooldown_info = {}
        for class_id, class_name in self.alert_classes.items():
            cooldown = self.class_cooldowns.get(class_id, 5)
            cooldown_info[class_id] = {
                'class_name': class_name,
                'cooldown_seconds': cooldown,
                'last_alert': self.alert_cooldown.get(class_id, 'Never')
            }
        return cooldown_info

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
        """Stop both threads and release resources"""
        self.running = False
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
            
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
            
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
    
    def _run_yolo_detection(self, frame, frame_num):
        """Run YOLO object detection on a single frame"""
        try:
            # Run YOLO inference:cite[1]
            results = self.model.predict(
                source=frame,
                conf=self.conf_threshold,
                verbose=False  # Set to True for detailed inference info
            )
            
            # Check for alerts
            self._check_alerts(results, frame_num)
            
            # Process results
            if results and len(results) > 0:
                # Annotate frame with detections:cite[1]
                annotated_frame = results[0].plot()
                detections_count = len(results[0].boxes) if results[0].boxes else 0
                self.detection_count += detections_count
                return annotated_frame, detections_count
            
            return frame, 0
            
        except Exception as e:
            print(f"YOLO inference error: {e}")
            return frame, 0
                
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
        cv.putText(frame, f"Time: {timestamp}", (10, 125), 
                  cv.FONT_HERSHEY_SIMPLEX, font_scale-0.1, (255, 255, 255), 1)
        cv.putText(frame, f"Interval: {self.processing_interval}s", (10, 140), 
                  cv.FONT_HERSHEY_SIMPLEX, font_scale-0.1, (255, 255, 255), 1)
        cv.putText(frame, "Press ESC to exit", (10, 155), 
                  cv.FONT_HERSHEY_SIMPLEX, font_scale-0.1, (255, 255, 255), 1)
    
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



class VideoFileProcessor:
    """
    Single-threaded video file processor with YOLO object detection and alert system
    Uses sequential processing to avoid threading complexities
    """
    
    def __init__(self, video_path, processing_interval=0.5, display_width=640,
                 model_path="path/to/your/model.pt", conf_threshold=0.5, 
                 alert_classes_path=None, log_file="video_detection_log.txt"):
        """
        Args:
            video_path: Path to video file (.mp4, .avi, .mov, .mkv, .wmv)
            processing_interval: Time in seconds between processing frames
            display_width: Width for resizing display frame
            model_path: Path to YOLO model weights
            conf_threshold: Confidence threshold for YOLO detections
            alert_classes_path: Path to alert classes configuration file
            log_file: Log file for detections
        """
        self.video_path = video_path
        self.processing_interval = processing_interval
        self.display_width = display_width
        self.conf_threshold = conf_threshold
        self.log_file = log_file
        
        # Initialize components (reusing your existing methods)
        self._initialize_logging()
        self.model = self._initialize_model(model_path)
        self.alert_classes = self._initialize_alert_classes(alert_classes_path)
        
        # Alert system state
        self.alert_cooldown = {}
        self.alert_cooldown_duration = 5
        self.active_alerts = set()  # Tracks currently active alert classes
        self.detection_count = 0
        
        # Video properties (will be set when video is opened)
        self.frame_width = 0
        self.frame_height = 0
        self.display_height = 0
        self.video_fps = 0
        self.total_frames = 0
        
    def _initialize_logging(self):
        """Initialize logging system (same as your existing method)"""
        try:
            if not os.path.exists(self.log_file):
                with open(self.log_file, 'w') as f:
                    f.write("Timestamp,Frame_Number,Class_ID,Class_Name,Confidence,Alert_Triggered\n")
            print(f"Video logging enabled: {self.log_file}")
        except Exception as e:
            print(f"Video log initialization warning: {e}")

    def _log_detection(self, class_id, class_name, confidence, frame_num, is_alert=False):
        """Log detection event to file (same as your existing method)"""
        try:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            alert_flag = "YES" if is_alert else "NO"
            log_entry = f"{timestamp},{frame_num},{class_id},{class_name},{confidence:.3f},{alert_flag}\n"
            
            with open(self.log_file, 'a') as f:
                f.write(log_entry)
        except Exception as e:
            print(f"Video logging error: {e}")

    def _initialize_model(self, model_path):
        """Initialize YOLO model (adapted from your existing method)"""
        try:
            placeholder_path = Path(model_path)
            
            if not placeholder_path.exists():
                print(f"Warning: Model path '{model_path}' does not exist.")
                print("Using pretrained YOLO11n model as placeholder.")
                model = YOLO("yolo11n.pt")
            else:
                model = YOLO(model_path)
            
            print(f"YOLO model loaded for video processing: {model.__class__.__name__}")
            return model
            
        except Exception as e:
            print(f"Error loading YOLO model for video: {e}")
            print("Falling back to pretrained YOLO11n model")
            return YOLO("yolo11n.pt")

    def _initialize_alert_classes(self, alert_classes_path):
        """Initialize alert classes (same as your existing method)"""
        alert_classes = {}
        
        if alert_classes_path and os.path.exists(alert_classes_path):
            try:
                with open(alert_classes_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            parts = line.split(':')
                            if len(parts) == 2:
                                class_id = int(parts[0].strip())
                                class_name = parts[1].strip()
                                alert_classes[class_id] = class_name
                
                print(f"Loaded {len(alert_classes)} alert classes for video processing")
                for class_id, class_name in alert_classes.items():
                    print(f"  - Class {class_id}: {class_name}")
                    
            except Exception as e:
                print(f"Error loading alert classes for video: {e}")
                print("Continuing without alert classes")
        else:
            print("No alert classes configuration file provided for video. Alerts disabled.")
        
        return alert_classes

    def _check_alerts(self, results, frame_num):
        """Check detections against alert classes and update active alerts"""
        current_time = time.time()
        currently_detected = set()  # Track all alert-class objects found in THIS FRAME
        
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
                
                # Log ALL detections
                self._log_detection(class_id, class_name, confidence, frame_num, is_alert=False)
                
                # Check if this class is in our alert classes
                if class_id in self.alert_classes:
                    # Add to currently detected set for active tracking
                    currently_detected.add(class_id)
                    
                    # Check cooldown for this class for alert triggering
                    last_alert = self.alert_cooldown.get(class_id, 0)
                    if current_time - last_alert >= self.alert_cooldown_duration:
                        # Trigger alert
                        alert_class_name = self.alert_classes[class_id]
                        self.alert_cooldown[class_id] = current_time
                        
                        # Log the alert
                        self._log_detection(class_id, alert_class_name, confidence, frame_num, is_alert=True)
                        
                        # Print alert message
                        alert_msg = f"🎬 VIDEO ALERT: {alert_class_name} detected (Confidence: {confidence:.2f})"
                        print(alert_msg)
        
        # Update active_alerts: shows what's currently detected in this frame
        # This replaces the old set with current detections
        self.active_alerts = currently_detected

    def _run_yolo_detection(self, frame, frame_num):
        """Run YOLO object detection on a single frame (same as your existing method)"""
        try:
            # Run YOLO inference
            results = self.model.predict(
                source=frame,
                conf=self.conf_threshold,
                verbose=False
            )
            
            # Check for alerts
            self._check_alerts(results, frame_num)
            
            # Process results
            if results and len(results) > 0:
                # Annotate frame with detections
                annotated_frame = results[0].plot()
                detections_count = len(results[0].boxes) if results[0].boxes else 0
                self.detection_count += detections_count
                return annotated_frame, detections_count
            
            return frame, 0
            
        except Exception as e:
            print(f"Video YOLO inference error: {e}")
            return frame, 0

    def _add_info_overlay(self, frame, frame_num, processed_count, detections_count):
        """Add informational overlay (adapted from your existing method)"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Scale font size based on display width
        font_scale = 0.5 if self.display_width <= 640 else 0.7
        thickness = 1 if self.display_width <= 640 else 2
        
        # Video-specific overlay
        cv.putText(frame, f"Source: Video File", (10, 25), 
                  cv.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)
        cv.putText(frame, f"Frame: {frame_num}/{self.total_frames}", (10, 45), 
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
        cv.putText(frame, "Press ESC to exit, SPACE to pause", (10, 190), 
                  cv.FONT_HERSHEY_SIMPLEX, font_scale-0.1, (255, 255, 255), 1)

    def _resize_frame(self, frame):
        """Resize frame for display"""
        return cv.resize(frame, (self.display_width, self.display_height))

    def process_video(self):
        """
        Main video processing loop - single-threaded sequential processing
        """
        print(f"Starting video file processing: {self.video_path}")
        
        # Open video file
        cap = cv.VideoCapture(self.video_path)
        if not cap.isOpened():
            error_msg = f"Could not open video file: {self.video_path}"
            raise RuntimeError(error_msg)
        
        # Get video properties
        self.frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.video_fps = cap.get(cv.CAP_PROP_FPS)
        self.total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        
        # Calculate display dimensions
        self.display_height = int((self.display_width / self.frame_width) * self.frame_height)
        
        print(f"Video properties: {self.frame_width}x{self.frame_height} at {self.video_fps:.2f} FPS")
        print(f"Total frames: {self.total_frames}")
        print(f"Display size: {self.display_width}x{self.display_height}")
        print(f"Processing interval: {self.processing_interval} seconds")
        
        # Calculate frame skip based on processing interval and video FPS
        frames_to_skip = max(1, int(self.processing_interval * self.video_fps))
        print(f"Processing every {frames_to_skip} frame(s)")
        
        frame_count = 0
        processed_count = 0
        paused = False
        
        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        print("End of video reached")
                        break
                    
                    frame_count += 1
                    
                    # Process frame based on interval
                    if frame_count % frames_to_skip == 0:
                        processed_count += 1
                        
                        # Run YOLO detection
                        processed_frame, detections = self._run_yolo_detection(frame, frame_count)
                        
                        # Resize for display
                        display_frame = self._resize_frame(processed_frame)
                        
                        # Add informational overlay
                        self._add_info_overlay(display_frame, frame_count, processed_count, detections)
                        
                        # Display the frame
                        cv.imshow("Video File Processing - YOLO Detection", display_frame)
                
                # Handle key presses
                key = cv.waitKey(1) & 0xFF
                if key == 27:  # ESC key - exit
                    break
                elif key == 32:  # SPACE key - pause/unpause
                    paused = not paused
                    print("Video paused" if paused else "Video resumed")
                elif key == ord('q'):  # Q key - exit
                    break
        
        except Exception as e:
            print(f"Video processing error: {e}")
        
        finally:
            # Cleanup
            cap.release()
            cv.destroyAllWindows()
            print(f"Video processing completed. Processed {processed_count} frames, Total detections: {self.detection_count}")

def main():
    """
    Demonstration of the SelectiveFrameProcessor with YOLO Object Detection
    Now includes video file processing option
    """
    print("Selective Frame Processing with YOLO Object Detection")
    print("=" * 60)
    print("Features:")
    print("- Camera & RTSP support (dual-threaded)")
    print("- Video file support (single-threaded)")  
    print("- Multi-threaded architecture for live sources") 
    print("- Single-threaded sequential processing for video files")  
    print("- Selective frame sampling for CPU efficiency")
    print("- YOLO object detection integration")
    print("- Class-based alert system")
    print("- Resizable display output")
    print("- Alert class with file input")
    print("- Logging for alerted classes")        
    print("- Real-time performance monitoring")
    print("\nControls:")
    print("  ESC: Exit")
    print("  SPACE: Pause/Resume (video files only)")
    print("=" * 60)
    
    # Choose source type
    while True:
        choice = input("Choose source type:\n1. Camera\n2. RTSP Stream\n3. Video File\nEnter choice (1, 2, or 3): ").strip()
        
        if choice == '1':
            # Your existing camera code...
            camera_index = int(input("Enter camera index (default 0): ") or "0")
            display_width = int(input("Enter display width (default 640): ") or "640")
            processing_interval = float(input("Enter processing interval in seconds (default 0.5): ") or "0.5")
            model_path = input("Enter YOLO model path (or press Enter for pretrained model): ").strip()
            alert_classes_path = input("Enter alert classes config file path (or press Enter to skip): ").strip()
            
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
                alert_classes_path=alert_classes_path if alert_classes_path else None
            )
            
            try:
                processor.start()
                while processor.running:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\nInterrupted by user")
            finally:
                processor.stop()
            break
            
        elif choice == '2':
            # Your existing RTSP code...
            rtsp_url = input("Enter RTSP URL: ").strip()
            if not rtsp_url:
                rtsp_url = "rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4"
                print(f"Using demo URL: {rtsp_url}")
            
            display_width = int(input("Enter display width (default 640): ") or "640")
            processing_interval = float(input("Enter processing interval in seconds (default 1.0): ") or "1.0")
            model_path = input("Enter YOLO model path (or press Enter for pretrained model): ").strip()
            alert_classes_path = input("Enter alert classes config file path (or press Enter to skip): ").strip()
            
            if not model_path:
                model_path = "yolo11n.pt"
                print("Using pretrained YOLO11n model")
            
            processor = SelectiveFrameProcessor(
                source=rtsp_url,
                processing_interval=processing_interval,
                is_rtsp=True,
                display_width=display_width,
                model_path=model_path,
                alert_classes_path=alert_classes_path if alert_classes_path else None
            )
            
            try:
                processor.start()
                while processor.running:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\nInterrupted by user")
            finally:
                processor.stop()
            break
            
        elif choice == '3':
            # Video file processing
            video_path = input("Enter video file path: ").strip()
            if not video_path:
                print("No video path provided. Using default sample video.")
                # You might want to provide a default sample video path
                video_path = "sample_video.mp4"  # Change this to a real sample path
            
            display_width = int(input("Enter display width (default 640): ") or "640")
            processing_interval = float(input("Enter processing interval in seconds (default 0.5): ") or "0.5")
            model_path = input("Enter YOLO model path (or press Enter for pretrained model): ").strip()
            alert_classes_path = input("Enter alert classes config file path (or press Enter to skip): ").strip()
            
            if not model_path:
                model_path = "yolo11n.pt"
                print("Using pretrained YOLO11n model")
            
            # Use the new VideoFileProcessor (single-threaded)
            video_processor = VideoFileProcessor(
                video_path=video_path,
                processing_interval=processing_interval,
                display_width=display_width,
                model_path=model_path,
                conf_threshold=0.5,
                alert_classes_path=alert_classes_path if alert_classes_path else None,
                log_file="video_detection_log.txt"
            )
            
            # Start single-threaded processing
            video_processor.process_video()
            break
            
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == '__main__':
    main()