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
import queue
import requests
import urllib.parse

class ObjectTracker:  # Was: class PersonTracker
    """
    Tracks individual objects across frames and manages state transitions
    """
    def __init__(self, track_id, initial_bbox, initial_class, frame_num):
        self.track_id = track_id
        self.bbox = initial_bbox
        self.current_class = initial_class
        self.state = "UNKNOWN"
        self.frame_history = []
        self.last_seen_frame = frame_num
        self.creation_frame = frame_num
        
        # State transition counters
        self.consecutive_no_mask_frames = 0
        self.consecutive_mask_frames = 0
        
        # State transition thresholds
        self.no_mask_confirmation_threshold = 20
        self.mask_compliance_threshold = 30
        
        print(f"🎯 Created Object Tracker {track_id} with initial state: {self.state}")  # Updated emoji

    def update(self, bbox, class_name, frame_num):
        """Update tracker with new detection"""
        self.bbox = bbox
        self.current_class = class_name
        self.last_seen_frame = frame_num
        
        # Add to frame history (limit to last 100 frames for memory efficiency)
        self.frame_history.append({
            'frame': frame_num,
            'class': class_name,
            'timestamp': time.time()
        })
        
        # Keep only recent history
        if len(self.frame_history) > 100:
            self.frame_history.pop(0)
        
        return self._check_state_transition()

    def _check_state_transition(self):
        """
        Check and update state based on frame history
        Returns: (state_changed, new_state, old_state)
        """
        old_state = self.state
        
        if len(self.frame_history) < 10:  # Need minimum frames for reliable state
            self.state = "INITIALIZING"
            return False, self.state, old_state
        
        # Analyze recent frames for state transitions
        recent_frames = self.frame_history[-max(30, len(self.frame_history)):]  # Last 30 frames or all available
        
        # Count mask vs no_mask in recent frames
        mask_frames = 0
        no_mask_frames = 0
        
        for frame in recent_frames:
            if "without_mask" in frame['class']:
                no_mask_frames += 1
            elif "with_mask" in frame['class']:
                mask_frames += 1
        
        total_recent_frames = len(recent_frames)
        
        # State transition logic
        if self.state in ["UNKNOWN", "INITIALIZING", "COMPLIANT"]:
            # Check if we should transition to CONFIRMED_NO_MASK
            if no_mask_frames >= self.no_mask_confirmation_threshold:
                if self.state != "CONFIRMED_NO_MASK":
                    self.state = "CONFIRMED_NO_MASK"
                    return True, self.state, old_state
        
        elif self.state == "CONFIRMED_NO_MASK":
            # Check if person has become compliant
            if mask_frames >= self.mask_compliance_threshold:
                self.state = "COMPLIANT"
                return True, self.state, old_state
        
        return False, self.state, old_state

    def get_state_summary(self):
        """Get summary of current state and statistics"""
        if not self.frame_history:
            return "No history"
        
        recent_frames = self.frame_history[-30:]  # Last 30 frames
        mask_count = sum(1 for f in recent_frames if "with_mask" in f['class'])
        no_mask_count = sum(1 for f in recent_frames if "without_mask" in f['class'])
        total = len(recent_frames)
        
        return f"State: {self.state} | Mask: {mask_count}/{total} | No Mask: {no_mask_count}/{total}"

    def is_stale(self, current_frame_num, stale_threshold=50):
        """Check if tracker hasn't been updated for too many frames"""
        return (current_frame_num - self.last_seen_frame) > stale_threshold

class AudioAlertManager:
    """
    Manages audio alerts in a separate thread to avoid blocking main processing.
    """
    def __init__(self, target_url):
        self.target_url = target_url
        self.alert_queue = queue.Queue()
        self.running = False
        self.thread = None
        
        # Existing gap control
        self.last_alert_time = 0
        self.min_alert_gap = 60.0
        
        #   Alert fatigue prevention
        self.message_rotations = self._initialize_message_rotations()
        self.last_message_index = {}
        
        self.start()
    
    def _initialize_message_rotations(self):
        """Define message variations to prevent fatigue"""
        return {
            "person_with_helmet_forklift": [
                "Forklift Operator tanpa masker terdeteksi",
                "Perhatian: Operator forklift tidak pakai masker",
                "Safety alert: Masker tidak dipakai operator forklift"
            ],
            "person_with_mask_forklift": [
                "Forklift Operator tanpa helm terdeteksi",
                "Perhatian: Operator forklift lupa helm safety",
                "Safety alert: Helm safety tidak dipakai operator"
            ],
            "person_without_mask_helmet_forklift": [
                "Forklift Operator tanpa masker dan helm terdeteksi",
                "Safety alert: Operator forklift tidak pakai masker dan helm",
                "Peringatan: Masker dan helm tidak digunakan operator"
            ],
            "person_without_mask_nonForklift": [
                "Masker nya bisa di benerin?",
                "Benerin dulu masker nya, keselamatan itu penting",
                "Bisa di pakai masker nya?"
            ]
        }
    
    def start(self):
        """Start the audio alert processing thread."""
        self.running = True
        self.thread = threading.Thread(target=self._process_alerts, name="AudioAlertThread")
        self.thread.daemon = True  # Daemon thread will exit when main thread exits
        self.thread.start()
        print("Audio Alert Manager started.")
    
    def trigger_alert(self, class_name, confidence):
        """Add alert to queue with smart deduplication"""
        current_time = time.time()
        
        # Enhanced duplicate prevention with timing
        if not self._is_duplicate_alert(class_name, current_time):
            alert_data = {
                'class_name': class_name,
                'confidence': confidence,
                'timestamp': current_time
            }
            
            try:
                self.alert_queue.put(alert_data, block=False)
                print(f"🎵 Audio alert queued: {class_name} (Confidence: {confidence:.2f})")
            except queue.Full:
                print("⚠️ Alert queue full, dropping alert.")

    def _is_duplicate_alert(self, class_name, current_time):
        """Enhanced duplicate detection with time window"""
        # Remove stale entries from queue consideration
        temp_queue = queue.Queue()
        is_duplicate = False
        
        while not self.alert_queue.empty():
            try:
                item = self.alert_queue.get_nowait()
                # Keep items that are recent enough
                if current_time - item['timestamp'] < 60.0:  # 60-second window
                    if item['class_name'] == class_name:
                        is_duplicate = True
                    temp_queue.put(item)
            except queue.Empty:
                break
        
        # Restore the queue
        while not temp_queue.empty():
            self.alert_queue.put(temp_queue.get_nowait())
        
        return is_duplicate
    
    def _process_alerts(self):
        """Process alerts from the queue with timing control"""
        while self.running:
            try:
                #   Add gap control before processing next alert
                current_time = time.time()
                time_since_last_alert = current_time - self.last_alert_time
                
                if time_since_last_alert < self.min_alert_gap:
                    # Wait remaining time before next alert
                    wait_time = self.min_alert_gap - time_since_last_alert
                    time.sleep(wait_time)
                
                alert_data = self.alert_queue.get(timeout=1.0)
                
                self._send_audio_alert(alert_data)
                self.alert_queue.task_done()
                
                #   Update last alert time after sending
                self.last_alert_time = time.time()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Error processing audio alert: {e}")
    
    def _send_audio_alert(self, alert_data):
        """Send HTTP GET request with rotated messages to prevent fatigue"""
        try:
            class_name = alert_data['class_name']
            confidence = alert_data['confidence']
            
            # Get rotated message
            message = self._get_rotated_message(class_name)
            
            # URL encode the message
            encoded_message = urllib.parse.quote(message)
            
            # Build and send request (existing code)
            url_with_params = f"{self.target_url}?pesan={encoded_message}"
            response = requests.get(url_with_params, timeout=10)  # Increased timeout
            
            if response.status_code == 200:
                try:
                    response_json = response.json()
                    if response_json.get('hasil') == 1:
                        print(f"✅ Audio alert sent for {class_name}")
                        print(f"🔊 Custom Message: '{message}'")
                    else:
                        print(f"❌ Server reported failure for {class_name}")
                except ValueError:
                    print(f"⚠️ Server returned non-JSON response: {response.text}")
            else:
                print(f"❌ Server returned HTTP {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Network error sending audio alert: {e}")
        except Exception as e:
            print(f"❌ Unexpected error sending audio alert: {e}")
   
    def stop(self):
        """Stop the alert processing thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        print("Audio Alert Manager stopped.")

    def _get_rotated_message(self, class_name):
        """Get next message in rotation to prevent fatigue"""
        # Default message if no rotation defined
        default_message = f"Peringatan: {class_name} terdeteksi"
        
        if class_name not in self.message_rotations:
            return default_message
        
        variations = self.message_rotations[class_name]
        
        # Initialize or rotate index
        if class_name not in self.last_message_index:
            self.last_message_index[class_name] = 0
        else:
            self.last_message_index[class_name] = (self.last_message_index[class_name] + 1) % len(variations)
        
        return variations[self.last_message_index[class_name]]


class SelectiveFrameProcessor:
    """
    A two-thread system for efficient frame capture with YOLO object detection:
    - Capture Thread: Continuously captures frames, keeping only the latest
    - Processing Thread: Samples frames at fixed intervals for YOLO inference
    - will display class if exist in alert_classes.txt
    Supports both camera devices and RTSP streams
    """
    
    def __init__(self, source=0, fps=30, processing_interval=0.5, is_rtsp=False, display_width=640, 
                model_path="path/to/your/model.pt", 
                conf_threshold=0.5, alert_classes_path=None, log_file="detection_log.csv",
                bbox_label_config_path=None, audio_alert_url=None, persistence_frames=5):
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
            persistence_frames: Number of consecutive frames required before alerting
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
        
        #   Detection persistence system
        self.persistence_enabled = False  # Start disabled by default
        self.persistence_frames = persistence_frames
        self.detection_history = {}  # class_id -> detection history
        
        # Enable persistence if frames > 1
        if persistence_frames > 1:
            self.enable_detection_persistence(persistence_frames)
        
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
        
        # FIXED: Audio alert system initialization
        self.audio_alert_url = audio_alert_url
        self.audio_alert_manager = None
        
        if self.audio_alert_url:
            self.audio_alert_manager = AudioAlertManager(self.audio_alert_url)
            # Optional: Configure gap after creating instance
            self.audio_alert_manager.min_alert_gap = 5.0  # Now this works!
            print(f"Audio alerts enabled for URL: {audio_alert_url}")
        
        # Object tracking system        
        self.object_trackers = {}  # Was: self.person_trackers
        self.next_track_id = 1
        self.tracking_enabled = True
        
        print("🎯 Object state tracking system initialized")  

    #   DETECTION PERSISTENCE METHODS
    def enable_detection_persistence(self, persistence_frames=None):
        """
        Enable detection persistence - require N consecutive frames before alerts
        This reduces false positives by requiring consistent detections
        """
        if persistence_frames:
            self.persistence_frames = max(1, persistence_frames)
        
        self.persistence_enabled = True
        self.detection_history = {}  # class_id -> detection history
        print(f"🔍 Detection persistence enabled: {self.persistence_frames} frames required")

    def disable_detection_persistence(self):
        """Disable detection persistence"""
        self.persistence_enabled = False
        self.detection_history.clear()
        print("🔍 Detection persistence disabled")

    def _check_persistence(self, class_id, frame_num):
        """
        Check if a detection has persisted for required frames
        FIXED: Uses time-based tracking instead of frame-based to handle processing intervals
        Returns: (should_alert, current_count)
        """
        if not self.persistence_enabled:
            return True, 1  # Always alert if persistence disabled
        
        current_time = time.time()
        
        # Initialize or update detection history
        if class_id not in self.detection_history:
            self.detection_history[class_id] = {
                'count': 1,
                'first_detection': current_time,
                'last_detection': current_time,
                'last_processed_time': current_time,
                'consecutive_frames': 1
            }
            return False, 1
        
        history = self.detection_history[class_id]
        
        # FIXED: Use time-based gap detection instead of frame-based
        time_since_last_detection = current_time - history['last_processed_time']
        max_time_gap = self.processing_interval * 3  # Allow reasonable time gap
        
        if time_since_last_detection > max_time_gap:
            # Reset if too much time has passed (object disappeared)
            history['count'] = 1
            history['consecutive_frames'] = 1
            history['first_detection'] = current_time
            history['last_processed_time'] = current_time
        else:
            # Increment count - object is persistently detected
            history['count'] += 1
            history['consecutive_frames'] += 1
            history['last_processed_time'] = current_time
        
        history['last_detection'] = current_time
        
        # Check if we've reached persistence threshold
        if history['consecutive_frames'] >= self.persistence_frames:
            return True, history['consecutive_frames']
        
        return False, history['consecutive_frames']


    def _cleanup_detection_history(self, current_time, max_age_seconds=10):
        """Remove old entries from detection history using time-based cleanup"""
        if not self.persistence_enabled:
            return
        
        stale_classes = []
        for class_id, history in self.detection_history.items():
            if current_time - history['last_detection'] > max_age_seconds:
                stale_classes.append(class_id)
        
        for class_id in stale_classes:
            del self.detection_history[class_id]

    def set_persistence_frames(self, frames):
        """Dynamically change persistence requirement"""
        self.persistence_frames = max(1, frames)
        if self.persistence_enabled:
            self.detection_history.clear()  # Reset history
        print(f"🔍 Persistence frames changed to {self.persistence_frames}")

    def get_persistence_status(self):
        """Get current persistence status and counts"""
        status = {
            'enabled': self.persistence_enabled,
            'required_frames': self.persistence_frames,
            'tracked_classes': len(self.detection_history),
            'current_counts': {}
        }
        
        for class_id, history in self.detection_history.items():
            class_name = self.alert_classes.get(class_id, f"class_{class_id}")
            status['current_counts'][class_name] = {
                'current': history['consecutive_frames'],
                'required': self.persistence_frames,
                'progress': f"{history['consecutive_frames']}/{self.persistence_frames}",
                'time_since_last': time.time() - history['last_detection']
            }
        
        return status
        
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
                model = YOLO("yolo11n.pt")  # Fallback to pretrained model
            else:
                model = YOLO(self.model_path)  # Load your custom model
            
            print(f"YOLO model loaded successfully: {model.__class__.__name__}")
            return model
            
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            print("Falling back to pretrained YOLO11n model")
            return YOLO("yolo11n.pt")  # Ultimate fallback

    def _track_objects(self, results, frame_num):  
        """
        Main object tracking method - matches detections with existing trackers
        and manages state transitions
        """
        if not self.tracking_enabled or not results or not results[0].boxes:
            return
        
        current_detections = []
        
        # Extract current frame detections
        for box in results[0].boxes:
            class_id = int(box.cls.item())
            confidence = box.conf.item()
            
            # Only track alert classes
            if class_id not in self.alert_classes:
                continue
                
            class_name = self.alert_classes[class_id]
            bbox = box.xyxy[0].tolist()
            
            current_detections.append({
                'bbox': bbox,
                'class_name': class_name,
                'class_id': class_id,
                'confidence': confidence
            })
        
        # Match with existing trackers
        matched_trackers = set()
        
        for detection in current_detections:
            tracker_id = self._find_best_tracker_match(detection['bbox'])
            
            if tracker_id is not None:
                # Update existing tracker
                tracker = self.object_trackers[tracker_id]  # Updated variable
                state_changed, new_state, old_state = tracker.update(
                    detection['bbox'], detection['class_name'], frame_num
                )
                
                if state_changed:
                    self._handle_state_transition(tracker, old_state, new_state, frame_num)
                
                matched_trackers.add(tracker_id)
            else:
                # Create new tracker
                self._create_new_tracker(detection, frame_num)
        
        # Remove stale trackers
        self._cleanup_stale_trackers(frame_num)
        
        # Debug: Print current tracking status
        if self.object_trackers and frame_num % 100 == 0:  # Updated variable
            print(f"🎯 Tracking {len(self.object_trackers)} objects:")  # Updated message
            for tracker_id, tracker in list(self.object_trackers.items())[:3]:  # Updated variable
                print(f"   ID{tracker_id}: {tracker.get_state_summary()}")
                
                
    def _find_best_tracker_match(self, detection_bbox, iou_threshold=0.3):
        best_tracker_id = None
        best_iou = iou_threshold
        
        for tracker_id, tracker in self.object_trackers.items():  # Updated variable
            iou = self._calculate_iou(tracker.bbox, detection_bbox)
            if iou > best_iou:
                best_iou = iou
                best_tracker_id = tracker_id
        
        return best_tracker_id
    
    def _calculate_iou(self, bbox1, bbox2):
        """
        Calculate Intersection over Union between two bounding boxes
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection area
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # Calculate union area
        bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = bbox1_area + bbox2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0

    def _create_new_tracker(self, detection, frame_num):
        """Create a new object tracker"""
        tracker_id = self.next_track_id
        self.next_track_id += 1
        
        tracker = ObjectTracker(  # Updated class name
            tracker_id, 
            detection['bbox'], 
            detection['class_name'], 
            frame_num
        )
        
        self.object_trackers[tracker_id] = tracker  # Updated variable
        print(f"🎯 New Object tracked: ID{tracker_id} - {detection['class_name']}")  # Updated emoji

    def _cleanup_stale_trackers(self, current_frame_num, stale_threshold=50):
        """Remove trackers that haven't been updated recently"""
        stale_ids = []
        
        for tracker_id, tracker in self.object_trackers.items():  # Updated variable
            if tracker.is_stale(current_frame_num, stale_threshold):
                stale_ids.append(tracker_id)
        
        for tracker_id in stale_ids:
            print(f"🧹 Removing stale tracker: ID{tracker_id}")
            del self.object_trackers[tracker_id]  # Updated variable
            
    def _handle_state_transition(self, tracker, old_state, new_state, frame_num):
        """
        Handle state transitions and trigger appropriate actions
        """
        print(f"🔄 STATE TRANSITION: Person ID{tracker.track_id} {old_state} → {new_state}")
        
        # Log the state transition
        self._log_state_transition(tracker.track_id, old_state, new_state, frame_num)
        
        # Trigger audio alerts based on state transitions
        if new_state == "CONFIRMED_NO_MASK":
            self._trigger_state_alert(tracker, "no_mask_confirmed")
        elif new_state == "COMPLIANT" and old_state == "CONFIRMED_NO_MASK":
            self._trigger_state_alert(tracker, "mask_compliant")

    def _trigger_state_alert(self, tracker, alert_type):
        """Trigger audio alerts for state transitions"""
        if not self.audio_alert_manager:
            return
            
        alert_messages = {
            "no_mask_confirmed": f"Peringatan: Orang ID{tracker.track_id} tidak pakai masker terkonfirmasi",
            "mask_compliant": f"Bagus: Orang ID{tracker.track_id} sekarang pakai masker"
        }
        
        # For now, we'll use a generic class name for state alerts
        # We can enhance this later with proper state-based alert classes
        self.audio_alert_manager.trigger_alert("state_transition", 0.9)
        
        print(f"🔊 State alert: {alert_messages[alert_type]}")

    def _log_state_transition(self, tracker_id, old_state, new_state, frame_num):
        """Log state transitions to file"""
        try:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"{timestamp},{frame_num},{tracker_id},{old_state},{new_state}\n"
            
            # Use a separate log file for state transitions
            state_log_file = "state_transitions_log.csv"
            if not os.path.exists(state_log_file):
                with open(state_log_file, 'w') as f:
                    f.write("Timestamp,Frame_Number,Tracker_ID,Old_State,New_State\n")
            
            with open(state_log_file, 'a') as f:
                f.write(log_entry)
                
        except Exception as e:
            print(f"State logging error: {e}")

    def _get_tracker_info_for_bbox(self, bbox, iou_threshold=0.3):
        """Get tracker information for a bounding box"""
        best_tracker = None
        best_iou = iou_threshold
        
        for tracker_id, tracker in self.object_trackers.items():  # Updated variable
            iou = self._calculate_iou(tracker.bbox, bbox)
            if iou > best_iou:
                best_iou = iou
                best_tracker = {
                    'id': tracker_id,
                    'state': tracker.state,
                    'summary': tracker.get_state_summary()
                }
        
        return best_tracker

# THE _check_alerts METHOD TO USE TIME-BASED CLEANUP:
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
            # Run object tracking before individual frame alerts
            self._track_objects(results, frame_num)  # Updated method name
            
            # FIXED: Use time-based cleanup instead of frame-based
            self._cleanup_detection_history(current_time)
            
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
                    alert_class_name = self.alert_classes[class_id]
                    
                    # FIXED: Pass current_time to persistence check
                    should_alert, persistence_count = self._check_persistence(class_id, frame_num)
                    
                    if should_alert:
                        # Check cooldown for this class
                        last_alert = self.alert_cooldown.get(class_id, 0)
                        if current_time - last_alert >= self.alert_cooldown_duration:
                            # Trigger alert
                            newly_triggered.add(class_id)
                            self.alert_cooldown[class_id] = current_time
                            
                            # Log the alert with persistence info
                            self._log_detection(class_id, alert_class_name, confidence, frame_num, is_alert=True)
                            
                            print(f"🚨 ALERT: {alert_class_name} (Confidence: {confidence:.2f}, Persistence: {persistence_count}/{self.persistence_frames})")
                        
                        # CONDITIONAL AUDIO ALERT TRIGGER
                        if self.audio_alert_manager and self.audio_alert_url:
                            # Trigger audio alert in separate thread
                            self.audio_alert_manager.trigger_alert(alert_class_name, confidence)
                    else:
                        # Show persistence progress (less frequent to reduce spam)
                        if persistence_count == 1 or persistence_count % 5 == 0:
                            print(f"🔍 Persistence: {alert_class_name} {persistence_count}/{self.persistence_frames} frames")
            
            # Update active alerts without clearing previous ones
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
        
        # Update tracking information text
        tracking_info = f"Objects Tracked: {len(self.object_trackers)}"  # Updated variable
        cv.putText(frame, tracking_info, (10, 200), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Show states of tracked objects (first 3)
        y_offset = 215
        for i, (tracker_id, tracker) in enumerate(list(self.object_trackers.items())[:3]):  # Updated variable
            state_info = f"ID{tracker_id}: {tracker.state}"
            color = (0, 255, 0) if tracker.state == "COMPLIANT" else (0, 0, 255) if tracker.state == "CONFIRMED_NO_MASK" else (255, 255, 255)
            cv.putText(frame, state_info, (10, y_offset), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            y_offset += 15
        
        #   Add persistence information
        if self.persistence_enabled:
            persistence_info = f"Persistence: {self.persistence_frames} frames"
            cv.putText(frame, persistence_info, (10, 260), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            # Show persistence progress for active detections
            y_offset = 275
            for class_id, history in list(self.detection_history.items())[:3]:  # Show first 3
                class_name = self.alert_classes.get(class_id, f"class_{class_id}")
                progress = f"{class_name}: {history['count']}/{self.persistence_frames}"
                color = (0, 255, 0) if history['count'] >= self.persistence_frames else (255, 255, 0)
                cv.putText(frame, progress, (10, y_offset), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                y_offset += 15
        
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
        if self.persistence_enabled:
            print(f"  - Detection Persistence: {self.persistence_frames} frames required")
        
    def stop(self):
        """Enhanced cleanup"""
        self.running = False
        
        # Stop audio alert manager if exists
        if self.audio_alert_manager:
            self.audio_alert_manager.stop()
        
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
                #   Try to find tracker for this detection and add tracking info
                tracker_info = self._get_tracker_info_for_bbox([x1, y1, x2, y2])
                
                if tracker_info:
                    display_label = f"ID{tracker_info['id']} {tracker_info['state']} | {display_label}"
                    # Color code by state
                    if tracker_info['state'] == "CONFIRMED_NO_MASK":
                        box_color = (0, 0, 255)  # Red for no mask
                    elif tracker_info['state'] == "COMPLIANT":
                        box_color = (0, 255, 0)  # Green for compliant
                
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




    def print_timing_debug(self):
        """Debug method to show timing information"""
        if not self.persistence_enabled:
            print("🔍 Persistence disabled")
            return
        
        print("🔍 Persistence Debug Info:")
        print(f"  - Processing Interval: {self.processing_interval}s")
        print(f"  - Required Frames: {self.persistence_frames}")
        print(f"  - Tracked Classes: {len(self.detection_history)}")
        
        for class_id, history in self.detection_history.items():
            class_name = self.alert_classes.get(class_id, f"class_{class_id}")
            time_gap = time.time() - history['last_processed_time']
            max_gap = self.processing_interval * 3
            print(f"  - {class_name}: {history['consecutive_frames']}/{self.persistence_frames} frames, "
                f"Time gap: {time_gap:.2f}s (max: {max_gap:.2f}s)")                   
        
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
    print("- Person state tracking system")
    print("- Audio alerts with fatigue prevention")
    print("- 🔍 DETECTION PERSISTENCE SYSTEM (NEW!)")
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
                
            audio_alert_url = input("Enter audio alert URL (or press Enter to disable): ").strip()
            if not audio_alert_url:
                audio_alert_url = None
                print("Audio alerts disabled")
            else:
                print(f"Audio alerts enabled for: {audio_alert_url}")                

            #   Detection persistence configuration
            persistence_frames = int(input("Enter detection persistence frames (default 5, 1=disabled): ") or "5")
            
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
                bbox_label_config_path=bbox_label_config_path,        
                audio_alert_url=audio_alert_url,
                persistence_frames=persistence_frames  #   Add persistence parameter
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

            audio_alert_url = input("Enter audio alert URL (or press Enter to disable): ").strip()
            if not audio_alert_url:
                audio_alert_url = None
                print("Audio alerts disabled")
            else:
                print(f"Audio alerts enabled for: {audio_alert_url}")                
                            
            #   Detection persistence configuration
            persistence_frames = int(input("Enter detection persistence frames (default 5, 1=disabled): ") or "5")
            
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
                bbox_label_config_path=bbox_label_config_path,     
                audio_alert_url=audio_alert_url,
                persistence_frames=persistence_frames  #   Add persistence parameter
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