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

class ObjectTracker:
    def __init__(self, track_id, initial_bbox, class_name, class_id, frame_num):
        self.track_id = track_id
        self.bbox = initial_bbox
        self.class_name = class_name
        self.class_id = class_id
        self.state = "APPEARED"
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.last_seen_frame = frame_num
        self.consecutive_misses = 0
        self.consecutive_detections = 1
        self.audio_alert_triggered = False
        
        # 🆕 CLASS-SPECIFIC STATE THRESHOLDS
        self.confirmation_threshold, self.disappearance_threshold = self._get_class_thresholds(class_name)
        
        print(f"🎯 Created ObjectTracker {track_id} for {class_name} - State: {self.state} "
              f"(Confirm: {self.confirmation_threshold}, Disappear: {self.disappearance_threshold})")

    def _get_class_thresholds(self, class_name):
        """Efficient class-specific threshold lookup"""
        # Lightweight dictionary - no external dependencies
        class_thresholds = {
            "person_with_helmet_forklift": (2, 3),      # Fast confirmation, quick disappearance
            "person_with_mask_forklift": (3, 5),        # Medium confirmation, standard disappearance  
            "person_without_mask_helmet_forklift": (1, 2), # Immediate confirmation, fast disappearance
            "person_without_mask_nonForklift": (3, 7)   # Slower confirmation, longer disappearance
        }
        return class_thresholds.get(class_name, (3, 5))  # Default fallback

    # 🆕 Add method to update thresholds dynamically
    def update_thresholds(self, confirmation=None, disappearance=None):
        """Update thresholds without recreating tracker"""
        if confirmation is not None:
            self.confirmation_threshold = confirmation
        if disappearance is not None:
            self.disappearance_threshold = disappearance

    def update(self, bbox, frame_num, detected=True):
        """Update tracker state based on detection"""
        self.last_seen_frame = frame_num
        
        if detected:
            self.bbox = bbox
            self.last_seen = time.time()
            self.consecutive_misses = 0
            self.consecutive_detections += 1
            
            # State transitions for detection
            old_state = self.state
            if self.state == "APPEARED" and self.consecutive_detections >= self.confirmation_threshold:
                self.state = "PRESENT"
                return "CONFIRMED", old_state
            elif self.state == "DISAPPEARING":
                self.state = "PRESENT"  # Object reappeared
                return "REAPPEARED", old_state
            elif self.state == "GONE":
                self.state = "APPEARED"  # Object reappeared after being gone
                self.consecutive_detections = 1
                return "REAPPEARED", old_state
                
        else:
            self.consecutive_misses += 1
            self.consecutive_detections = 0
            
            # State transitions for misses
            old_state = self.state
            if self.state == "PRESENT" and self.consecutive_misses >= self.disappearance_threshold:
                self.state = "GONE"
                return "DISAPPEARED", old_state
            elif self.state == "PRESENT" and self.consecutive_misses > 0:
                self.state = "DISAPPEARING"
                return "DISAPPEARING", old_state
            elif self.state == "APPEARED" and self.consecutive_misses > 2:
                self.state = "GONE"  # Never confirmed, just disappeared
                return "DISAPPEARED", old_state
        
        return self.state, self.state

    def should_trigger_audio(self):
        """Determine if audio should play based on object state"""
        # Trigger audio when object first becomes PRESENT and hasn't been alerted yet
        if self.state == "PRESENT" and not self.audio_alert_triggered:
            self.audio_alert_triggered = True
            return True
        return False
    
    def is_stale(self, current_frame_num, threshold=50):
        """Check if tracker should be removed"""
        return (current_frame_num - self.last_seen_frame) > threshold
    
    def get_state_info(self):
        return {
            'track_id': self.track_id,
            'class_name': self.class_name,
            'state': self.state,
            'consecutive_detections': self.consecutive_detections,
            'consecutive_misses': self.consecutive_misses,
            'audio_triggered': self.audio_alert_triggered
        }

class AudioAlertManager:
    """
    Manages audio alerts in a separate thread to avoid blocking main processing.
    """
    def __init__(self, target_url):
        self.target_url = target_url
        self.alert_queue = queue.Queue()
        self.running = False
        self.thread = None
        
        # Alert gap control
        self.last_alert_time = 0
        self.min_alert_gap = 60.0
        
        # Alert fatigue prevention
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
        self.thread.daemon = True
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
                # Add gap control before processing next alert
                current_time = time.time()
                time_since_last_alert = current_time - self.last_alert_time
                
                if time_since_last_alert < self.min_alert_gap:
                    # Wait remaining time before next alert
                    wait_time = self.min_alert_gap - time_since_last_alert
                    time.sleep(wait_time)
                
                alert_data = self.alert_queue.get(timeout=1.0)
                
                self._send_audio_alert(alert_data)
                self.alert_queue.task_done()
                
                # Update last alert time after sending
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
            
            # Build and send request
            url_with_params = f"{self.target_url}?pesan={encoded_message}"
            response = requests.get(url_with_params, timeout=10)
            
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
        self.alert_cooldown = {}
        self.alert_cooldown_duration = 5
        self.active_alerts = set()
        
        # Initialize capture based on source type
        if self.is_rtsp:
            print(f"Initializing RTSP stream: {source}")
            self.capture = cv.VideoCapture(source)
            self.capture.set(cv.CAP_PROP_BUFFERSIZE, 1)
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
        
        # Calculate display dimensions
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
        
        # Audio alert system
        self.audio_alert_url = audio_alert_url
        self.audio_alert_manager = None
        
        if self.audio_alert_url:
            self.audio_alert_manager = AudioAlertManager(self.audio_alert_url)
            self.audio_alert_manager.min_alert_gap = 5.0
            print(f"Audio alerts enabled for URL: {audio_alert_url}")
        
        # STATE SYSTEM: Initialize stateful object tracking
        self.object_trackers = {}  # track_id -> ObjectTracker
        self.next_track_id = 1
        self.tracking_enabled = True

        # Initialize persistence_frames properly
        self.persistence_frames = persistence_frames
        self.persistence_enabled = False  # Start disabled by default
        self.detection_history = {}  # Add this line
            
        # Enable persistence if frames > 1
        if persistence_frames > 1:
            self.enable_detection_persistence(persistence_frames)
        
        # FIX: Remove duplicate tracking systems - use only object_trackers
        self.object_trackers = {}  # track_id -> ObjectTracker
        self.next_track_id = 1
        self.tracking_enabled = True
        
        print("🎯 Stateful object tracking system initialized")

        # Add this line in your __init__ method after all other initializations:
        self._cleanup_duplicate_systems()

        # 🆕 CLASS-SPECIFIC CONFIGURATION
        self.class_thresholds_config = self._initialize_class_thresholds()
        
    def _initialize_class_thresholds(self):
        """Initialize class-specific threshold configuration"""
        # Lightweight in-memory config - no file I/O unless needed
        return {
            "person_with_helmet_forklift": {"confirmation": 2, "disappearance": 3},
            "person_with_mask_forklift": {"confirmation": 3, "disappearance": 5},
            "person_without_mask_helmet_forklift": {"confirmation": 1, "disappearance": 2},
            "person_without_mask_nonForklift": {"confirmation": 3, "disappearance": 7}
        }
    
    # 🆕 Add method to get thresholds for a class
    def get_class_thresholds(self, class_name):
        """Get thresholds for specific class with fallback"""
        default = {"confirmation": 3, "disappearance": 5}
        return self.class_thresholds_config.get(class_name, default)
    
    # 🆕 Add method to update configuration dynamically
    def update_class_thresholds(self, class_name, confirmation=None, disappearance=None):
        """Update thresholds for a specific class"""
        if class_name not in self.class_thresholds_config:
            self.class_thresholds_config[class_name] = {"confirmation": 3, "disappearance": 5}
        
        if confirmation is not None:
            self.class_thresholds_config[class_name]["confirmation"] = confirmation
        if disappearance is not None:
            self.class_thresholds_config[class_name]["disappearance"] = disappearance
            
        # Update existing trackers of this class
        self._update_existing_trackers(class_name)
        
        print(f"✅ Updated thresholds for {class_name}: confirm={confirmation}, disappear={disappearance}")
    
    def _update_existing_trackers(self, class_name):
        """Update thresholds for existing trackers of specified class"""
        updated_count = 0
        for tracker in self.object_trackers.values():
            if tracker.class_name == class_name:
                thresholds = self.get_class_thresholds(class_name)
                tracker.confirmation_threshold = thresholds["confirmation"]
                tracker.disappearance_threshold = thresholds["disappearance"]
                updated_count += 1
        
        if updated_count > 0:
            print(f"🔄 Updated {updated_count} active trackers for {class_name}")        
    
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

    def _initialize_tracking_system(self):
        """Initialize stateful object tracking for all alert classes"""
        self.object_trackers = {}
        self.next_track_id = 1
        self.tracking_enabled = True
        print("🎯 Stateful object tracking system initialized")

    def _track_objects(self, results, frame_num):
        """
        Stateful object tracking - maintains object identities across frames
        FIXED: Handle None results properly
        """
        if not self.tracking_enabled:
            # Update all trackers as not detected in this frame
            for tracker in self.object_trackers.values():
                state_change, old_state = tracker.update(None, frame_num, detected=False)
                self._handle_state_change(tracker, state_change, old_state)
            return
        
        current_detections = []
        
        # Extract current frame detections
        if results and results[0].boxes:
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
        
        # Mark all existing trackers as not detected initially
        detected_tracker_ids = set()
        
        # Match detections with existing trackers
        for detection in current_detections:
            tracker_id = self._find_tracker_match(detection['bbox'], detection['class_id'])
            
            if tracker_id is not None:
                # Update existing tracker
                tracker = self.object_trackers[tracker_id]
                state_change, old_state = tracker.update(detection['bbox'], frame_num, detected=True)
                detected_tracker_ids.add(tracker_id)
                self._handle_state_change(tracker, state_change, old_state)
            else:
                # Create new tracker for new object
                self._create_new_tracker(detection, frame_num)
        
        # Update trackers that weren't detected in this frame
        for tracker_id, tracker in self.object_trackers.items():
            if tracker_id not in detected_tracker_ids:
                state_change, old_state = tracker.update(None, frame_num, detected=False)
                self._handle_state_change(tracker, state_change, old_state)
        
        # Remove stale trackers
        self._cleanup_stale_trackers(frame_num)
        
        # Debug output (less frequent to reduce console spam)
        if self.object_trackers and frame_num % 100 == 0:
            self._print_tracking_status()
  
    def _find_tracker_match(self, detection_bbox, class_id, iou_threshold=0.3):
        """
        Find best matching tracker using Intersection over Union (IoU)
        Only matches trackers of the same class
        """
        best_tracker_id = None
        best_iou = iou_threshold
        
        for tracker_id, tracker in self.object_trackers.items():
            # Only match with trackers of the same class
            if tracker.class_id != class_id:
                continue
                
            iou = self._calculate_iou(tracker.bbox, detection_bbox)
            if iou > best_iou:
                best_iou = iou
                best_tracker_id = tracker_id
        
        return best_tracker_id

    def _calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union between two bounding boxes"""
        if bbox1 is None or bbox2 is None:
            return 0
            
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
        
        tracker = ObjectTracker(
            tracker_id, 
            detection['bbox'], 
            detection['class_name'], 
            detection['class_id'], 
            frame_num
        )
        
        self.object_trackers[tracker_id] = tracker
        print(f"🎯 New object tracked: ID{tracker_id} - {detection['class_name']}")

    def _cleanup_stale_trackers(self, current_frame_num, stale_threshold=50):
        """Remove trackers that haven't been updated recently"""
        stale_ids = []
        
        for tracker_id, tracker in self.object_trackers.items():
            if tracker.is_stale(current_frame_num, stale_threshold):
                stale_ids.append(tracker_id)
        
        for tracker_id in stale_ids:
            print(f"🧹 Removing stale tracker: ID{tracker_id}")
            del self.object_trackers[tracker_id]

    def _handle_state_change(self, tracker, state_change, old_state):
        """Handle object state changes and trigger appropriate actions"""
        if state_change == "CONFIRMED":
            print(f"🎯 Object ID{tracker.track_id} ({tracker.class_name}) confirmed - State: {old_state} → {tracker.state}")
            self._trigger_object_audio_alert(tracker)
        elif state_change == "DISAPPEARED":
            print(f"👋 Object ID{tracker.track_id} ({tracker.class_name}) disappeared")
            # Reset audio trigger for when object reappears
            tracker.audio_alert_triggered = False
        elif state_change == "REAPPEARED":
            print(f"↩️ Object ID{tracker.track_id} ({tracker.class_name}) reappeared - State: {old_state} → {tracker.state}")

    def _trigger_object_audio_alert(self, tracker):
        """Trigger audio alert based on object state"""
        if not self.audio_alert_manager:
            return
        
        # Use the object's class name for audio alert
        self.audio_alert_manager.trigger_alert(tracker.class_name, 0.9)

    def _print_tracking_status(self):
        """Print current tracking status"""
        active_objects = sum(1 for t in self.object_trackers.values() if t.state == "PRESENT")
        print(f"🎯 Tracking {len(self.object_trackers)} objects ({active_objects} active):")
        
        for tracker_id, tracker in list(self.object_trackers.items())[:3]:  # Show first 3
            state_info = tracker.get_state_info()
            print(f"   ID{tracker_id}: {state_info['class_name']} - {state_info['state']} "
                  f"(Detections: {state_info['consecutive_detections']}, Misses: {state_info['consecutive_misses']})")

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

    def _get_tracker_info_for_bbox(self, bbox, class_id, iou_threshold=0.3):
        """Get tracker information for a bounding box"""
        best_tracker = None
        best_iou = iou_threshold
        
        for tracker_id, tracker in self.object_trackers.items():
            # Only consider trackers of the same class
            if tracker.class_id != class_id:
                continue
                
            # Handle case where tracker bbox might be None
            if tracker.bbox is None:
                continue
                
            iou = self._calculate_iou(tracker.bbox, bbox)
            if iou > best_iou:
                best_iou = iou
                best_tracker = {
                    'id': tracker_id,
                    'state': tracker.state,
                    'class_name': tracker.class_name
                }
        
        return best_tracker     
   
    def _check_alerts(self, results, frame_num):
        """
        Stateful alert checking using object tracking
        """
        current_time = time.time()
        
        # Clear expired alerts from cooldown
        expired_alerts = []
        for class_id, last_alert in self.alert_cooldown.items():
            if current_time - last_alert > self.alert_cooldown_duration:
                expired_alerts.append(class_id)
        
        for class_id in expired_alerts:
            if class_id in self.active_alerts:
                self.active_alerts.remove(class_id)
        
        # Use stateful object tracking
        self._track_objects(results, frame_num)
        
        # Log all detections for historical tracking
        if results and results[0].boxes:
            for box in results[0].boxes:
                class_id = int(box.cls.item())
                confidence = box.conf.item()
                
                class_name = "unknown"
                if hasattr(self.model, 'names') and self.model.names:
                    class_name = self.model.names.get(class_id, f"class_{class_id}")
                else:
                    class_name = f"class_{class_id}"
                
                self._log_detection(class_id, class_name, confidence, frame_num, is_alert=False)
        
        # Update active alerts based on current object states
        self._update_active_alerts()
            
    def _run_yolo_detection(self, frame, frame_num):
        """Run YOLO object detection with custom bounding box rendering"""
        try:
            # Run YOLO inference
            results = self.model.predict(
                source=frame,
                conf=self.conf_threshold,
                verbose=False
            )
            
            # 🆕 Add quick debug every 50 frames
            if frame_num % 50 == 0:
                self._quick_debug_check(results, frame_num)
            
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
        """Add class-specific threshold information to overlay"""
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
        
        # STATE SYSTEM: Stateful tracking information
        tracking_info = f"Objects Tracked: {len(self.object_trackers)}"
        cv.putText(frame, tracking_info, (10, 200), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # 🆕 ADD CLASS THRESHOLD INFO
        threshold_info = f"Class Thresholds: {len(self.class_thresholds_config)} configured"
        cv.putText(frame, threshold_info, (10, 240), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)                  
        
        # Show object states (first 5 active objects)
        y_offset = 215
        active_count = 0
        for tracker_id, tracker in list(self.object_trackers.items())[:5]:
            if tracker.state == "PRESENT":
                state_info = f"ID{tracker_id}: {tracker.class_name} - {tracker.state}"
                color = (0, 255, 0)  # Green for present
                cv.putText(frame, state_info, (10, y_offset), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                y_offset += 15
                active_count += 1

        # Show thresholds for active classes (first 3)
        y_offset = 255
        active_classes = set()
        for tracker in self.object_trackers.values():
            active_classes.add(tracker.class_name)
        
        for class_name in list(active_classes)[:3]:
            thresholds = self.get_class_thresholds(class_name)
            threshold_text = f"{class_name}: C{thresholds['confirmation']}/D{thresholds['disappearance']}"
            cv.putText(frame, threshold_text, (10, y_offset), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 255), 1)
            y_offset += 12

        
        if active_count == 0:
            inactive_count = sum(1 for t in self.object_trackers.values() if t.state != "PRESENT")
            if inactive_count > 0:
                cv.putText(frame, f"{inactive_count} inactive objects", (10, 215), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            else:
                cv.putText(frame, "No objects tracked", (10, 215), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv.putText(frame, f"Time: {timestamp}", (10, 160), 
                  cv.FONT_HERSHEY_SIMPLEX, font_scale-0.1, (255, 255, 255), 1)
        cv.putText(frame, f"Interval: {self.processing_interval}s", (10, 175), 
                  cv.FONT_HERSHEY_SIMPLEX, font_scale-0.1, (255, 255, 255), 1)
        cv.putText(frame, "Press ESC to exit", (10, 190), 
                  cv.FONT_HERSHEY_SIMPLEX, font_scale-0.1, (255, 255, 255), 1)

    def _update_active_alerts(self):
        """Update active alerts based on object states"""
        current_alerts = set()
        
        for tracker in self.object_trackers.values():
            if tracker.state == "PRESENT":
                current_alerts.add(tracker.class_id)
        
        self.active_alerts = current_alerts 

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
        print(f"  - Stateful Object Tracking: Enabled")
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
    
    def _get_state_color(self, state):
        """Get color based on object state"""
        state_colors = {
            "APPEARED": (255, 255, 0),    # Yellow - initial detection
            "PRESENT": (0, 255, 0),       # Green - confirmed present
            "DISAPPEARING": (255, 165, 0), # Orange - starting to disappear
            "GONE": (128, 128, 128)       # Gray - disappeared
        }
        return state_colors.get(state, (255, 255, 255))

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
        Draw bounding boxes with object tracking states - FIXED VERSION
        """
        if not results or len(results) == 0 or not results[0].boxes:
            return frame
        
        annotated_frame = frame.copy()
        boxes = results[0].boxes
        
        for box in boxes:
            class_id = int(box.cls.item())
            confidence = box.conf.item()
            
            # 🆕 FIX: Display ALL detections, not just alert classes
            # This ensures bounding boxes are visible for all detected objects
            if hasattr(self.model, 'names') and self.model.names:
                class_name = self.model.names.get(class_id, f"class_{class_id}")
            else:
                class_name = f"class_{class_id}"
            
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # 🆕 FIX: Always get display properties, don't filter by alert_classes
            display_label, box_color, should_display = self._get_bbox_display_properties(class_id, confidence, class_name)
            
            if should_display:
                # Find tracker for this detection - only for alert classes
                tracker_info = None
                if class_id in self.alert_classes:
                    tracker_info = self._get_tracker_info_for_bbox([x1, y1, x2, y2], class_id)
                
                if tracker_info:
                    # Use tracker state for display
                    state_color = self._get_state_color(tracker_info['state'])
                    state_label = f"ID{tracker_info['id']} {tracker_info['state']}"
                    display_label = f"{state_label} | {display_label}"
                    box_color = state_color
                
                # Draw bounding box with state information
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
        
        return frame

    def _get_bbox_display_properties(self, class_id, confidence, class_name=None):
        """
        Determine display properties for a bounding box - ULTIMATE FIX
        Now displays ALL detections with minimal filtering
        Returns: (label, color, should_display)
        """
        # Always get class name first
        if class_name is None:
            if hasattr(self.model, 'names') and self.model.names:
                class_name = self.model.names.get(class_id, f"class_{class_id}")
            else:
                class_name = f"class_{class_id}"
        
        # Only filter by model's confidence threshold
        if confidence < self.conf_threshold:
            return "", (0, 0, 0), False
        
        # Check custom bbox config without additional filtering
        if class_id in self.bbox_label_map:
            config = self.bbox_label_map[class_id]
            # Use custom label and color, but don't apply extra confidence filtering
            return config['label'], config['color'], True
        
        # Display ALL classes that pass model confidence
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), 
                (255, 0, 255), (0, 255, 255), (128, 0, 128)]
        color = colors[class_id % len(colors)]
        
        # Use alert class name if available, otherwise use model name
        if class_id in self.alert_classes:
            display_name = self.alert_classes[class_id]
        else:
            display_name = class_name
            
        return display_name, color, True
            
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


    # Add these methods to SelectiveFrameProcessor for interactive control
    def set_class_threshold(self, class_name, confirmation_threshold, disappearance_threshold):
        """Set thresholds for a specific class"""
        self.update_class_thresholds(class_name, confirmation_threshold, disappearance_threshold)
    
    def print_class_thresholds(self):
        """Print current class threshold configuration"""
        print("🎯 CURRENT CLASS THRESHOLDS:")
        for class_name, thresholds in self.class_thresholds_config.items():
            print(f"  {class_name}: confirm={thresholds['confirmation']}, disappear={thresholds['disappearance']}")
    
    def reset_class_thresholds(self):
        """Reset all class thresholds to defaults"""
        self.class_thresholds_config = self._initialize_class_thresholds()
        # Update all active trackers
        for tracker in self.object_trackers.values():
            thresholds = self.get_class_thresholds(tracker.class_name)
            tracker.confirmation_threshold = thresholds["confirmation"]
            tracker.disappearance_threshold = thresholds["disappearance"]
        print("✅ All class thresholds reset to defaults")

    def _cleanup_duplicate_systems(self):
        """Clean up duplicate tracking systems"""
        # Remove old person tracking system if it exists
        if hasattr(self, 'person_trackers'):
            delattr(self, 'person_trackers')
        
        # Ensure we only have one tracking system
        if not hasattr(self, 'object_trackers'):
            self.object_trackers = {}
            self.next_track_id = 1

def _quick_debug_check(self, results, frame_num):
    """Lightweight debug to verify detections are working"""
    if results and results[0].boxes:
        detections = len(results[0].boxes)
        print(f"🔍 Frame {frame_num}: {detections} detections found")
        
        # Show first 3 detections
        for i, box in enumerate(results[0].boxes[:3]):
            class_id = int(box.cls.item())
            confidence = box.conf.item()
            if hasattr(self.model, 'names') and self.model.names:
                class_name = self.model.names.get(class_id, f"class_{class_id}")
            else:
                class_name = f"class_{class_id}"
            print(f"   Detection {i}: {class_name} (ID:{class_id}) - {confidence:.2f}")

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