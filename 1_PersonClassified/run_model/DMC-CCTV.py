import cv2
from ultralytics import YOLO
from pathlib import Path
import pygame
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Dict, Tuple, Any, Union, Optional, List
import numpy as np
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ObjectDetection.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class AppConfig:
    """Centralized configuration for the application"""
    # Model configuration
    model_path: Path = None
    
    # Video source configuration
    camera_source: Union[int, str] = 0
    frame_width: int = 640
    frame_height: int = 480
    
    # Detection configuration
    confidence_threshold: float = 0.5
    class_color_map: Dict[int, Tuple[int, int, int]] = None
    
    # Alert configuration
    alert_cooldown: int = 5
    class_cooldowns: Dict[int, int] = None
    
    # Counter configuration
    counter_time_window: int = 10
    
    # RTSP configuration
    reconnect_delay: int = 5
    max_consecutive_failures: int = 3
    
    # Conditional labeling configuration
    special_classes: Tuple[int, ...] = (0, 1, 2)
    conditional_box_colors: Dict[str, Tuple[int, int, int]] = None
    
    # Frame skipping configuration
    frame_skip: int = 1
    display_every_frame: bool = True
    target_fps: int = 30

    # Enhanced audio configuration
    class_sound_map: Dict[int, Path] = None
    default_alert_sound: Path = None
    
    def __post_init__(self):
        # Initialize audio configuration
        if self.class_sound_map is None:
            self.class_sound_map = {
                0: Path(r"D:\RaihanFarid\Dokumen\Object Detection\usedAudio\level-up-07-383747.mp3"),
                1: Path(r"D:\RaihanFarid\Dokumen\Object Detection\usedAudio\new-notification-09-352705.mp3"),
            }
        
        if self.default_alert_sound is None:
            self.default_alert_sound = Path(r"D:\RaihanFarid\Dokumen\Object Detection\usedAudio\notification-alert-269289.mp3")

        # Initialize default values for mutable objects
        if self.model_path is None:
            self.model_path = Path(r"D:\RaihanFarid\Dokumen\Object Detection\CV_model\HBDetect.torchscript")
            
        if self.class_color_map is None:
            self.class_color_map = {0: (0, 0, 255), 1: (0, 255, 0)}
        
        if self.class_cooldowns is None:
            self.class_cooldowns = {0: 10, 1: 5, 2: 15}
            
        if self.conditional_box_colors is None:
            self.conditional_box_colors = {
                "all_three": (0, 255, 255),
                "two_classes": (255, 0, 255)
            }
        
        if self.frame_skip < 1:
            self.frame_skip = 1

class AlertManager:
    """Simplified alert manager that uses AudioManager"""
    
    def __init__(self, config: AppConfig, alert_timers=None):
        self.config = config
        self.audio_manager = AudioManager(config)
        self.alert_timers = alert_timers if alert_timers is not None else {}
    
    def should_trigger_alert(self, cls_id: int) -> bool:
        """Check if an alert should be triggered based on cooldown"""
        return self.audio_manager.should_trigger_alert(cls_id)
    
    def play_alert(self, class_id: int = None):
        """Play alert sound for specific class or default"""
        try:
            if class_id is not None:
                self.audio_manager.play_class_alert(class_id)
            else:
                self.audio_manager.play_default_alert()
        except Exception as e:
            logger.error(f"Error in play_alert: {e}")
    
    def get_audio_status(self):
        """Get audio system status"""
        return self.audio_manager.get_audio_status()

class TimeWindowCounter:
    """Manages a time-based counter with a rolling window"""
    def __init__(self, time_window_seconds: int = 10):
        self.time_window = time_window_seconds
        self.detection_history = deque()
        self.current_counts = Counter()
    
    def add_detection(self, class_name: str):
        """Add a detection to the history and update counts"""
        try:
            current_time = time.time()
            self.detection_history.append((current_time, class_name))
            self.current_counts[class_name] += 1
            self._prune_old_detections(current_time)
        except Exception as e:
            logger.error(f"Error adding detection to time window counter: {e}")
    
    def _prune_old_detections(self, current_time: float):
        """Remove detections older than the time window"""
        try:
            while self.detection_history and current_time - self.detection_history[0][0] > self.time_window:
                old_time, old_class = self.detection_history.popleft()
                self.current_counts[old_class] -= 1
                if self.current_counts[old_class] <= 0:
                    del self.current_counts[old_class]
        except Exception as e:
            logger.error(f"Error pruning old detections: {e}")
    
    def get_counts(self) -> Counter:
        """Get current counts within the time window"""
        try:
            current_time = time.time()
            self._prune_old_detections(current_time)
            return self.current_counts.copy()
        except Exception as e:
            logger.error(f"Error getting counts: {e}")
            return Counter()
    
    def get_time_remaining(self) -> float:
        """Get time until the next counter update"""
        try:
            if not self.detection_history:
                return self.time_window
            
            current_time = time.time()
            oldest_time = self.detection_history[0][0]
            time_elapsed = current_time - oldest_time
            return max(0, self.time_window - time_elapsed)
        except Exception as e:
            logger.error(f"Error getting time remaining: {e}")
            return self.time_window

class DetectionVisualizer:
    """Handles visualization of detection results"""
    def __init__(self, config: AppConfig):
        self.config = config
        self.line_height = 22
        self.padding = 8
    
    def draw_detection(self, frame, box, class_name, conf, cls_id, is_conditional=False, conditional_type=None):
        """Draw a single detection bounding box and label"""
        try:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Choose color based on whether this is a conditional box
            if is_conditional:
                if conditional_type == "all_three":
                    color = self.config.conditional_box_colors["all_three"]
                elif conditional_type == "two_classes":
                    color = self.config.conditional_box_colors["two_classes"]
                else:
                    color = (255, 255, 255)
            else:
                color = self.config.class_color_map.get(cls_id, (255, 255, 255))
            
            # Draw bounding box
            thickness = 4 if is_conditional else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Only draw individual labels for non-conditional boxes
            if not is_conditional:
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
        except Exception as e:
            logger.error(f"Error drawing detection: {e}")
            return "Error"
    
    def draw_conditional_bounding_box(self, frame, boxes, conditional_type):
        """Draw a special bounding box that encompasses all special class detections"""
        try:
            if not boxes:
                return
            
            # Calculate the combined bounding box
            x1 = min(int(box.xyxy[0][0]) for box in boxes)
            y1 = min(int(box.xyxy[0][1]) for box in boxes)
            x2 = max(int(box.xyxy[0][2]) for box in boxes)
            y2 = max(int(box.xyxy[0][3]) for box in boxes)
            
            # Add padding
            padding = 10
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(frame.shape[1], x2 + padding)
            y2 = min(frame.shape[0], y2 + padding)
            
            # Choose color based on conditional type
            if conditional_type == "all_three":
                color = self.config.conditional_box_colors["all_three"]
                label = "ALL THREE CLASSES"
            else:
                color = self.config.conditional_box_colors["two_classes"]
                label = "TWO SPECIAL CLASSES"
            
            # Draw the combined bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
            
            # Draw label
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 3
            )
            
            # Draw label background
            cv2.rectangle(
                frame, 
                (x1, y1 - text_height - 15), 
                (x1 + text_width + 10, y1), 
                color, 
                -1
            )
            
            # Draw label text
            cv2.putText(
                frame, 
                label, 
                (x1 + 5, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.9, 
                (255, 255, 255), 
                3
            )
        except Exception as e:
            logger.error(f"Error drawing conditional bounding box: {e}")
    
    def draw_text_panel(self, frame, lines, x0=10, y0=30, panel_w=250, right_align=False):
        """Draw a semi-transparent text panel with information"""
        try:
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
        except Exception as e:
            logger.error(f"Error drawing text panel: {e}")
    
    def draw_cooldown_status(self, frame, alert_manager):
        """Display cooldown status for each class in top-left"""
        try:
            current_time = time.time()
            for i, (cls_id, last_alert) in enumerate(alert_manager.alert_timers.items()):
                time_remaining = max(0, self.config.alert_cooldown - (current_time - last_alert))
                status_text = f"Class {cls_id} alert: {time_remaining:.1f}s cooldown"
                cv2.putText(
                    frame, status_text, (10, 30 + i*30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
                )
        except Exception as e:
            logger.error(f"Error drawing cooldown status: {e}")
    
    def draw_connection_status(self, frame, is_connected, failure_count):
        """Display connection status"""
        try:
            status_text = "Connected" if is_connected else f"Disconnected ({failure_count})"
            color = (0, 255, 0) if is_connected else (0, 0, 255)
            cv2.putText(
                frame, f"Status: {status_text}", (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )
        except Exception as e:
            logger.error(f"Error drawing connection status: {e}")
    
    def draw_error_message(self, frame, message):
        """Display error message on the frame"""
        try:
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
        except Exception as e:
            logger.error(f"Error drawing error message: {e}")
    
    def draw_audio_status(self, frame, alert_manager):
        """Display audio system status"""
        try:
            audio_status = alert_manager.get_audio_status()
            
            status_lines = [
                "Audio Status:",
                f"Initialized: {'Yes' if audio_status['initialized'] else 'No'}",
                f"Loaded Sounds: {audio_status['sounds_loaded']}",
                f"Available Channels: {audio_status['active_channels']}",
                f"Class Sounds: {len(audio_status['class_sounds'])} classes"
            ]
            
            # Draw audio status panel in bottom-left
            self.draw_text_panel(frame, status_lines, 10, frame.shape[0] - 150, 300)
            
        except Exception as e:
            logger.error(f"Error drawing audio status: {e}")

class AudioManager:
    """Modular audio manager for class-specific sound conditioning"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.sound_library = {}
        self.alert_timers = {}
        self._initialized = False
        
        self.initialize_audio()
    
    def initialize_audio(self):
        """Initialize audio system and load sounds"""
        try:
            pygame.mixer.init()
            pygame.mixer.set_num_channels(16)
            self._load_sound_library()
            self._initialized = True
            logger.info("AudioManager initialized successfully")
            
        except pygame.error as e:
            logger.error(f"PyGame mixer initialization failed: {e}")
            self._initialized = False
        except Exception as e:
            logger.error(f"AudioManager initialization error: {e}")
            self._initialized = False
    
    def _load_sound_library(self):
        """Load all configured sounds into memory"""
        try:
            # Load class-specific sounds
            for class_id, sound_path in self.config.class_sound_map.items():
                if sound_path.exists():
                    self.sound_library[class_id] = pygame.mixer.Sound(str(sound_path))
                    logger.info(f"Loaded sound for class {class_id}: {sound_path}")
                else:
                    logger.warning(f"Sound file not found for class {class_id}: {sound_path}")
            
            # Load default sound
            if self.config.default_alert_sound.exists():
                self.sound_library['default'] = pygame.mixer.Sound(str(self.config.default_alert_sound))
                logger.info(f"Loaded default sound: {self.config.default_alert_sound}")
            else:
                logger.warning(f"Default sound file not found: {self.config.default_alert_sound}")
                
        except Exception as e:
            logger.error(f"Error loading sound library: {e}")
    
    def get_sound_for_class(self, class_id: int) -> Optional[pygame.mixer.Sound]:
        """Get the appropriate sound for a class ID"""
        try:
            if class_id in self.sound_library:
                return self.sound_library[class_id]
            
            if 'default' in self.sound_library:
                return self.sound_library['default']
            
            return None
        except Exception as e:
            logger.error(f"Error getting sound for class {class_id}: {e}")
            return None
    
    def should_trigger_alert(self, cls_id: int) -> bool:
        """Check if an alert should be triggered based on cooldown"""
        try:
            current_time = time.time()
            cooldown = self.config.class_cooldowns.get(cls_id, self.config.alert_cooldown)
            
            if cls_id not in self.alert_timers or (current_time - self.alert_timers[cls_id]) >= cooldown:
                self.alert_timers[cls_id] = current_time
                return True
            return False
        except Exception as e:
            logger.error(f"Error in should_trigger_alert: {e}")
            return False
    
    def play_class_alert(self, class_id: int):
        """Play the appropriate sound for a specific class"""
        try:
            if not self._initialized:
                logger.warning("Audio system not initialized")
                return
            
            sound = self.get_sound_for_class(class_id)
            if sound:
                channel = pygame.mixer.find_channel()
                if channel:
                    channel.play(sound)
                    logger.debug(f"Playing sound for class {class_id}")
                else:
                    logger.warning("No available audio channels")
            else:
                logger.warning(f"No sound available for class {class_id}")
                
        except Exception as e:
            logger.error(f"Error playing alert for class {class_id}: {e}")
    
    def play_default_alert(self):
        """Play the default alert sound"""
        self.play_class_alert('default')
    
    def stop_all_sounds(self):
        """Stop all currently playing sounds"""
        try:
            pygame.mixer.stop()
        except Exception as e:
            logger.error(f"Error stopping sounds: {e}")
    
    def get_audio_status(self) -> Dict[str, Any]:
        """Get current audio system status"""
        return {
            'initialized': self._initialized,
            'sounds_loaded': len(self.sound_library),
            'active_channels': pygame.mixer.get_num_channels() - pygame.mixer.get_busy(),
            'class_sounds': list(self.config.class_sound_map.keys())
        }

class FPSCounter:
    """Simple FPS counter"""
    def __init__(self):
        self.start_time = time.time()
        self.frame_count = 0
        self.current_fps = 0
        
    def update(self):
        """Update FPS calculation"""
        self.frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        if elapsed >= 1.0:
            self.current_fps = self.frame_count / elapsed
            self.frame_count = 0
            self.start_time = current_time
    
    def get_fps(self):
        """Get current FPS"""
        return self.current_fps

class ObjectDetector:
    """Simplified object detection workflow without ModelManager"""
    def __init__(self, config: AppConfig):
        self.config = config
        self.model = None
        self.cap = None
        self.alert_manager = None
        self.visualizer = None
        self.time_window_counter = None
        self.consecutive_failures = 0
        
        # Frame skipping variables
        self.frame_count = 0
        self.last_processed_frame = None
        self.fps_counter = FPSCounter()
        
        # Initialize components
        try:
            self._load_model()
            self.alert_manager = AlertManager(config)
            self.visualizer = DetectionVisualizer(config)
            self.time_window_counter = TimeWindowCounter(config.counter_time_window)
            
            if not self.setup_video_source():
                raise RuntimeError("Failed to initialize video source")
                
            logger.info("ObjectDetector initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ObjectDetector: {e}")
            raise

    def _load_model(self):
        """Load the detection model"""
        try:
            if not self.config.model_path.exists():
                raise FileNotFoundError(f"Model file not found: {self.config.model_path}")
            
            logger.info(f"Loading detection model: {self.config.model_path}")
            self.model = YOLO(str(self.config.model_path))
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def should_process_frame(self) -> bool:
        """Determine if current frame should be processed based on frame_skip setting"""
        return self.frame_count % self.config.frame_skip == 0

    def process_frame_with_skipping(self, frame):
        """Process frame with frame skipping logic"""
        try:
            self.frame_count += 1
            
            if self.should_process_frame():
                processed_frame = self.process_frame(frame)
                self.last_processed_frame = processed_frame
                return processed_frame
            else:
                if self.last_processed_frame is not None and self.config.display_every_frame:
                    return self.last_processed_frame
                else:
                    return cv2.resize(frame, (1024, 576))
                    
        except Exception as e:
            logger.error(f"Error in process_frame_with_skipping: {e}")
            return cv2.resize(frame, (1024, 576))
    
    def setup_video_source(self):
        """Initialize and configure the video source"""
        try:
            if self.cap is not None:
                self.cap.release()
                
            if isinstance(self.config.camera_source, str) and self.config.camera_source.isdigit():
                camera_source = int(self.config.camera_source)
            else:
                camera_source = self.config.camera_source
                
            self.cap = cv2.VideoCapture(camera_source)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open video source: {camera_source}")
                return False
            
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
            ret, frame = self.cap.read()
            if not ret:
                logger.error("Failed to read test frame from video source")
                return False
                
            logger.info(f"Video source initialized: {camera_source}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up video source: {e}")
            return False
    
    def process_frame(self, frame):
        """Process a single frame for object detection"""
        try:
            frame_counts = Counter()
            special_detections = []
            
            if self.model is None:
                error_msg = "No model available for processing"
                logger.error(error_msg)
                self.visualizer.draw_error_message(frame, error_msg)
                return frame
            
            # Process frame for detection only
            results = self.model(frame, stream=True, conf=self.config.confidence_threshold)
            
            for r in results:
                if hasattr(r, 'boxes') and r.boxes is not None:
                    for box in r.boxes:
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        
                        if conf > self.config.confidence_threshold:
                            class_name = self.model.names[cls_id]
                            
                            # Play class-specific alert
                            if cls_id in self.config.class_cooldowns and self.alert_manager.should_trigger_alert(cls_id):
                                self.alert_manager.play_alert(cls_id)
                            
                            # Count the detection
                            frame_counts[class_name] += 1
                            self.time_window_counter.add_detection(class_name)
                            
                            # Check if this is a special class
                            if cls_id in self.config.special_classes:
                                special_detections.append((box, cls_id, conf, class_name))
                            else:
                                self.visualizer.draw_detection(frame, box, class_name, conf, cls_id)
            
            # Process special class detections
            special_class_ids = set(cls_id for _, cls_id, _, _ in special_detections)
            
            if special_class_ids == set(self.config.special_classes):
                special_boxes = [box for box, _, _, _ in special_detections]
                self.visualizer.draw_conditional_bounding_box(frame, special_boxes, "all_three")
            elif len(special_class_ids) == 2:
                special_boxes = [box for box, _, _, _ in special_detections]
                self.visualizer.draw_conditional_bounding_box(frame, special_boxes, "two_classes")
            else:
                for box, cls_id, conf, class_name in special_detections:
                    self.visualizer.draw_detection(frame, box, class_name, conf, cls_id)
            
            # Get time window counts
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
                
            self.visualizer.draw_text_panel(frame, lines, 10, 30, 250, right_align=True)
            
            # Display cooldown status
            self.visualizer.draw_cooldown_status(frame, self.alert_manager)
            
            # Display audio status
            self.visualizer.draw_audio_status(frame, self.alert_manager)
            
            # Display connection status
            is_connected = self.cap.isOpened() and self.consecutive_failures < self.config.max_consecutive_failures
            self.visualizer.draw_connection_status(frame, is_connected, self.consecutive_failures)

            frame = cv2.resize(frame, (1024, 576))
            return frame
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            error_msg = f"Processing Error: {str(e)[:50]}..."
            self.visualizer.draw_error_message(frame, error_msg)
            return frame
 
    def run(self):
        """Main application loop with reconnection capability"""
        try:
            blank_frame = None
            
            while True:
                # Check connection status and reconnect if needed
                if not self.cap.isOpened() or self.consecutive_failures >= self.config.max_consecutive_failures:
                    logger.warning(f"Attempting to reconnect to video source... (Failures: {self.consecutive_failures})")
                    try:
                        if self.setup_video_source():
                            self.consecutive_failures = 0
                            logger.info("Reconnection successful")
                        else:
                            logger.error("Reconnection failed")
                            time.sleep(self.config.reconnect_delay)
                            continue
                    except Exception as e:
                        logger.error(f"Reconnection error: {e}")
                        time.sleep(self.config.reconnect_delay)
                        continue
                
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    self.consecutive_failures += 1
                    logger.warning(f"Failed to read frame ({self.consecutive_failures}/{self.config.max_consecutive_failures})")
                    
                    if blank_frame is None:
                        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    
                    error_msg = f"Cannot read from video source\nFailures: {self.consecutive_failures}/{self.config.max_consecutive_failures}"
                    self.visualizer.draw_error_message(blank_frame, error_msg)
                    cv2.imshow('YOLO Live Detection', blank_frame)
                    
                    time.sleep(1)
                    continue
                
                # Reset failure counter on successful frame read
                self.consecutive_failures = 0
                
                # Process frame with skipping logic
                processed_frame = self.process_frame_with_skipping(frame)
                
                # Add FPS information to frame
                self.fps_counter.update()
                fps = self.fps_counter.get_fps()
                skip_status = f"Skip: {self.config.frame_skip}x" if self.config.frame_skip > 1 else "Processing all frames"
                
                cv2.putText(
                    processed_frame, 
                    f"FPS: {fps:.1f} | {skip_status} | Frame: {self.frame_count}", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (255, 255, 0), 
                    2
                )
                
                # Display the resulting frame
                cv2.imshow('YOLO Live Detection', processed_frame)
                
                # Exit on 'q' key press, allow changing skip rate with number keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif ord('1') <= key <= ord('9'):
                    new_skip = key - ord('0')
                    self.config.frame_skip = new_skip
                    logger.info(f"Frame skip rate changed to: {new_skip}")
                    
        except Exception as e:
            logger.critical(f"Critical error in main loop: {e}")
        finally:
            try:
                if self.cap is not None:
                    self.cap.release()
                cv2.destroyAllWindows()
                logger.info("Application shut down successfully")
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")

def main():
    """Main function to initialize and run the application"""
    try:
        # Configuration
        #camera_source = 0
        camera_source = "rtsp://admin:CemaraMas2025!@192.168.2.190:554/Streaming/Channels/101"
        
        # Enhanced audio configuration
        class_sound_config = {
            0: Path(r"D:\RaihanFarid\Dokumen\Object Detection\usedAudio\positive-notification-alert-351299.mp3"),
            1: Path(r"D:\RaihanFarid\Dokumen\Object Detection\usedAudio\new-notification-026-380249.mp3"),
        }
        
        config = AppConfig(
            model_path=Path(r"D:\RaihanFarid\Dokumen\Object Detection\CV_model\CCTV1.onnx"),
            camera_source=camera_source,
            frame_width=640,
            frame_height=480,
            confidence_threshold=0.5,
            class_color_map={1: (0, 0, 255), 0: (0, 255, 0)},
            alert_cooldown=2,
            class_cooldowns={0: 5, 1: 5, 2: 2},
            counter_time_window=10,
            reconnect_delay=5,
            max_consecutive_failures=3,
            special_classes=(0, 1, 2),
            conditional_box_colors={
                "all_three": (0, 255, 255),
                "two_classes": (255, 0, 255)
            },
            class_sound_map=class_sound_config,
            frame_skip=1,
            display_every_frame=True,
            target_fps=30
        )
        
        # Create and run the detector
        detector = ObjectDetector(config)
        detector.run()
        
    except Exception as e:
        logger.critical(f"Failed to start application: {e}")
        print(f"Application failed to start: {e}")
        print("Check ObjectDetection.log for details")

if __name__ == "__main__":
    main()