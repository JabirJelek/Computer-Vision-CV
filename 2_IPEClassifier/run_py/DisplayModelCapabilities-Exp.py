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
        logging.FileHandler("DisplayModelCapabilities-Exp.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class AppConfig:
    """Centralized configuration for the application"""
    # Model configuration - now supports multiple models with fallback
    model_paths: List[Path] = None
    
    # Video source configuration - can be camera index (0, 1, 2) or RTSP URL
    camera_source: Union[int, str] = 0  # Can be 0 for webcam or "rtsp://..." for RTSP
    frame_width: int = 640
    frame_height: int = 480
    
    # Detection configuration
    confidence_threshold: float = 0.5
    class_color_map: Dict[int, Tuple[int, int, int]] = None
    
    # Alert configuration
    alert_sound_path: Path = Path(r"D:\RaihanFarid\Dokumen\Object Detection\usedAudio\notification-alert-269289.mp3")
    alert_cooldown: int = 5
    class_cooldowns: Dict[int, int] = None
    
    # Counter configuration
    counter_time_window: int = 10  # Time window in seconds for the cumulative counter
    
    # RTSP configuration
    reconnect_delay: int = 5  # Seconds between reconnection attempts
    max_consecutive_failures: int = 3  # Maximum failures before attempting reconnection
    
    # Conditional labeling configuration
    special_classes: Tuple[int, ...] = (0, 1, 2)  # Classes to check for conditional labeling
    conditional_box_colors: Dict[str, Tuple[int, int, int]] = None  # Colors for conditional boxes
    
    # Model fallback configuration
    model_load_timeout: int = 30  # Seconds to wait for model to load
    model_retry_interval: int = 5  # Seconds between model loading retries
    
    # Frame skipping configuration for high FPS
    frame_skip: int = 1  # Process every nth frame (1 = process all frames)
    display_every_frame: bool = True  # Whether to display every frame or only processed ones
    target_fps: int = 30  # Target FPS for display

    # Enhanced audio configuration
    class_sound_map: Dict[int, Path] = None  # Map class IDs to specific sound files
    default_alert_sound: Path = None  # Fallback sound for unmapped classes    
    
    def __post_init__(self):
        # Initialize audio configuration
        if self.class_sound_map is None:
            self.class_sound_map = {
                0: Path(r"D:\RaihanFarid\Dokumen\Object Detection\usedAudio\level-up-07-383747.mp3"),
                1: Path(r"D:\RaihanFarid\Dokumen\Object Detection\usedAudio\new-notification-09-352705.mp3"),
                #2: Path(r"D:\RaihanFarid\Dokumen\Object Detection\usedAudio\class2_alert.mp3"),
            }
        
        if self.default_alert_sound is None:
            self.default_alert_sound = Path(r"D:\RaihanFarid\Dokumen\Object Detection\usedAudio\notification-alert-269289.mp3")

        # Initialize default values for mutable objects
        if self.model_paths is None:
            self.model_paths = [Path(r"D:\RaihanFarid\Dokumen\Object Detection\CV_model\HBDetect.torchscript")]
            
        if self.class_color_map is None:
            self.class_color_map = {0: (0, 0, 255), 1: (0, 255, 0)}
        
        if self.class_cooldowns is None:
            self.class_cooldowns = {0: 10, 1: 5, 2: 15}
            
        if self.conditional_box_colors is None:
            self.conditional_box_colors = {
                "all_three": (0, 255, 255),  # Yellow for all three classes
                "two_classes": (255, 0, 255)  # Magenta for two classes
            }
        
        # Validate frame skip value
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
        self.detection_history = deque()  # Stores (timestamp, class_name) tuples
        self.current_counts = Counter()
    
    def add_detection(self, class_name: str):
        """Add a detection to the history and update counts"""
        try:
            current_time = time.time()
            self.detection_history.append((current_time, class_name))
            self.current_counts[class_name] += 1
            
            # Remove old detections outside the time window
            self._prune_old_detections(current_time)
        except Exception as e:
            logger.error(f"Error adding detection to time window counter: {e}")
    
    def _prune_old_detections(self, current_time: float):
        """Remove detections older than the time window"""
        try:
            while self.detection_history and current_time - self.detection_history[0][0] > self.time_window:
                old_time, old_class = self.detection_history.popleft()
                self.current_counts[old_class] -= 1
                
                # Remove zero counts to keep the counter clean
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
        """Get time until the next counter update (oldest detection expires)"""
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
                    color = (255, 255, 255)  # Default white
            else:
                color = self.config.class_color_map.get(cls_id, (255, 255, 255))
            
            # Draw bounding box with different thickness for conditional boxes
            thickness = 4 if is_conditional else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Only draw individual labels for non-conditional boxes
            if not is_conditional:
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
        except Exception as e:
            logger.error(f"Error drawing detection: {e}")
            return "Error"
    
    def draw_segmentation_mask(self, frame, mask, cls_id, alpha=0.5):
        """Draw segmentation mask with transparency"""
        try:
            # Get the color for this class
            color = self.config.class_color_map.get(cls_id, (255, 255, 255))
            
            # Convert mask to binary and apply color
            mask_binary = mask.cpu().numpy().astype(bool)
            colored_mask = np.zeros_like(frame)
            colored_mask[mask_binary] = color
            
            # Blend with original frame
            frame[mask_binary] = cv2.addWeighted(frame[mask_binary], 1 - alpha, 
                                                colored_mask[mask_binary], alpha, 0)
        except Exception as e:
            logger.error(f"Error drawing segmentation mask: {e}")
    
    def draw_conditional_bounding_box(self, frame, boxes, conditional_type):
        """Draw a special bounding box that encompasses all special class detections"""
        try:
            if not boxes:
                return
            
            # Calculate the combined bounding box that contains all special detections
            x1 = min(int(box.xyxy[0][0]) for box in boxes)
            y1 = min(int(box.xyxy[0][1]) for box in boxes)
            x2 = max(int(box.xyxy[0][2]) for box in boxes)
            y2 = max(int(box.xyxy[0][3]) for box in boxes)
            
            # Add some padding to the combined box
            padding = 10
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(frame.shape[1], x2 + padding)
            y2 = min(frame.shape[0], y2 + padding)
            
            # Choose color based on conditional type
            if conditional_type == "all_three":
                color = self.config.conditional_box_colors["all_three"]
                label = "ALL THREE CLASSES"
            else:  # two_classes
                color = self.config.conditional_box_colors["two_classes"]
                label = "TWO SPECIAL CLASSES"
            
            # Draw the combined bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
            
            # Draw label for the combined box
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
    
    def draw_model_status(self, frame, model_type, model_index, total_models):
        """Display current model type and status"""
        try:
            status_text = f"Model: {model_type} ({model_index+1}/{total_models})"
            cv2.putText(
                frame, status_text, (10, frame.shape[0] - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
            )
        except Exception as e:
            logger.error(f"Error drawing model status: {e}")

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
        self.sound_library = {}  # Cache for loaded sounds
        self.alert_timers = {}
        self._initialized = False
        
        self.initialize_audio()
    
    def initialize_audio(self):
        """Initialize audio system and load sounds"""
        try:
            pygame.mixer.init()
            pygame.mixer.set_num_channels(16)  # Allow multiple concurrent sounds
            
            # Pre-load all class-specific sounds
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
            # Try class-specific sound first
            if class_id in self.sound_library:
                return self.sound_library[class_id]
            
            # Fall back to default sound
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
                # Find an available channel
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
        self.play_class_alert('default')  # Use string key for default
    
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

class ModelManager:
    """Manages multiple models with fallback functionality"""
    def __init__(self, config: AppConfig):
        self.config = config
        self.models = []
        self.current_model_index = 0
        self.model_types = []  # 'detection' or 'segmentation'
        
    def load_models(self):
        """Load all models with fallback support"""
        for i, model_path in enumerate(self.config.model_paths):
            try:
                if not model_path.exists():
                    logger.warning(f"Model file not found: {model_path}")
                    continue
                
                # Determine model type based on file extension or other criteria
                model_type = self.determine_model_type(model_path)
                
                logger.info(f"Loading {model_type} model: {model_path}")
                
                # Load the model with timeout
                model = self.load_model_with_timeout(model_path, model_type)
                if model is not None:
                    self.models.append(model)
                    self.model_types.append(model_type)
                    logger.info(f"Successfully loaded {model_type} model: {model_path}")
                    
                    # If we have at least one model, we can proceed
                    if len(self.models) > 0:
                        return True
                
            except Exception as e:
                logger.error(f"Failed to load model {model_path}: {e}")
        
        return len(self.models) > 0
    
    def determine_model_type(self, model_path: Path) -> str:
        """Determine if model is for detection or segmentation"""
        # Check file extension first
        if model_path.suffix == '.pt':
            # Try to load and check model metadata
            try:
                model = YOLO(str(model_path), task='detect', verbose=False)
                # Simple heuristic: if model has segmentation head, it's segmentation
                if hasattr(model, 'model') and hasattr(model.model, 'model'):
                    for layer in model.model.model:
                        if hasattr(layer, 'proto') or hasattr(layer, 'nm'):
                            return 'segmentation'
            except:
                pass
        
        # Default to detection
        return 'detection'
    
    def load_model_with_timeout(self, model_path: Path, model_type: str):
        """Load model with timeout to prevent hanging"""
        result = [None]  # Use list to store result from thread
        
        def load_model_thread():
            try:
                model = YOLO(str(model_path), task='detect' if model_type == 'detection' else 'segment')
                result[0] = model
            except Exception as e:
                logger.error(f"Error loading model in thread: {e}")
        
        thread = threading.Thread(target=load_model_thread)
        thread.daemon = True
        thread.start()
        
        # Wait for thread to complete or timeout
        thread.join(self.config.model_load_timeout)
        
        if thread.is_alive():
            logger.error(f"Timeout loading model: {model_path}")
            return None
        
        return result[0]
    
    def get_current_model(self):
        """Get the current active model"""
        if not self.models:
            return None
        return self.models[self.current_model_index]
    
    def get_current_model_type(self):
        """Get the type of the current active model"""
        if not self.model_types:
            return None
        return self.model_types[self.current_model_index]
    
    def switch_to_next_model(self):
        """Switch to the next available model"""
        if len(self.models) <= 1:
            return False
        
        self.current_model_index = (self.current_model_index + 1) % len(self.models)
        logger.info(f"Switched to model {self.current_model_index + 1}/{len(self.models)}")
        return True

class ObjectDetector:
    """Main class for object detection workflow"""
    def __init__(self, config: AppConfig):
        self.config = config
        self.model_manager = None
        self.cap = None
        self.alert_manager = None
        self.visualizer = None
        self.time_window_counter = None
        self.consecutive_failures = 0
        self.model_failures = 0
        self.max_model_failures = 3  # Maximum model failures before switching
        
        # Frame skipping variables
        self.frame_count = 0
        self.last_processed_frame = None
        self.fps_counter = FPSCounter()  # Add FPS counter
        
        # Initialize components with error handling
        try:
            self.model_manager = ModelManager(config)
            if not self.model_manager.load_models():
                raise RuntimeError("Failed to load any models")
                
            self.alert_manager = AlertManager(config)
            self.visualizer = DetectionVisualizer(config)
            self.time_window_counter = TimeWindowCounter(config.counter_time_window)
            
            # Initialize video source
            if not self.setup_video_source():
                raise RuntimeError("Failed to initialize video source")
                
            logger.info("ObjectDetector initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ObjectDetector: {e}")
            raise
    
    def should_process_frame(self) -> bool:
        """Determine if current frame should be processed based on frame_skip setting"""
        return self.frame_count % self.config.frame_skip == 0
    
    def process_frame_with_skipping(self, frame):
        """Process frame with frame skipping logic"""
        try:
            # Update frame counter
            self.frame_count += 1
            
            # Check if we should process this frame
            if self.should_process_frame():
                # Process the frame normally
                processed_frame = self.process_frame(frame)
                self.last_processed_frame = processed_frame
                return processed_frame
            else:
                # Use the last processed frame or current frame with overlay
                if self.last_processed_frame is not None and self.config.display_every_frame:
                    # We have a previous processed frame, use it
                    return self.last_processed_frame
                else:
                    # No previous processed frame or we want to show raw frames
                    # Just resize the current frame without processing
                    return cv2.resize(frame, (1024, 576))
                    
        except Exception as e:
            logger.error(f"Error in process_frame_with_skipping: {e}")
            # Fallback to basic frame display
            return cv2.resize(frame, (1024, 576))
        
        # Initialize components with error handling
        try:
            self.model_manager = ModelManager(config)
            if not self.model_manager.load_models():
                raise RuntimeError("Failed to load any models")
                
            self.alert_manager = AlertManager(config)
            self.visualizer = DetectionVisualizer(config)
            self.time_window_counter = TimeWindowCounter(config.counter_time_window)
            
            # Initialize video source
            if not self.setup_video_source():
                raise RuntimeError("Failed to initialize video source")
                
            logger.info("ObjectDetector initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ObjectDetector: {e}")
            raise
    
    def setup_video_source(self):
        """Initialize and configure the video source (camera or RTSP)"""
        try:
            if self.cap is not None:
                self.cap.release()
                
            # Convert camera index to int if it's a numeric string
            if isinstance(self.config.camera_source, str) and self.config.camera_source.isdigit():
                camera_source = int(self.config.camera_source)
            else:
                camera_source = self.config.camera_source
                
            self.cap = cv2.VideoCapture(camera_source)
            
            # Check if camera opened successfully
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
        """Process a single frame for object detection/segmentation"""
        try:
            frame_counts = Counter()
            special_detections = []
            
            # Get current model and its type
            model = self.model_manager.get_current_model()
            model_type = self.model_manager.get_current_model_type()
            
            if model is None:
                error_msg = "No model available for processing"
                logger.error(error_msg)
                self.visualizer.draw_error_message(frame, error_msg)
                return frame
            
            # Process frame based on model type
            if model_type == 'segmentation':
                results = model(frame, stream=True, conf=self.config.confidence_threshold)
            else:
                results = model(frame, stream=True)
            
            for r in results:
                if hasattr(r, 'boxes') and r.boxes is not None:
                    for box in r.boxes:
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        
                        if conf > self.config.confidence_threshold:
                            class_name = model.names[cls_id]
                            
                            # ENHANCED: Play class-specific alert
                            if cls_id in self.config.class_cooldowns and self.alert_manager.should_trigger_alert(cls_id):
                                self.alert_manager.play_alert(cls_id)  # Pass class_id for specific sound
                            
                            # Count the detection
                            frame_counts[class_name] += 1
                            
                            # Add to time window counter
                            self.time_window_counter.add_detection(class_name)
                            
                            # Check if this is a special class
                            if cls_id in self.config.special_classes:
                                special_detections.append((box, cls_id, conf, class_name))
                            else:
                                self.visualizer.draw_detection(frame, box, class_name, conf, cls_id)
                
                # Process masks for segmentation models
                if model_type == 'segmentation' and hasattr(r, 'masks') and r.masks is not None:
                    for i, mask in enumerate(r.masks):
                        if i < len(r.boxes):
                            box = r.boxes[i]
                            conf = float(box.conf[0])
                            cls_id = int(box.cls[0])
                            
                            if conf > self.config.confidence_threshold:
                                self.visualizer.draw_segmentation_mask(frame, mask.data[0], cls_id)
            
            # Process special class detections (conditional logic remains the same)
            special_class_ids = set(cls_id for _, cls_id, _, _ in special_detections)
            
            if special_class_ids == set(self.config.special_classes):
                special_boxes = [box for box, _, _, _ in special_detections]
                self.visualizer.draw_conditional_bounding_box(frame, special_boxes, "all_three")
            elif len(special_class_ids) == 2:
                special_boxes = [box for box, _, _, _, _ in special_detections]
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
            
            # NEW: Display audio status
            self.visualizer.draw_audio_status(frame, self.alert_manager)
            
            # Display connection status
            is_connected = self.cap.isOpened() and self.consecutive_failures < self.config.max_consecutive_failures
            self.visualizer.draw_connection_status(frame, is_connected, self.consecutive_failures)
            
            # Display model status
            self.visualizer.draw_model_status(
                frame, 
                self.model_manager.get_current_model_type(),
                self.model_manager.current_model_index,
                len(self.model_manager.models)
            )

            frame = cv2.resize(frame, (1024, 576))
            return frame
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            self.model_failures += 1
            
            if self.model_failures >= self.max_model_failures:
                logger.warning(f"Model failures exceeded threshold, attempting to switch model")
                if self.model_manager.switch_to_next_model():
                    self.model_failures = 0
                else:
                    logger.error("No alternative models available")
            
            error_msg = f"Processing Error: {str(e)[:50]}..."
            self.visualizer.draw_error_message(frame, error_msg)
            return frame 
 
    def run(self):
        """Main application loop with reconnection capability"""
        try:
            # Create a blank frame for error display
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
                cv2.imshow('YOLOv8 Live Detection', processed_frame)
                
                # Exit on 'q' key press, allow changing skip rate with number keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif ord('1') <= key <= ord('9'):  # Press 1-9 to change frame skip rate
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
        
        if elapsed >= 1.0:  # Update FPS every second
            self.current_fps = self.frame_count / elapsed
            self.frame_count = 0
            self.start_time = current_time
    
    def get_fps(self):
        """Get current FPS"""
        return self.current_fps

def main():
    """Main function to initialize and run the application"""
    try:
        # Configuration - change only these values to modify application behavior
        
        # For webcam (default)
        camera_source = 1
        
        # For RTSP stream (uncomment and modify as needed)
        #camera_source = "rtsp://admin:CemaraMas2025!@192.168.2.190:554/Streaming/Channels/501"
        
        # Try different camera indices if 0 doesn't work
        for camera_idx in [1, 2]:
            try:
                test_cap = cv2.VideoCapture(camera_idx)
                if test_cap.isOpened():
                    print(f"Found camera at index {camera_idx}")
                    test_cap.release()
                    camera_source = camera_idx
                    break
            except Exception as e:
                logger.warning(f"Error testing camera index {camera_idx}: {e}")
        
        # Define multiple models with fallback - first one is preferred
        model_configs = [
            Path(r"D:\RaihanFarid\Dokumen\Object Detection\CV_model\HBDetect1.torchscript"),  # Primary detection model
            #Path(r"D:\RaihanFarid\Dokumen\Object Detection\CV_model\segmentation_model.pt"),  # Fallback segmentation model
            #Path(r"D:\RaihanFarid\Dokumen\Object Detection\CV_model\backup_detection_model.pt"),  # Backup detection model
        ]

                # Enhanced audio configuration
        class_sound_config = {
            0: Path(r"D:\RaihanFarid\Dokumen\Object Detection\usedAudio\new-notification-09-352705.mp3"),
            1: Path(r"D:\RaihanFarid\Dokumen\Object Detection\usedAudio\new-notification-026-380249.mp3"),
            #2: Path(r"D:\RaihanFarid\Dokumen\Object Detection\usedAudio\animal_alert.mp3"),
            # Add more class-sound mappings as needed
        }
        
        config = AppConfig(
            model_paths=model_configs,
            camera_source=camera_source,  # Use 0 for webcam or RTSP URL for stream
            frame_width=640,
            # Enhanced audio configuration
            class_sound_map=class_sound_config,
            #default_alert_sound=Path(r"D:\RaihanFarid\Dokumen\Object Detection\usedAudio\default_alert.mp3"),            
            frame_height=480,
            confidence_threshold=0.5,
            class_color_map={1: (0, 0, 255), 0: (0, 255, 0)},  # Red, Green
            #alert_sound_path=Path(r"D:\RaihanFarid\Dokumen\Object Detection\usedAudio\notification-alert-269289.mp3"),
            alert_cooldown=5,
            class_cooldowns={0: 10, 1: 5, 2: 15},
            counter_time_window=10,  # 10-second time window for counter
            reconnect_delay=5,  # Seconds between reconnection attempts
            max_consecutive_failures=3,  # Maximum failures before attempting reconnection
            special_classes=(0, 1, 2),  # Classes to check for conditional labeling
            conditional_box_colors={
                "all_three": (0, 255, 255),  # Yellow for all three classes
                "two_classes": (255, 0, 255)  # Magenta for two classes
            },
            model_load_timeout=30,  # Seconds to wait for model to load
            model_retry_interval=5,  # Seconds between model loading retries
            # New frame skipping configuration
            frame_skip=1,  # Process every 2nd frame (2x performance boost)
            display_every_frame=True,  # Show detections on all frames
            target_fps=30  # Target FPS for display
        )
        
        # Create and run the detector
        detector = ObjectDetector(config)
        detector.run()
        
    except Exception as e:
        logger.critical(f"Failed to start application: {e}")
        print(f"Application failed to start: {e}")
        print("Check object_detection.log for details")

if __name__ == "__main__":
    main()