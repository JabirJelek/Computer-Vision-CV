import cv2
from ultralytics import YOLO
from pathlib import Path
import pygame
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Dict, Tuple, Any, Union, List, Optional, Set
import numpy as np
from abc import ABC, abstractmethod
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AppConfig:
    """Centralized configuration for the application"""
    # Model configuration
    model_path: Path = Path(r"D:\RaihanFarid\Dokumen\Object Detection\CV_model\Segmentation1.torchscript")
    
    # Video source configuration
    camera_source: Union[int, str] = 0
    frame_width: int = 640
    frame_height: int = 480
    
    # Detection configuration
    confidence_threshold: float = 0.5
    class_color_map: Dict[int, Tuple[int, int, int]] = field(default_factory=dict)
    
    # Alert configuration
    alert_sound_path: Path = Path(r"D:\RaihanFarid\Dokumen\Object Detection\usedAudio\notification-alert-269289.mp3")
    combination_alert_sounds: Dict[str, Path] = field(default_factory=dict)
    alert_cooldown: int = 5
    combination_cooldowns: Dict[str, int] = field(default_factory=dict)
    
    # Sequential suppression configuration
    sequential_suppression_time: int = 30  # Seconds to suppress after sequence detection
    suppression_cooldowns: Dict[str, int] = field(default_factory=dict)  # Individual suppression times

    # Detection delay configuration
    detection_delay_time: float = 2.0 # Seconds to wait before confirming combination
    minimum_detection_frames: int = 5 # Minimum frames combination must be detected
    
    # Counter configuration
    counter_time_window: int = 10
    
    # RTSP configuration
    reconnect_delay: int = 5
    max_consecutive_failures: int = 3
    
    # Conditional labeling configuration
    special_classes: Tuple[int, ...] = (0, 1, 2)
    conditional_box_colors: Dict[str, Tuple[int, int, int]] = field(default_factory=dict)
    
    # Segmentation configuration
    mask_alpha: float = 0.5
    draw_masks: bool = True
    draw_boxes: bool = True
    
    # Display configuration
    display_width: int = 1024
    display_height: int = 576
    
    def __post_init__(self):
        # Initialize default values
        if not self.class_color_map:
            self.class_color_map = {0: (0, 0, 255), 1: (0, 255, 0)}
            
        if not self.conditional_box_colors:
            self.conditional_box_colors = {
                "all_three": (0, 255, 255),
                "two_classes": (255, 0, 255),
                "sequential_complete": (0, 255, 0),
                "suppressed": (128, 128, 128),
                "pending": (255, 165, 0),
            }

        # Initialize combination alert sounds
        if not self.combination_alert_sounds:
            self.combination_alert_sounds = {
                "class_0_1": Path(r"D:\RaihanFarid\Dokumen\Object Detection\usedAudio\01_combination.mp3"),
                "class_0_2": Path(r"D:\RaihanFarid\Dokumen\Object Detection\usedAudio\02_combination.mp3"),
                "class_1_2": Path(r"D:\RaihanFarid\Dokumen\Object Detection\usedAudio\12_combination.mp3"),
                "all_three": Path(r"D:\RaihanFarid\Dokumen\Object Detection\usedAudio\level-up-07-383747.mp3"),
            }
        
        if not self.combination_cooldowns:
            self.combination_cooldowns = {
                "class_0_1": 10,
                "class_0_2": 10, 
                "class_1_2": 10,
                "all_three": 30,
            }        

        # Initialize suppression cooldowns
        if not self.suppression_cooldowns:
            self.suppression_cooldowns = {
                "class_0_1": 25,  
                "class_0_2": 25,
                "class_1_2": 25,
                "all_three": 35,
            }        

class TemporalMemory:
    """Stores class detection states with time-based expiration"""
    
    def __init__(self, time_window: float = 5.0):
        self.time_window = time_window
        self.class_states = {}  # class_id -> last_detection_time
        self.class_presence = {}  # class_id -> is_active_until
    
    def update_detection(self, class_id: int, current_time: float):
        """Update detection time for a class"""
        self.class_states[class_id] = current_time
        self.class_presence[class_id] = current_time + self.time_window
    
    def is_class_active(self, class_id: int, current_time: float) -> bool:
        """Check if class was detected within time window"""
        if class_id not in self.class_presence:
            return False
        return current_time <= self.class_presence[class_id]
    
    def get_active_classes(self, current_time: float) -> Set[int]:
        """Get all classes active within time window"""
        self._cleanup_expired(current_time)
        return set(self.class_presence.keys())
    
    def _cleanup_expired(self, current_time: float):
        """Remove expired class states"""
        expired = [cls_id for cls_id, expiry in self.class_presence.items() 
                  if current_time > expiry]
        for cls_id in expired:
            self.class_presence.pop(cls_id, None)
            self.class_states.pop(cls_id, None)

class DetectionBuffer:
    """Buffers detections to prevent instant audio alerts"""
    
    def __init__(self, delay_time: float = 2.0, min_frames: int = 5):
        self.delay_time = delay_time
        self.min_frames = min_frames
        self.buffered_combinations = {}  # combination_name -> {start_time, frame_count, confirmed}
        self.confirmed_combinations = set()  # Fixed typo: was confirmed_comninations

    def update_combination(self, combination_name: str, is_present: bool, current_time: float) -> bool:
        """Update combination status and return True if confirmed"""
        if is_present:
            if combination_name not in self.buffered_combinations:
                # Start tracking new combination
                self.buffered_combinations[combination_name] = {
                    'start_time': current_time,
                    'frame_count': 1,
                    'confirmed': False
                }
                logger.info(f"Started buffering combination: {combination_name}")
            else:
                # Update existing combination
                buffer_data = self.buffered_combinations[combination_name]
                buffer_data['frame_count'] += 1
                
                # Check if combination should be confirmed
                time_elapsed = current_time - buffer_data['start_time']
                if (time_elapsed >= self.delay_time and 
                    buffer_data['frame_count'] >= self.min_frames and
                    not buffer_data['confirmed']):
                    
                    buffer_data['confirmed'] = True
                    self.confirmed_combinations.add(combination_name)
                    logger.info(f"Combination confirmed: {combination_name} "
                               f"(time: {time_elapsed:.1f}s, frames: {buffer_data['frame_count']})")
                    return True
        else:
            # Combination disappeared, remove from buffer
            if combination_name in self.buffered_combinations:
                if not self.buffered_combinations[combination_name]['confirmed']:
                    logger.info(f"Combination disappeared before confirmation: {combination_name}")
                del self.buffered_combinations[combination_name]
                self.confirmed_combinations.discard(combination_name)
        
        return False

    def is_combination_confirmed(self, combination_name: str) -> bool:
        """Check if combination is confirmed"""
        return combination_name in self.confirmed_combinations
    
    def get_pending_combinations(self) -> List[str]:
        """Get list of combinations that are being buffered but not confirmed"""
        return [name for name, data in self.buffered_combinations.items() 
                if not data['confirmed']]
    
    def get_buffer_progress(self, combination_name: str, current_time: float) -> Tuple[float, int]:
        """Get progress of buffering (0.0 to 1.0) and frame count"""
        if combination_name not in self.buffered_combinations:
            return 0.0, 0
        
        data = self.buffered_combinations[combination_name]
        time_progress = min(1.0, (current_time - data['start_time']) / self.delay_time)
        frame_progress = min(1.0, data['frame_count'] / self.min_frames)
        
        # Use the slower progressing metric
        progress = min(time_progress, frame_progress)
        return progress, data['frame_count']
    
    def cleanup_old_buffers(self, current_time: float):
        """Clean up buffers that are too old"""
        expired = []
        for name, data in self.buffered_combinations.items():
            if current_time - data['start_time'] > self.delay_time * 2:  # 2x delay time
                expired.append(name)
        
        for name in expired:
            del self.buffered_combinations[name]
            self.confirmed_combinations.discard(name)

class EnhancedSequentialLogic:
    def __init__(self, config: AppConfig, temporal_memory: TemporalMemory):
        self.config = config
        self.temporal_memory = temporal_memory
    
    def check_enhanced_sequence(self, current_frame_classes: Set[int], 
                              current_time: float) -> Dict[str, bool]:
        """Check various sequence conditions using temporal memory"""
        
        # Get all classes active in time window (current + recent)
        all_active_classes = self.temporal_memory.get_active_classes(current_time)
        
        conditions = {
            "all_three_current": current_frame_classes == set(self.config.special_classes),
            "all_three_temporal": all_active_classes == set(self.config.special_classes),
            "sequential_complete": self._check_sequential_complete(all_active_classes),
            "any_two_temporal": len(all_active_classes) >= 2
        }
        
        return conditions
    
    def _check_sequential_complete(self, active_classes: Set[int]) -> bool:
        """Check if required sequence is present in temporal memory"""
        return active_classes.issuperset(set(self.config.special_classes))

class IVideoSource(ABC):
    """Abstract interface for video sources"""
    
    @abstractmethod
    def is_opened(self) -> bool:
        pass
    
    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        pass
    
    @abstractmethod
    def release(self):
        pass
    
    @abstractmethod
    def reconnect(self) -> bool:
        pass

class CameraVideoSource(IVideoSource):
    """Video source for camera input"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.cap = None
        self.setup_camera()
    
    def setup_camera(self) -> bool:
        """Initialize camera source"""
        try:
            camera_source = (int(self.config.camera_source) 
                           if isinstance(self.config.camera_source, str) and self.config.camera_source.isdigit()
                           else self.config.camera_source)
            
            self.cap = cv2.VideoCapture(camera_source)
            
            if self.config.frame_width > 0:
                self.cap.set(3, self.config.frame_width)
            if self.config.frame_height > 0:
                self.cap.set(4, self.config.frame_height)
            
            return self.cap.isOpened()
        except Exception as e:
            logger.error(f"Camera setup error: {e}")
            return False
    
    def is_opened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self.is_opened():
            return False, None
        return self.cap.read()
    
    def release(self):
        if self.cap is not None:
            self.cap.release()
    
    def reconnect(self) -> bool:
        self.release()
        return self.setup_camera()

class RTSPVideoSource(IVideoSource):
    """Video source for RTSP streams"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.cap = None
        self.setup_rtsp()
    
    def setup_rtsp(self) -> bool:
        """Initialize RTSP source"""
        try:
            self.cap = cv2.VideoCapture(self.config.camera_source)
            
            # Optimize for RTSP
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            if self.config.frame_width > 0:
                self.cap.set(3, self.config.frame_width)
            if self.config.frame_height > 0:
                self.cap.set(4, self.config.frame_height)
            
            # Test connection
            if not self.cap.isOpened():
                return False
                
            ret, frame = self.cap.read()
            return ret
        except Exception as e:
            logger.error(f"RTSP setup error: {e}")
            return False
    
    def is_opened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self.is_opened():
            return False, None
        return self.cap.read()
    
    def release(self):
        if self.cap is not None:
            self.cap.release()
    
    def reconnect(self) -> bool:
        self.release()
        time.sleep(self.config.reconnect_delay)
        return self.setup_rtsp()

class VideoSourceFactory:
    """Factory for creating appropriate video sources"""
    
    @staticmethod
    def create_video_source(config: AppConfig) -> IVideoSource:
        """Create video source based on configuration"""
        if isinstance(config.camera_source, str) and "rtsp" in config.camera_source.lower():
            return RTSPVideoSource(config)
        else:
            return CameraVideoSource(config)

class DetectionResult:
    """Data class to store detection results"""
    
    def __init__(self, box, mask, class_id: int, confidence: float, class_name: str):
        self.box = box
        self.mask = mask
        self.class_id = class_id
        self.confidence = confidence
        self.class_name = class_name

class IModel(ABC):
    """Abstract interface for detection models"""
    
    @abstractmethod
    def predict(self, frame: np.ndarray) -> List[DetectionResult]:
        pass
    
    @abstractmethod
    def get_class_name(self, class_id: int) -> str:
        pass

class YOLOModel(IModel):
    """YOLO segmentation model implementation"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.model = self._load_model()
    
    def _load_model(self) -> YOLO:
        """Load the YOLO model"""
        try:
            model = YOLO(str(self.config.model_path), task='segment')
            logger.info("Model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def predict(self, frame: np.ndarray) -> List[DetectionResult]:
        """Run inference on a frame"""
        results = []
        
        try:
            model_results = self.model(frame, stream=True, verbose=False)
            
            for r in model_results:
                if r.boxes is not None and len(r.boxes) > 0:
                    for i, box in enumerate(r.boxes):
                        conf = float(box.conf[0])
                        
                        if conf > self.config.confidence_threshold:
                            cls_id = int(box.cls[0])
                            class_name = self.get_class_name(cls_id)
                            
                            # Get corresponding mask
                            mask = None
                            if r.masks is not None and i < len(r.masks):
                                mask = r.masks[i].data[0].cpu().numpy()
                                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                            
                            results.append(DetectionResult(box, mask, cls_id, conf, class_name))
        except Exception as e:
            logger.error(f"Prediction error: {e}")
        
        return results
    
    def get_class_name(self, class_id: int) -> str:
        """Get class name from model"""
        return self.model.names.get(class_id, f"Class_{class_id}")

class SequentialMemory:
    """Stores detection sequences across multiple frames"""
    
    def __init__(self, sequence_timeout: float = 15.0):
        self.sequence_timeout = sequence_timeout
        self.detection_sequences = {}
        self.current_sequence_id = 0
        self.suppression_timers = {}  # Track suppression for each combination
        self.sequence_start_time = None  # Track when current sequence started        
    
    def add_detection(self, class_id: int) -> int:
        """Add detection to sequence"""
        current_time = time.time()
        active_sequence = self._get_active_sequence()
        
        if active_sequence is None:
            self.sequence_start_time = current_time
        
        if active_sequence and class_id not in active_sequence:
            active_sequence[class_id] = current_time
            return self.current_sequence_id
        else:
            self.current_sequence_id += 1
            self.detection_sequences[self.current_sequence_id] = {class_id: current_time}
            self.sequence_start_time = current_time
            return self.current_sequence_id 

    def _get_active_sequence(self) -> Optional[Dict[int, float]]:
        """Get most recent active sequence"""
        current_time = time.time()
        
        # Clean expired sequences
        expired = [
            seq_id for seq_id, seq_data in self.detection_sequences.items()
            if current_time - max(seq_data.values()) > self.sequence_timeout
        ]
        for seq_id in expired:
            del self.detection_sequences[seq_id]
        
        return self.detection_sequences.get(self.current_sequence_id)
    
    def check_sequence_complete(self, required_classes: Tuple[int, ...]) -> bool:
        """Check if sequence contains all required classes"""
        active_sequence = self._get_active_sequence()
        return active_sequence and all(cls_id in active_sequence for cls_id in required_classes)
    
    def get_sequence_classes(self) -> Set[int]:
        """Get classes in current sequence"""
        active_sequence = self._get_active_sequence()
        return set(active_sequence.keys()) if active_sequence else set()
    
    def get_sequence_duration(self) -> float:
        """Get how long the current sequence has been active"""
        if self.sequence_start_time is None:
            return 0.0
        return time.time() - self.sequence_start_time
    
    def is_combination_suppressed(self, combination_name: str, suppression_time: float) -> bool:
        """Check if a combination should be suppressed based on recent sequence activity"""
        current_time = time.time()
        
        if combination_name in self.suppression_timers:
            time_since_suppression = current_time - self.suppression_timers[combination_name]
            if time_since_suppression < suppression_time:
                return True
        
        return False
    
    def suppress_combination(self, combination_name: str):
        """Mark a combination as suppressed"""
        self.suppression_timers[combination_name] = time.time()
    
    def cleanup_suppressions(self, current_time: float):
        """Clean up old suppression records"""
        expired_suppressions = [
            combo for combo, suppress_time in self.suppression_timers.items()
            if current_time - suppress_time > 60
        ]
        for combo in expired_suppressions:
            del self.suppression_timers[combo]

class AlertManager:
    """Manages conditional audio alerts based on class combinations"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.alert_timers = {}  # Tracks combination alerts
        self.combination_sounds = {}  # Stores loaded sound objects for combinations
        self._init_audio()
    
    def _init_audio(self):
        """Initialize audio system with combination sounds"""
        try:
            pygame.mixer.init()
            
            # Load combination alert sounds
            for combo_name, sound_path in self.config.combination_alert_sounds.items():
                if sound_path.exists():
                    self.combination_sounds[combo_name] = pygame.mixer.Sound(str(sound_path))
                else:
                    logger.warning(f"Alert sound not found for {combo_name}: {sound_path}")
                    
        except Exception as e:
            logger.error(f"Audio initialization error: {e}")
            self.combination_sounds = {}
    
    def should_trigger_combination_alert(self, combination_name: str, 
                                       sequential_memory: SequentialMemory,
                                       detection_buffer: DetectionBuffer) -> bool:  # Fixed: Added detection_buffer parameter
        """Check if combination alert should be triggered with suppression logic"""
        current_time = time.time()
        cooldown = self.config.combination_cooldowns.get(combination_name, self.config.alert_cooldown)
        
        # Check if this combination is suppressed due to recent sequence activity
        suppression_time = self.config.suppression_cooldowns.get(combination_name, 
                                                               self.config.sequential_suppression_time)
        
        if sequential_memory.is_combination_suppressed(combination_name, suppression_time):
            logger.info(f"Combination {combination_name} suppressed due to recent sequence activity")
            return False
        
        # Check if combination is confirmed by detection buffer
        if not detection_buffer.is_combination_confirmed(combination_name):
            logger.info(f"Combination {combination_name} waiting for buffer confirmation")
            return False
        
        # Original cooldown check
        if (combination_name not in self.alert_timers or 
            (current_time - self.alert_timers[combination_name]) >= cooldown):
            self.alert_timers[combination_name] = current_time
            return True
        return False
    
    def play_combination_alert(self, combination_name: str, sequential_memory: SequentialMemory):
        """Play alert sound and mark combination as suppressed"""
        sound = self.combination_sounds.get(combination_name)
        if sound:
            # Suppress this combination to prevent immediate re-triggering
            sequential_memory.suppress_combination(combination_name)
            
            threading.Thread(target=sound.play, daemon=True).start()
            logger.info(f"Playing alert for combination: {combination_name} (now suppressed)")
        else:
            logger.warning(f"No sound configured for combination: {combination_name}")

class TimeWindowCounter:
    """Manages time-based counter with rolling window"""
    
    def __init__(self, time_window_seconds: int = 10):
        self.time_window = time_window_seconds
        self.detection_history = deque()
        self.current_counts = Counter()
    
    def add_detection(self, class_name: str):
        """Add detection to history"""
        current_time = time.time()
        self.detection_history.append((current_time, class_name))
        self.current_counts[class_name] += 1
        self._prune_old_detections(current_time)
    
    def _prune_old_detections(self, current_time: float):
        """Remove old detections"""
        while (self.detection_history and 
               current_time - self.detection_history[0][0] > self.time_window):
            old_time, old_class = self.detection_history.popleft()
            self.current_counts[old_class] -= 3
            
            if self.current_counts[old_class] <= 0:
                del self.current_counts[old_class]
    
    def get_counts(self) -> Counter:
        """Get current counts"""
        self._prune_old_detections(time.time())
        return self.current_counts.copy()
    
    def get_time_remaining(self) -> float:
        """Get time until next counter update"""
        if not self.detection_history:
            return self.time_window
        
        current_time = time.time()
        oldest_time = self.detection_history[0][0]
        return max(0, self.time_window - (current_time - oldest_time))

class DetectionVisualizer:
    """Handles visualization of detection segmentation results"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.line_height = 22
        self.padding = 8
    
    def draw_mask(self, frame: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int]):
        """Draw segmentation mask"""
        colored_mask = np.zeros_like(frame)
        colored_mask[:] = color
        
        mask_bool = mask.astype(bool)
        frame[mask_bool] = cv2.addWeighted(
            frame[mask_bool], 1 - self.config.mask_alpha,
            colored_mask[mask_bool], self.config.mask_alpha, 0
        ) 
 
    def draw_detection(self, frame: np.ndarray, detection: DetectionResult, 
                      is_conditional: bool = False, conditional_type: Optional[str] = None):
        """Draw single detection"""
        x1, y1, x2, y2 = map(int, detection.box.xyxy[0])
        
        # Choose color
        if is_conditional:
            color = self.config.conditional_box_colors.get(conditional_type, (255, 255, 255))
        else:
            color = self.config.class_color_map.get(detection.class_id, (255, 255, 255))
        
        # Draw mask
        if detection.mask is not None and self.config.draw_masks:
            self.draw_mask(frame, detection.mask, color)
        
        # Draw bounding box
        if self.config.draw_boxes:
            thickness = 4 if is_conditional else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            if not is_conditional:
                self._draw_label(frame, x1, y1, detection.class_name, detection.confidence, color)  
 
    def _draw_label(self, frame: np.ndarray, x: int, y: int, class_name: str, 
                   confidence: float, color: Tuple[int, int, int]):
        """Draw label for detection"""
        label = f"{class_name} {confidence:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        
        # Label background
        cv2.rectangle(frame, (x, y - text_height - 10), 
                     (x + text_width, y), color, -1)
        
        # Label text
        cv2.putText(frame, label, (x, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    def draw_combined_detection(self, frame: np.ndarray, detections: List[DetectionResult], 
                               conditional_type: str, buffer_progress: float = 1.0):
        """Draw combined bounding box with buffering progress indicator"""
        if not detections:
            return
        
        # Calculate combined bounding box
        boxes = [detection.box for detection in detections]
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
        
        # Adjust color based on buffer progress
        base_color = self.config.conditional_box_colors.get(conditional_type, (255, 255, 255))
        if buffer_progress < 1.0:
            # Blend towards orange for pending detections
            pending_color = self.config.conditional_box_colors.get("pending", (255, 165, 0))
            color = tuple(int(base_color[i] * buffer_progress + pending_color[i] * (1 - buffer_progress)) 
                         for i in range(3))
        else:
            color = base_color
        
        labels = {
            "all_three": "Item Completed",
            "two_classes": "TWO SPECIAL CLASSES", 
            "sequential_complete": "SEQUENCE COMPLETE",
            "class_0_1": "Missing Menu",
            "class_0_2": "Missing Rice", 
            "class_1_2": "Missing Fork",
            "suppressed": "SUPPRESSED",
            "pending": "PENDING CONFIRMATION"
        }
        label = labels.get(conditional_type, "COMBINED DETECTION")
        
        # Add progress indicator to label
        if buffer_progress < 1.0:
            label += f" [{buffer_progress*100:.0f}%]"
        
        if self.config.draw_boxes:
            # Draw box
            thickness = 4 if buffer_progress == 1.0 else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 3)
            cv2.rectangle(frame, (x1, y1 - text_height - 15),
                         (x1 + text_width + 10, y1), color, -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3)
        
        # Draw combined masks
        if self.config.draw_masks:
            combined_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=bool)
            for detection in detections:
                if detection.mask is not None:
                    combined_mask = np.logical_or(combined_mask, detection.mask.astype(bool))
            self.draw_mask(frame, combined_mask, color)  
 
    def draw_info_panel(self, frame: np.ndarray, frame_counts: Counter, 
                       time_window_counter: TimeWindowCounter, alert_manager: AlertManager,
                       sequential_memory: SequentialMemory, detection_buffer: DetectionBuffer,
                       is_connected: bool, failure_count: int):
        """Draw all information panels including buffering progress"""
        # Resize frame for display
        frame = cv2.resize(frame, (self.config.display_width, self.config.display_height))
        
        # Draw counter panel
        self._draw_counter_panel(frame, frame_counts, time_window_counter)
        
        # Draw cooldown status
        self._draw_cooldown_status(frame, alert_manager)
        
        # Draw buffering progress
        self._draw_buffering_status(frame, detection_buffer)
        
        # Draw connection status
        self._draw_connection_status(frame, is_connected, failure_count)
        
        # Draw sequential memory status
        self._draw_sequential_status(frame, sequential_memory)
        
        return frame 
   
    def _draw_counter_panel(self, frame: np.ndarray, frame_counts: Counter, 
                           time_window_counter: TimeWindowCounter):
        """Draw counter information panel"""
        lines = ["Counts (frame):"]
        if frame_counts:
            for name, cnt in frame_counts.items():
                lines.append(f"{name}: {cnt}")
        else:
            lines.append("None")
            
        lines.append("")
        lines.append(f"Last {time_window_counter.time_window}s:")
        time_counts = time_window_counter.get_counts()
        if time_counts:
            for name, cnt in time_counts.items():
                lines.append(f"{name}: {cnt}")
        else:
            lines.append("None")
            
        lines.append(f"Reset in: {time_window_counter.get_time_remaining():.1f}s")
        
        self._draw_text_panel(frame, lines, 10, 30, 250, right_align=True)  
 
    def _draw_cooldown_status(self, frame: np.ndarray, alert_manager: AlertManager):
        """Draw alert cooldown status"""
        current_time = time.time()
        for i, (cls_id, last_alert) in enumerate(alert_manager.alert_timers.items()):
            time_remaining = max(0, alert_manager.config.alert_cooldown - (current_time - last_alert))
            status_text = f"Class {cls_id} alert: {time_remaining:.1f}s cooldown"
            cv2.putText(frame, status_text, (10, 30 + i*30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    def _draw_buffering_status(self, frame: np.ndarray, detection_buffer: DetectionBuffer):
        """Draw detection buffering progress"""
        current_time = time.time()
        pending_combos = detection_buffer.get_pending_combinations()
        
        if pending_combos:
            cv2.putText(frame, "Buffering Combinations:", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
            
            for i, combo_name in enumerate(pending_combos):
                progress, frames = detection_buffer.get_buffer_progress(combo_name, current_time)
                status_text = f"{combo_name}: {progress*100:.0f}% ({frames} frames)"
                cv2.putText(frame, status_text, (10, 180 + i*25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
    
    def _draw_connection_status(self, frame: np.ndarray, is_connected: bool, failure_count: int):
        """Draw connection status"""
        status_text = "Connected" if is_connected else f"Disconnected ({failure_count})"
        color = (0, 255, 0) if is_connected else (0, 0, 255)
        cv2.putText(frame, f"Status: {status_text}", (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
 
    def _draw_sequential_status(self, frame: np.ndarray, sequential_memory: SequentialMemory):
        """Draw sequential detection status"""
        memory_classes = sequential_memory.get_sequence_classes()
        memory_text = f"Memory: {sorted(memory_classes)}"
        cv2.putText(frame, memory_text, (10, frame.shape[0] - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _draw_text_panel(self, frame: np.ndarray, lines: List[str], x0: int, y0: int, 
                        panel_w: int, right_align: bool = False):
        """Draw semi-transparent text panel"""
        if right_align:
            x0 = frame.shape[1] - panel_w - x0
        
        panel_h = self.line_height * len(lines) + self.padding
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0 - self.padding//2, y0 - self.line_height),
                     (x0 + panel_w, y0 - self.line_height + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Draw text
        y = y0
        for line in lines:
            cv2.putText(frame, line, (x0, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            y += self.line_height

    def draw_error_message(self, frame: np.ndarray, message: str):
        """Display error message"""
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        text_lines = message.split('\n')
        y_pos = frame.shape[0] // 2 - (len(text_lines) * 30) // 2
        
        for line in text_lines:
            text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            x_pos = (frame.shape[1] - text_size[0]) // 2
            cv2.putText(frame, line, (x_pos, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            y_pos += 40

class DetectionProcessor:
    """Processes detections and applies combination-based alert logic"""
    
    def __init__(self, config: AppConfig, model: IModel, sequential_memory: SequentialMemory, 
                 temporal_memory: TemporalMemory, detection_buffer: DetectionBuffer):
        self.config = config
        self.model = model
        self.sequential_memory = sequential_memory
        self.temporal_memory = temporal_memory
        self.detection_buffer = detection_buffer
    
    def process_detections(self, frame: np.ndarray, alert_manager: AlertManager, 
                          time_window_counter: TimeWindowCounter) -> Tuple[Counter, List[DetectionResult], Set[int]]:
        """Process frame with buffering and suppression logic"""
        frame_counts = Counter()
        special_detections = []
        current_frame_classes = set()
        current_time = time.time()
        
        # Clean up old buffers
        self.detection_buffer.cleanup_old_buffers(current_time)
        self.sequential_memory.cleanup_suppressions(current_time)
        
        # Run model inference
        detections = self.model.predict(frame)
        
        for detection in detections:
            # Update counters
            frame_counts[detection.class_name] += 1
            time_window_counter.add_detection(detection.class_name)
            current_frame_classes.add(detection.class_id)
            
            # Update temporal memory
            self.temporal_memory.update_detection(detection.class_id, current_time)
            
            # Track special classes
            if detection.class_id in self.config.special_classes:
                special_detections.append(detection)
                self.sequential_memory.add_detection(detection.class_id)
        
        # Apply combination-based alert logic with buffering - USING TEMPORAL MEMORY
        self._check_combination_alerts(current_time, alert_manager)
        
        return frame_counts, special_detections, current_frame_classes
    
    def _check_combination_alerts(self, current_time: float, alert_manager: AlertManager):
        """Check for specific class combinations using TEMPORAL MEMORY instead of current frame"""
        # Get all classes active in time window (from temporal memory)
        active_classes = self.temporal_memory.get_active_classes(current_time)
        active_special = active_classes.intersection(set(self.config.special_classes))
        
        # Check combinations based on temporal memory
        combinations_to_check = [
            ({0, 1}, "class_0_1"),
            ({0, 2}, "class_0_2"), 
            ({1, 2}, "class_1_2"),
            ({0, 1, 2}, "all_three")
        ]
        
        for required_classes, alert_name in combinations_to_check:
            # Check if combination is present in TEMPORAL MEMORY
            is_present = active_special == required_classes
            
            # Update detection buffer with temporal memory state
            buffer_confirmed = self.detection_buffer.update_combination(
                alert_name, is_present, current_time
            )
            
            # Trigger alert if buffer confirms and other conditions met
            if (is_present and buffer_confirmed and
                alert_manager.should_trigger_combination_alert(
                    alert_name, self.sequential_memory, self.detection_buffer)):
                
                alert_manager.play_combination_alert(alert_name, self.sequential_memory)
                logger.info(f"Temporal memory combination alert: {alert_name} confirmed")
    

class EnhancedConditionalLogicProcessor:
    """Handles conditional detection logic with combination-based alerts"""
    
    def __init__(self, config: AppConfig, sequential_memory: SequentialMemory, 
                 temporal_memory: TemporalMemory, detection_buffer: DetectionBuffer):
        self.config = config
        self.sequential_memory = sequential_memory
        self.temporal_memory = temporal_memory
        self.detection_buffer = detection_buffer
    
    def apply_conditional_logic(self, frame: np.ndarray, special_detections: List[DetectionResult],
                              current_frame_classes: Set[int], visualizer: DetectionVisualizer, 
                              alert_manager: AlertManager, current_time: float) -> bool:
        """Apply conditional logic using TEMPORAL MEMORY for combination detection"""
        
        # Get all classes active in time window (from temporal memory)
        all_active_classes = self.temporal_memory.get_active_classes(current_time)
        active_special = all_active_classes.intersection(set(self.config.special_classes))
        
        # Check if combinations are suppressed
        is_suppressed = self._check_suppression_status()
        
        # Get buffering progress for visualization based on temporal memory
        buffer_progress = self._get_combination_buffer_progress(active_special, current_time)
        
        # Check various conditions for visualization using TEMPORAL MEMORY
        if all_active_classes == set(self.config.special_classes):
            combo_type = "suppressed" if is_suppressed else "all_three"
            visualizer.draw_combined_detection(frame, special_detections, combo_type, buffer_progress)
            return True
            
        elif len(active_special) == 2:
            # Determine which specific combination for visualization
            combo_type = self._get_combination_type(active_special)
            if is_suppressed:
                combo_type = "suppressed"
            visualizer.draw_combined_detection(frame, special_detections, combo_type, buffer_progress)
            return True
            
        elif self._check_sequential_complete(active_special, current_time):
            combo_type = "suppressed" if is_suppressed else "sequential_complete"
            visualizer.draw_combined_detection(frame, special_detections, combo_type, buffer_progress)
            return True
        else:
            # Draw individual detections
            for detection in special_detections:
                visualizer.draw_detection(frame, detection)
            return False
    
    def _get_combination_type(self, active_special: Set[int]) -> str:
        """Get the specific combination type"""
        if active_special == {0, 1}:
            return "class_0_1"
        elif active_special == {0, 2}:
            return "class_0_2"
        elif active_special == {1, 2}:
            return "class_1_2"
        else:
            return "two_classes"  
    
    def _get_combination_buffer_progress(self, active_special: Set[int], current_time: float) -> float:
        """Get buffering progress for the current combination in temporal memory"""
        if len(active_special) < 2:
            return 1.0  # No combination to buffer
            
        combo_type = self._get_combination_type(active_special)
        if combo_type != "two_classes":  # Only for specific combinations
            progress, _ = self.detection_buffer.get_buffer_progress(combo_type, current_time)
            return progress
        return 1.0  # Default to confirmed for generic combinations
    
    def _check_suppression_status(self) -> bool:
        """Check if any relevant combinations are currently suppressed"""
        current_time = time.time()
        combinations_to_check = ["class_0_1", "class_0_2", "class_1_2", "all_three"]
        
        for combo_name in combinations_to_check:
            suppression_time = self.config.suppression_cooldowns.get(combo_name, 
                                                                   self.config.sequential_suppression_time)
            if self.sequential_memory.is_combination_suppressed(combo_name, suppression_time):
                return True
        return False
    
    def _check_sequential_complete(self, active_classes: Set[int], current_time: float) -> bool:
        """Enhanced sequential check using temporal memory"""
        return active_classes.issuperset(set(self.config.special_classes))
class ObjectDetectorApp:
    """Main application class"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.video_source = VideoSourceFactory.create_video_source(config)
        self.model = YOLOModel(config)
        self.sequential_memory = SequentialMemory()
        self.temporal_memory = TemporalMemory(time_window=10.0)  
        self.detection_buffer = DetectionBuffer(
            delay_time=config.detection_delay_time,
            min_frames=config.minimum_detection_frames
        )
        self.alert_manager = AlertManager(config)
        self.visualizer = DetectionVisualizer(config)
        self.time_window_counter = TimeWindowCounter(config.counter_time_window)
        self.detection_processor = DetectionProcessor(
            config, self.model, self.sequential_memory, self.temporal_memory, self.detection_buffer)
        self.conditional_processor = EnhancedConditionalLogicProcessor(
            config, self.sequential_memory, self.temporal_memory, self.detection_buffer)

        self.consecutive_failures = 0
        self.is_running = False
    
    def run(self):
        """Main application loop"""
        self.is_running = True
        blank_frame = None
        
        try:
            while self.is_running:
                if not self._ensure_connection():
                    continue
                
                ret, frame = self.video_source.read()
                if not ret:
                    self._handle_frame_read_error(blank_frame)
                    continue
                
                self._process_successful_frame(frame)
                
                # Check for exit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            self.cleanup()
    
    def _ensure_connection(self) -> bool:
        """Ensure video source is connected"""
        if (not self.video_source.is_opened() or 
            self.consecutive_failures >= self.config.max_consecutive_failures):
            
            logger.info(f"Attempting to reconnect... (Failures: {self.consecutive_failures})")
            if self.video_source.reconnect():
                self.consecutive_failures = 0
                logger.info("Reconnection successful")
            else:
                logger.error("Reconnection failed")
                time.sleep(self.config.reconnect_delay)
                return False
        return True
    
    def _handle_frame_read_error(self, blank_frame: Optional[np.ndarray]):
        """Handle frame read errors"""
        self.consecutive_failures += 1
        logger.warning(f"Frame read failed ({self.consecutive_failures}/{self.config.max_consecutive_failures})")
        
        if blank_frame is None:
            blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        error_msg = f"Cannot read from video source\nFailures: {self.consecutive_failures}/{self.config.max_consecutive_failures}"
        self.visualizer.draw_error_message(blank_frame, error_msg)
        cv2.imshow('YOLOv8 Live Detection', blank_frame)
        time.sleep(1)
    
    def _process_successful_frame(self, frame: np.ndarray):
        """Process a successfully read frame"""
        self.consecutive_failures = 0
        current_time = time.time()
        
        # Process detections
        frame_counts, special_detections, current_frame_classes = self.detection_processor.process_detections(
            frame, self.alert_manager, self.time_window_counter
        )
        
        # Apply conditional logic
        self.conditional_processor.apply_conditional_logic(
            frame, special_detections, current_frame_classes, self.visualizer,
            self.alert_manager, current_time
        )
        
        # Draw information and display - FIXED: Added detection_buffer parameter
        processed_frame = self.visualizer.draw_info_panel(
            frame, frame_counts, self.time_window_counter, self.alert_manager,
            self.sequential_memory, self.detection_buffer, True, self.consecutive_failures
        )
        
        cv2.imshow('YOLOv8 Live Detection', processed_frame)
    
    def stop(self):
        """Stop the application"""
        self.is_running = False
    
    def cleanup(self):
        """Clean up resources"""
        self.video_source.release()
        cv2.destroyAllWindows()

def main():
    """Main function to initialize and run the application"""
    # Try different camera indices
    camera_source = 1
    for camera_idx in [1, 2]:
        try:
            test_cap = cv2.VideoCapture(camera_idx)
            if test_cap.isOpened():
                logger.info(f"Found camera at index {camera_idx}")
                test_cap.release()
                camera_source = camera_idx
                break
        except Exception as e:
            logger.warning(f"Camera test failed for index {camera_idx}: {e}")
    
    config = AppConfig(
        model_path=Path(r"D:\RaihanFarid\Dokumen\Object Detection\CV_model\Segmentation1.torchscript"),
        camera_source=camera_source,
        frame_width=640,
        frame_height=480,
        confidence_threshold=0.5,
        detection_delay_time=1.0,  # Fixed: should be float
        minimum_detection_frames=3,  # Fixed: consistent with delay time
        class_color_map={1: (0, 0, 255), 0: (0, 235, 0), 2: (0, 0, 255)},
        alert_sound_path=Path(r"D:\RaihanFarid\Dokumen\Object Detection\usedAudio\level-up-07-383747.mp3"),
        combination_alert_sounds={
            "class_0_1": Path(r"D:\RaihanFarid\Dokumen\Object Detection\usedAudio\new-notification-09-352705.mp3"),
            "class_0_2": Path(r"D:\RaihanFarid\Dokumen\Object Detection\usedAudio\new-notification-026-380249.mp3"),
            "class_1_2": Path(r"D:\RaihanFarid\Dokumen\Object Detection\usedAudio\notification-alert-269289.mp3"),
            "all_three": Path(r"D:\RaihanFarid\Dokumen\Object Detection\usedAudio\positive-notification-alert-351299.mp3"),
        },
        combination_cooldowns={
            "class_0_1": 15,
            "class_0_2": 15,
            "class_1_2": 15,
            "all_three": 30,
        },
        alert_cooldown=5,
        sequential_suppression_time=30,
        suppression_cooldowns={
            "class_0_1": 25,
            "class_0_2": 25,
            "class_1_2": 25,
            "all_three": 35,
        },
        counter_time_window=10,
        reconnect_delay=5,
        max_consecutive_failures=3,
        special_classes=(0, 1, 2),
        conditional_box_colors={
            "all_three": (0, 255, 255),
            "two_classes": (255, 0, 255),
            "sequential_complete": (0, 255, 0),
            "suppressed": (128, 128, 128),
            "pending": (255, 165, 0)
        },
        mask_alpha=0.5,
        draw_masks=True,
        draw_boxes=True,
        display_width=1024,
        display_height=576
    )
    
    # Create and run the application
    app = ObjectDetectorApp(config)
    
    try:
        app.run()
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
    finally:
        app.cleanup()

if __name__ == "__main__":
    main()