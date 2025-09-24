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
    
    # Updated alert configuration
    alert_sound_path: Path = Path(r"D:\RaihanFarid\Dokumen\Object Detection\usedAudio\notification-alert-269289.mp3")
    combination_alert_sounds: Dict[str, Path] = field(default_factory=dict)  # New: sound files for combinations
    alert_cooldown: int = 5
    combination_cooldowns: Dict[str, int] = field(default_factory=dict)  # New: cooldowns for combinations
    
    # Counter configuration
    counter_time_window: int = 10

    # Add detection confirmation delay
    detection_confirmation_delay: float = 1.5  # seconds to wait before confirming detection
    combination_wait_times: Dict[str, float] = field(default_factory=dict)  # per-combination delays
    
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
        
        #if not self.class_cooldowns:
        #    self.class_cooldowns = {0: 10, 1: 5, 2: 15}

        # Initialize combination-specific wait times
        if not self.combination_wait_times:
            self.combination_wait_times = {
                "class_0_1": 2.0,  # 2 seconds for class 0+1
                "class_0_2": 1.5,  # 1.5 seconds for class 0+2
                "class_1_2": 2.5,  # 2.5 seconds for class 1+2
                "all_three": 3.0   # 3 seconds for all three classes
            }        
            
        if not self.conditional_box_colors:
            self.conditional_box_colors = {
                "all_three": (0, 255, 255),
                "two_classes": (255, 0, 255),
                "sequential_complete": (0, 255, 0)
            }

        # Initialize combination alert sounds
        if not self.combination_alert_sounds:
            self.combination_alert_sounds = {
                "class_0_1": Path(r"D:\RaihanFarid\Dokumen\Object Detection\usedAudio\01_combination.mp3"),
                "class_0_2": Path(r"D:\RaihanFarid\Dokumen\Object Detection\usedAudio\02_combination.mp3"),
                "class_1_2": Path(r"D:\RaihanFarid\Dokumen\Object Detection\usedAudio\12_combination.mp3"),
            }
        
        if not self.combination_cooldowns:
            self.combination_cooldowns = {
                "class_0_1": 10,
                "class_0_2": 10, 
                "class_1_2": 10,
                "all_three": 15,
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

class DetectionConfirmationTracker:
    """Tracks detection states and confirms them after a delay"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.detection_start_times = {}  # combination_name -> first detection time
        self.confirmed_detections = set()  # combinations that have been confirmed
        self.pending_detections = set()  # combinations currently being tracked
    
    def update_detection_state(self, active_combinations: Set[str], current_time: float) -> Set[str]:
        """Update detection states and return confirmed combinations"""
        confirmed = set()
        
        # Check each active combination
        for combo_name in active_combinations:
            if combo_name not in self.pending_detections:
                # Start tracking new detection
                self.detection_start_times[combo_name] = current_time
                self.pending_detections.add(combo_name)
                logger.info(f"Started tracking combination: {combo_name}")
            
            # Check if detection should be confirmed
            wait_time = self.config.combination_wait_times.get(
                combo_name, self.config.detection_confirmation_delay
            )
            
            elapsed_time = current_time - self.detection_start_times[combo_name]
            if elapsed_time >= wait_time and combo_name not in self.confirmed_detections:
                confirmed.add(combo_name)
                self.confirmed_detections.add(combo_name)
                logger.info(f"Combination confirmed: {combo_name} after {elapsed_time:.1f}s")
        
        # Remove combinations that are no longer active
        for combo_name in list(self.pending_detections):
            if combo_name not in active_combinations:
                self.pending_detections.discard(combo_name)
                self.detection_start_times.pop(combo_name, None)
                self.confirmed_detections.discard(combo_name)
                logger.info(f"Stopped tracking combination: {combo_name}")
        
        return confirmed
    
    def get_pending_detection_time(self, combo_name: str, current_time: float) -> float:
        """Get how long a detection has been pending"""
        if combo_name in self.detection_start_times:
            return current_time - self.detection_start_times[combo_name]
        return 0.0
    
    def reset(self):
        """Reset all tracking states"""
        self.detection_start_times.clear()
        self.confirmed_detections.clear()
        self.pending_detections.clear()

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
    
    def add_detection(self, class_id: int) -> int:
        """Add detection to sequence"""
        current_time = time.time()
        active_sequence = self._get_active_sequence()
        
        if active_sequence and class_id not in active_sequence:
            active_sequence[class_id] = current_time
            return self.current_sequence_id
        else:
            self.current_sequence_id += 1
            self.detection_sequences[self.current_sequence_id] = {class_id: current_time}
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
    
    def get_sequence_duration(self, current_time: float) -> float:
        """Get how long the current sequence has been active"""
        active_sequence = self._get_active_sequence()
        if not active_sequence or not active_sequence.values():
            return 0.0
        
        earliest_detection = min(active_sequence.values())
        return current_time - earliest_detection

    def get_sequence_classes(self) -> Set[int]:
        """Get classes in current sequence"""
        active_sequence = self._get_active_sequence()
        return set(active_sequence.keys()) if active_sequence else set()

    def get_class_detection_time(self, class_id: int) -> Optional[float]:
        """Get when a specific class was first detected in current sequence"""
        active_sequence = self._get_active_sequence()
        return active_sequence.get(class_id) if active_sequence else None
    
    def is_sequence_mature(self, required_classes: Set[int], min_duration: float, current_time: float) -> bool:
        """Check if sequence contains required classes and has existed for min duration"""
        active_sequence = self._get_active_sequence()
        if not active_sequence:
            return False
        
        # Check if all required classes are present
        if not required_classes.issubset(set(active_sequence.keys())):
            return False
        
        # Check if sequence has existed long enough
        sequence_duration = self.get_sequence_duration(current_time)
        return sequence_duration >= min_duration

class AlertManager:
    """Manages conditional audio alerts based on class combinations"""
    
    def __init__(self, config: AppConfig, sequential_memory: SequentialMemory):
        self.config = config
        self.alert_timers = {}  # Now tracks combination alerts instead of class alerts
        self.combination_sounds = {}  # Stores loaded sound objects for combinations
        self.sequential_memory = sequential_memory
        self.confirmation_tracker = DetectionConfirmationTracker(config)        
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
    
    def _should_trigger_combination(self, combination_name: str, current_time: float) -> bool:
        """Check if combination should trigger based on SequentialMemory state"""
        # Get the sequence data from SequentialMemory
        active_sequence = self.sequential_memory._get_active_sequence()
        
        if not active_sequence:
            return False
        
        # Check if the required classes are in the current sequence
        required_classes = self._get_required_classes_for_combination(combination_name)
        sequence_classes = set(active_sequence.keys())
        
        # All required classes must be in the sequence
        if not required_classes.issubset(sequence_classes):
            return False
        
        # Check if the sequence has been active long enough
        class_times = [active_sequence[cls_id] for cls_id in required_classes if cls_id in active_sequence]
        if not class_times:
            return False
        
        # Use the earliest detection time in the sequence
        earliest_detection_time = min(class_times)
        wait_time = self.config.combination_wait_times.get(combination_name, self.config.detection_confirmation_delay)
        
        # Check if enough time has passed since the first detection in this sequence
        if (current_time - earliest_detection_time) >= wait_time:
            # Check cooldown
            if self._is_off_cooldown(combination_name, current_time):
                return True
        
        return False    

    def _get_required_classes_for_combination(self, combination_name: str) -> Set[int]:
        """Map combination names to required class sets"""
        combination_map = {
            "class_0_1": {0, 1},
            "class_0_2": {0, 2},
            "class_1_2": {1, 2}
        }
        return combination_map.get(combination_name, set())
    
    def _is_off_cooldown(self, combination_name: str, current_time: float) -> bool:
        """Check if combination is off cooldown"""
        cooldown = self.config.combination_cooldowns.get(combination_name, self.config.alert_cooldown)
        
        if (combination_name not in self.alert_timers or 
            (current_time - self.alert_timers[combination_name]) >= cooldown):
            self.alert_timers[combination_name] = current_time
            return True
        return False

    def should_trigger_combination_alert(self, combination_name: str) -> bool:
        """Check if combination alert should be triggered based on cooldown"""
        current_time = time.time()
        cooldown = self.config.combination_cooldowns.get(combination_name, self.config.alert_cooldown)
        
        if (combination_name not in self.alert_timers or 
            (current_time - self.alert_timers[combination_name]) >= cooldown):
            self.alert_timers[combination_name] = current_time
            return True
        return False
    
    def play_combination_alert(self, combination_name: str):
        """Play alert sound for specific combination"""
        sound = self.combination_sounds.get(combination_name)
        if sound:
            threading.Thread(target=sound.play, daemon=True).start()
            logger.info(f"Playing alert for combination: {combination_name}")
        else:
            logger.warning(f"No sound configured for combination: {combination_name}")

    def check_combination_alerts(self, current_frame_classes: Set[int], current_time: float) -> Set[str]:
        """Check which combinations should trigger alerts based on SequentialMemory"""
        confirmed_combinations = set()
        
        # Get current special classes in frame
        current_special = current_frame_classes.intersection(set(self.config.special_classes))
        
        # Check each combination condition
        if current_special == {0, 1}:  # Only class 0 and 1
            if self._should_trigger_combination("class_0_1", current_time):
                confirmed_combinations.add("class_0_1")
                
        elif current_special == {0, 2}:  # Only class 0 and 2
            if self._should_trigger_combination("class_0_2", current_time):
                confirmed_combinations.add("class_0_2")
                
        elif current_special == {1, 2}:  # Only class 1 and 2
            if self._should_trigger_combination("class_1_2", current_time):
                confirmed_combinations.add("class_1_2")
        
        return confirmed_combinations

    def update_detection_states(self, current_frame_classes: Set[int], 
                              active_temporal_classes: Set[int], current_time: float) -> Set[str]:
        """Update detection states and return confirmed combinations"""
        # Determine which combinations are currently active
        active_combinations = self._get_active_combinations(current_frame_classes, active_temporal_classes)
        
        # Update confirmation tracker
        confirmed = self.confirmation_tracker.update_detection_state(active_combinations, current_time)
        
        return confirmed
    
    def _get_active_combinations(self, current_frame_classes: Set[int], 
                               active_temporal_classes: Set[int]) -> Set[str]:
        """Get currently active combinations"""
        active_combinations = set()
        
        # Check current frame combinations
        current_special = current_frame_classes.intersection(set(self.config.special_classes))
        
        if current_special == {0, 1}:
            active_combinations.add("class_0_1")
        elif current_special == {0, 2}:
            active_combinations.add("class_0_2")
        elif current_special == {1, 2}:
            active_combinations.add("class_1_2")
        elif current_special == {0, 1, 2}:
            active_combinations.add("all_three")
        
        # Also check temporal combinations for robustness
        temporal_special = active_temporal_classes.intersection(set(self.config.special_classes))
        
        if temporal_special == {0, 1}:
            active_combinations.add("class_0_1_temporal")
        elif temporal_special == {0, 2}:
            active_combinations.add("class_0_2_temporal")
        elif temporal_special == {1, 2}:
            active_combinations.add("class_1_2_temporal")
        
        return active_combinations
    
    def should_trigger_combination_alert(self, combination_name: str) -> bool:
        """Check if combination alert should be triggered (cooldown only)"""
        current_time = time.time()
        cooldown = self.config.combination_cooldowns.get(combination_name, self.config.alert_cooldown)
        
        if (combination_name not in self.alert_timers or 
            (current_time - self.alert_timers[combination_name]) >= cooldown):
            self.alert_timers[combination_name] = current_time
            return True
        return False
    
    def play_confirmed_alert(self, confirmed_combinations: Set[str], current_time: float):
        """Play alerts only for confirmed combinations"""
        for combo_name in confirmed_combinations:
            # Remove temporal suffix for sound lookup
            sound_combo_name = combo_name.replace('_temporal', '')
            
            if self.should_trigger_combination_alert(combo_name):
                self.play_combination_alert(sound_combo_name)
                
                # Log confirmation details
                wait_time = self.config.combination_wait_times.get(
                    sound_combo_name, self.config.detection_confirmation_delay
                )
                logger.info(f"Alert played for {combo_name} after {wait_time}s confirmation")
    
    def get_pending_detection_info(self) -> Dict[str, float]:
        """Get information about pending detections for UI display"""
        current_time = time.time()
        pending_info = {}
        
        for combo_name in self.confirmation_tracker.pending_detections:
            elapsed = self.confirmation_tracker.get_pending_detection_time(combo_name, current_time)
            required = self.config.combination_wait_times.get(
                combo_name.replace('_temporal', ''), 
                self.config.detection_confirmation_delay
            )
            pending_info[combo_name] = (elapsed, required)
        
        return pending_info

class TimeWindowCounter:
    """Manages time-based counter with rolling window"""
    
    def __init__(self, time_window_seconds: int = 20):
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
            self.current_counts[old_class] -= 1
            
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
                               conditional_type: str):
        """Draw combined bounding box for multiple detections"""
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
        
        color = self.config.conditional_box_colors.get(conditional_type, (255, 255, 255))
        labels = {
            "all_three": "ALL THREE CLASSES",
            "two_classes": "TWO SPECIAL CLASSES", 
            "sequential_complete": "SEQUENCE COMPLETE"
        }
        label = labels.get(conditional_type, "COMBINED DETECTION")
        
        if self.config.draw_boxes:
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
            
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
                        sequential_memory: SequentialMemory, is_connected: bool, 
                        failure_count: int):
        """Draw all information panels including confirmation status"""
        # Resize frame for display
        frame = cv2.resize(frame, (self.config.display_width, self.config.display_height))
        
        # Draw counter panel
        self._draw_counter_panel(frame, frame_counts, time_window_counter)
        
        # Draw confirmation status
        self._draw_confirmation_status(frame, alert_manager)
        
        # Draw cooldown status
        self._draw_cooldown_status(frame, alert_manager)
        
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
    
    def _draw_connection_status(self, frame: np.ndarray, is_connected: bool, failure_count: int):
        """Draw connection status"""
        status_text = "Connected" if is_connected else f"Disconnected ({failure_count})"
        color = (0, 255, 0) if is_connected else (0, 0, 255)
        cv2.putText(frame, f"Status: {status_text}", (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def _draw_sequential_status(self, frame: np.ndarray, sequential_memory: SequentialMemory):
        """Draw enhanced sequential detection status with timing info"""
        current_time = time.time()
        memory_classes = sequential_memory.get_sequence_classes()
        sequence_duration = sequential_memory.get_sequence_duration(current_time)
        
        memory_text = f"Sequence: {sorted(memory_classes)} ({sequence_duration:.1f}s)"
        cv2.putText(frame, memory_text, (10, frame.shape[0] - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show wait times for active combinations
        if memory_classes:
            y_offset = frame.shape[0] - 80
            for combo_name, wait_time in self.config.combination_wait_times.items():
                required_classes = self._get_required_classes_for_display(combo_name)
                if required_classes.issubset(memory_classes):
                    status = "READY" if sequence_duration >= wait_time else "WAITING"
                    color = (0, 255, 0) if sequence_duration >= wait_time else (0, 255, 255)
                    combo_text = f"{combo_name}: {status} ({sequence_duration:.1f}s/{wait_time}s)"
                    cv2.putText(frame, combo_text, (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    y_offset -= 20
    
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
    
    def _draw_confirmation_status(self, frame: np.ndarray, alert_manager: AlertManager):
        """Draw detection confirmation status"""
        pending_info = alert_manager.get_pending_detection_info()
        current_time = time.time()
        
        y_offset = 30
        for i, (combo_name, (elapsed, required)) in enumerate(pending_info.items()):
            progress = min(elapsed / required, 1.0)
            status_text = f"{combo_name}: {progress:.0%} ({elapsed:.1f}s/{required:.1f}s)"
            
            # Color based on progress (yellow -> green)
            if progress < 1.0:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 255, 0)    # Green
            
            cv2.putText(frame, status_text, (10, y_offset + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw progress bar
            bar_width = 200
            bar_height = 8
            bar_x = 10
            bar_y = y_offset + i * 25 + 15
            
            # Background bar
            cv2.rectangle(frame, (bar_x, bar_y), 
                         (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
            
            # Progress bar
            progress_width = int(bar_width * progress)
            cv2.rectangle(frame, (bar_x, bar_y), 
                         (bar_x + progress_width, bar_y + bar_height), color, -1)
        
        # If no pending detections, show waiting message
        if not pending_info:
            cv2.putText(frame, "Waiting for detections...", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

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

    def _get_required_classes_for_display(self, combo_name: str) -> Set[int]:
        """Helper for display purposes"""
        combo_map = {
            "class_0_1": {0, 1},
            "class_0_2": {0, 2},
            "class_1_2": {1, 2},
            "all_three": {0, 1, 2}
        }
        return combo_map.get(combo_name, set())

class DetectionProcessor:
    """Processes detections and applies combination-based alert logic"""
    
    def __init__(self, config: AppConfig, model: IModel, sequential_memory: SequentialMemory, temporal_memory: TemporalMemory):
        self.config = config
        self.model = model
        self.sequential_memory = sequential_memory
        self.temporal_memory = temporal_memory
    
    def process_detections(self, frame: np.ndarray, alert_manager: AlertManager, 
                          time_window_counter: TimeWindowCounter) -> Tuple[Counter, List[DetectionResult], Set[int]]:
        """Process frame and update detection states"""
        frame_counts = Counter()
        special_detections = []
        current_frame_classes = set()
        current_time = time.time()
        
        # Run model inference
        detections = self.model.predict(frame)
        
        for detection in detections:
            # Update counters and tracking
            frame_counts[detection.class_name] += 1
            time_window_counter.add_detection(detection.class_name)
            current_frame_classes.add(detection.class_id)
            
            # Update temporal memory
            self.temporal_memory.update_detection(detection.class_id, current_time)
            
            # Track special classes in sequential memory
            if detection.class_id in self.config.special_classes:
                special_detections.append(detection)
                self.sequential_memory.add_detection(detection.class_id)
        
        # Get active temporal classes
        active_temporal_classes = self.temporal_memory.get_active_classes(current_time)
        
        # Check for combination alerts using SequentialMemory
        confirmed_combinations = alert_manager.check_combination_alerts(current_frame_classes, current_time)
        
        # Play alerts for confirmed combinations
        for combo_name in confirmed_combinations:
            alert_manager.play_combination_alert(combo_name)
            
            # Log the sequence duration that triggered the alert
            seq_duration = self.sequential_memory.get_sequence_duration(current_time)
            logger.info(f"Alert triggered for {combo_name} after {seq_duration:.1f}s sequence duration")
        
        return frame_counts, special_detections, current_frame_classes
    
    def _check_combination_alerts(self, current_frame_classes: Set[int], 
                                alert_manager: AlertManager, current_time: float):
        """Check for specific class combinations and trigger alerts"""
        # Get all classes active in time window
        active_classes = self.temporal_memory.get_active_classes(current_time)
        
        # Convert to sets for easier comparison
        current_special = current_frame_classes.intersection(set(self.config.special_classes))
        active_special = active_classes.intersection(set(self.config.special_classes))
        
        # Check for exact combinations in current frame
        if current_special == {0, 1}:  # Only class 0 and 1
            if alert_manager.should_trigger_combination_alert("class_0_1"):
                alert_manager.play_combination_alert("class_0_1")
                logger.info("Combination alert: Class 0 and 1 detected together")
                
        elif current_special == {0, 2}:  # Only class 0 and 2
            if alert_manager.should_trigger_combination_alert("class_0_2"):
                alert_manager.play_combination_alert("class_0_2")
                logger.info("Combination alert: Class 0 and 2 detected together")
                
        elif current_special == {1, 2}:  # Only class 1 and 2
            if alert_manager.should_trigger_combination_alert("class_1_2"):
                alert_manager.play_combination_alert("class_1_2")
                logger.info("Combination alert: Class 1 and 2 detected together")

        elif current_special == {0, 1, 2}:  # All three classes
            if alert_manager.should_trigger_combination_alert("all_three"):
                alert_manager.play_combination_alert("all_three")
                logger.info("Combination alert: All three special classes detected together")
        
        # Optional: Also check in temporal memory (across multiple frames)
        self._check_temporal_combinations(active_special, alert_manager, current_time)
    
    def _check_temporal_combinations(self, active_special: Set[int], 
                                   alert_manager: AlertManager, current_time: float):
        """Check for combinations across temporal window"""
        if active_special == {0, 1}:  # Class 0 and 1 active in time window
            if alert_manager.should_trigger_combination_alert("class_0_1_temporal"):
                alert_manager.play_combination_alert("class_0_1")
                logger.info("Temporal combination: Class 0 and 1 active in time window")
                
        elif active_special == {0, 2}:  # Class 0 and 2 active in time window
            if alert_manager.should_trigger_combination_alert("class_0_2_temporal"):
                alert_manager.play_combination_alert("class_0_2")
                logger.info("Temporal combination: Class 0 and 2 active in time window")
                
        elif active_special == {1, 2}:  # Class 1 and 2 active in time window
            if alert_manager.should_trigger_combination_alert("class_1_2_temporal"):
                alert_manager.play_combination_alert("class_1_2")
                logger.info("Temporal combination: Class 1 and 2 active in time window")

        elif active_special == {0, 1, 2}:  # All three classes active in time window
            if alert_manager.should_trigger_combination_alert("all_three_temporal"):
                alert_manager.play_combination_alert("all_three")
                logger.info("Temporal combination: All three special classes active in time window")

class EnhancedConditionalLogicProcessor:
    """Handles conditional detection logic using SequentialMemory"""
    
    def __init__(self, config: AppConfig, sequential_memory: SequentialMemory, temporal_memory: TemporalMemory):
        self.config = config
        self.sequential_memory = sequential_memory
        self.temporal_memory = temporal_memory
    
    def apply_conditional_logic(self, frame: np.ndarray, special_detections: List[DetectionResult],
                              current_frame_classes: Set[int], visualizer: DetectionVisualizer, 
                              current_time: float) -> bool:
        """Apply conditional logic using SequentialMemory state"""
        
        # Get sequence information
        sequence_classes = self.sequential_memory.get_sequence_classes()
        sequence_duration = self.sequential_memory.get_sequence_duration(current_time)
        
        # Check various conditions for visualization
        if sequence_classes == set(self.config.special_classes):
            visualizer.draw_combined_detection(frame, special_detections, "all_three")
            return True
            
        elif len(sequence_classes) == 2:
            # Determine which specific combination for visualization
            if sequence_classes == {0, 1}:
                combo_type = "class_0_1"
            elif sequence_classes == {0, 2}:
                combo_type = "class_0_2"
            elif sequence_classes == {1, 2}:
                combo_type = "class_1_2"
            else:
                combo_type = "two_classes"
            
            visualizer.draw_combined_detection(frame, special_detections, combo_type)
            return True
            
        elif self._check_sequential_complete(sequence_classes):
            visualizer.draw_combined_detection(frame, special_detections, "sequential_complete")
            return True
        else:
            # Draw individual detections
            for detection in special_detections:
                visualizer.draw_detection(frame, detection)
            return False
    
    def _check_sequential_complete(self, sequence_classes: Set[int]) -> bool:
        """Check if required sequence is complete"""
        return sequence_classes.issuperset(set(self.config.special_classes))
class ObjectDetectorApp:
    """Main application class"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.video_source = VideoSourceFactory.create_video_source(config)
        self.model = YOLOModel(config)
        self.sequential_memory = SequentialMemory()
        self.temporal_memory = TemporalMemory(time_window=10.0)  
        self.alert_manager = AlertManager(config, self.sequential_memory)  
        self.visualizer = DetectionVisualizer(config)
        self.time_window_counter = TimeWindowCounter(config.counter_time_window)
        self.detection_processor = DetectionProcessor(config, self.model, self.sequential_memory, self.temporal_memory)  # Update this line
        self.conditional_processor = EnhancedConditionalLogicProcessor(config, self.sequential_memory, self.temporal_memory)  # Update this line
        
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
        
        # Process detections (alerts are now handled within process_detections)
        frame_counts, special_detections, current_frame_classes = self.detection_processor.process_detections(
            frame, self.alert_manager, self.time_window_counter
        )
        
        # Apply conditional logic
        self.conditional_processor.apply_conditional_logic(
            frame, special_detections, current_frame_classes, self.visualizer, current_time
        )
        
        # Draw information and display
        processed_frame = self.visualizer.draw_info_panel(
            frame, frame_counts, self.time_window_counter, self.alert_manager,
            self.sequential_memory, True, self.consecutive_failures
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
        class_color_map={1: (0, 0, 255), 0: (0, 235, 0), 2: (0, 0, 255)},
        # Updated alert configuration
        alert_sound_path=Path(r"D:\RaihanFarid\Dokumen\Object Detection\usedAudio\level-up-07-383747.mp3"),
        combination_alert_sounds={
            "class_0_1": Path(r"D:\RaihanFarid\Dokumen\Object Detection\usedAudio\new-notification-09-352705.mp3"),
            "class_0_2": Path(r"D:\RaihanFarid\Dokumen\Object Detection\usedAudio\new-notification-026-380249.mp3"),
            "class_1_2": Path(r"D:\RaihanFarid\Dokumen\Object Detection\usedAudio\notification-alert-269289.mp3"),
            "all_three": Path(r"D:\RaihanFarid\Dokumen\Object Detection\usedAudio\level-up-07-383747.mp3"),
        },
        combination_cooldowns={
            "class_0_1": 15,
            "class_0_2": 15,
            "class_1_2": 15,
            "all_three": 30,
        },
        alert_cooldown=5,
        #class_cooldowns={},
        counter_time_window=10,
        reconnect_delay=5,
        max_consecutive_failures=3,
        special_classes=(0, 1, 2),
        conditional_box_colors={
            "all_three": (0, 255, 255),
            "two_classes": (255, 0, 255),
            "sequential_complete": (0, 255, 0)
        },
        mask_alpha=0.5,
        draw_masks=True,
        draw_boxes=True,
        display_width=1024,
        display_height=576,
        
        # Add confirmation delay settings
        detection_confirmation_delay=1.5,
        combination_wait_times={
            "class_0_1": 2.0,
            "class_0_2": 1.5, 
            "class_1_2": 2.5,
            "all_three": 1.0,
        },
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