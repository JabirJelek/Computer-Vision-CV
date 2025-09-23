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
    alert_cooldown: int = 5
    class_cooldowns: Dict[int, int] = field(default_factory=dict)
    
    # Counter configuration
    counter_time_window: int = 10
    
    # RTSP configuration
    reconnect_delay: int = 5
    max_consecutive_failures: int = 3
    
    # Sequence detection configuration (NEW)
    sequence_time_window: float = 10.0  # Time window for sequence detection in seconds
    min_sequence_length: int = 2  # Minimum sequence length to track
    max_sequence_length: int = 10  # Maximum sequence length to store

    # Segmentation configuration
    mask_alpha: float = 0.5  # Controls transparency of segmentation masks
    draw_masks: bool = True   # Whether to draw segmentation masks
    mask_ratio: int = 4       # Downsample ratio for mask resolution :cite[1]
    
    # Display configuration
    display_width: int = 1024
    display_height: int = 576
    
    def __post_init__(self):
        # Initialize default values
        if not self.class_color_map:
            self.class_color_map = {0: (0, 0, 255), 1: (0, 255, 0)}
        
        if not self.class_cooldowns:
            self.class_cooldowns = {0: 10, 1: 5, 2: 15}

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
    """YOLO model implementation"""
    
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
    """Stores detection sequences across multiple frames with time window"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.detection_history = deque(maxlen=config.max_sequence_length)
        self.sequence_buffer = deque()
    
    def add_detection(self, class_id: int, class_name: str, confidence: float):
        """Add detection to memory with timestamp"""
        current_time = time.time()
        detection_record = {
            'class_id': class_id,
            'class_name': class_name,
            'confidence': confidence,
            'timestamp': current_time,
            'time_str': time.strftime('%H:%M:%S', time.localtime(current_time))
        }
        
        self.detection_history.append(detection_record)
        self._cleanup_old_detections(current_time)
        
        return detection_record
    
    def _cleanup_old_detections(self, current_time: float):
        """Remove detections older than the time window"""
        while (self.detection_history and 
               current_time - self.detection_history[0]['timestamp'] > self.config.sequence_time_window):
            self.detection_history.popleft()
    
    def get_current_sequence(self) -> List[dict]:
        """Get all detections within the time window"""
        current_time = time.time()
        self._cleanup_old_detections(current_time)
        return list(self.detection_history)
    
    def get_sequence_summary(self) -> dict:
        """Get summary of current sequence"""
        sequence = self.get_current_sequence()
        if not sequence:
            return {
                'total_detections': 0,
                'unique_classes': set(),
                'class_counts': {},
                'time_span': 0,
                'is_sequence_active': False
            }
        
        class_counts = Counter([det['class_id'] for det in sequence])
        time_span = sequence[-1]['timestamp'] - sequence[0]['timestamp']
        
        return {
            'total_detections': len(sequence),
            'unique_classes': set(class_counts.keys()),
            'class_counts': dict(class_counts),
            'time_span': time_span,
            'is_sequence_active': time_span < self.config.sequence_time_window,
            'start_time': sequence[0]['time_str'],
            'end_time': sequence[-1]['time_str']
        }
    
    def detect_patterns(self) -> List[dict]:
        """Detect patterns in the current sequence"""
        sequence = self.get_current_sequence()
        if len(sequence) < self.config.min_sequence_length:
            return []
        
        patterns = []
        
        # Detect repeated classes
        class_sequence = [det['class_id'] for det in sequence]
        current_class = class_sequence[0]
        pattern_length = 1
        
        for i in range(1, len(class_sequence)):
            if class_sequence[i] == current_class:
                pattern_length += 1
            else:
                if pattern_length >= 2:  # Only report patterns of 2 or more
                    patterns.append({
                        'type': 'repetition',
                        'class_id': current_class,
                        'class_name': sequence[i-1]['class_name'],
                        'length': pattern_length,
                        'start_index': i - pattern_length,
                        'end_index': i - 1
                    })
                current_class = class_sequence[i]
                pattern_length = 1
        
        # Check for the last pattern
        if pattern_length >= 2:
            patterns.append({
                'type': 'repetition',
                'class_id': current_class,
                'class_name': sequence[-1]['class_name'],
                'length': pattern_length,
                'start_index': len(class_sequence) - pattern_length,
                'end_index': len(class_sequence) - 1
            })
        
        return patterns
    
    def clear_sequence(self):
        """Clear the current sequence"""
        self.detection_history.clear()

class AlertManager:
    """Manages audio alerts with cooldown functionality"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.alert_timers = {}
        self._init_audio()
    
    def _init_audio(self):
        """Initialize audio system"""
        try:
            pygame.mixer.init()
            self.alert_sound = pygame.mixer.Sound(str(self.config.alert_sound_path))
        except Exception as e:
            logger.error(f"Audio initialization error: {e}")
            self.alert_sound = None
    
    def should_trigger_alert(self, cls_id: int) -> bool:
        """Check if alert should be triggered"""
        current_time = time.time()
        cooldown = self.config.class_cooldowns.get(cls_id, self.config.alert_cooldown)
        
        if cls_id not in self.alert_timers or (current_time - self.alert_timers[cls_id]) >= cooldown:
            self.alert_timers[cls_id] = current_time
            return True
        return False
    
    def play_alert(self):
        """Play alert sound"""
        if self.alert_sound:
            threading.Thread(target=self.alert_sound.play, daemon=True).start()

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
    """Handles visualization of detection results"""
    
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
    
    def draw_detection(self, frame: np.ndarray, detection: DetectionResult):
        """Draw single detection"""
        x1, y1, x2, y2 = map(int, detection.box.xyxy[0])
        
        # Choose color based on class
        color = self.config.class_color_map.get(detection.class_id, (255, 255, 255))
        
        # Draw mask
        if detection.mask is not None:
            self.draw_mask(frame, detection.mask, color)
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
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
    
    def draw_sequence_info(self, frame: np.ndarray, sequential_memory: SequentialMemory):
        """Draw sequence information panel"""
        summary = sequential_memory.get_sequence_summary()
        patterns = sequential_memory.detect_patterns()
        
        lines = [
            "=== SEQUENCE DETECTION ===",
            f"Time Window: {self.config.sequence_time_window}s",
            f"Total Detections: {summary['total_detections']}",
            f"Unique Classes: {len(summary['unique_classes'])}",
            f"Time Span: {summary['time_span']:.1f}s",
            f"Active: {'Yes' if summary['is_sequence_active'] else 'No'}",
            ""
        ]
        
        # Add class counts
        if summary['class_counts']:
            lines.append("Class Counts:")
            for cls_id, count in summary['class_counts'].items():
                lines.append(f"  Class {cls_id}: {count}")
        else:
            lines.append("No detections")
        
        # Add patterns
        if patterns:
            lines.append("")
            lines.append("Detected Patterns:")
            for pattern in patterns:
                if pattern['type'] == 'repetition':
                    lines.append(f"  {pattern['class_name']} x{pattern['length']}")
        
        # Draw the panel
        self._draw_text_panel(frame, lines, 10, 30, 300, right_align=False)
    
    def draw_info_panel(self, frame: np.ndarray, frame_counts: Counter, 
                       time_window_counter: TimeWindowCounter, alert_manager: AlertManager,
                       sequential_memory: SequentialMemory, is_connected: bool, 
                       failure_count: int):
        """Draw all information panels on the frame"""
        # Resize frame for display
        frame = cv2.resize(frame, (self.config.display_width, self.config.display_height))
        
        # Draw sequence info (left side)
        self.draw_sequence_info(frame, sequential_memory)
        
        # Draw counter panel (right side)
        self._draw_counter_panel(frame, frame_counts, time_window_counter)
        
        # Draw cooldown status
        self._draw_cooldown_status(frame, alert_manager)
        
        # Draw connection status
        self._draw_connection_status(frame, is_connected, failure_count)
        
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
    """Processes detections and applies business logic"""
    
    def __init__(self, config: AppConfig, model: IModel, sequential_memory: SequentialMemory):
        self.config = config
        self.model = model
        self.sequential_memory = sequential_memory
    
    def process_detections(self, frame: np.ndarray, alert_manager: AlertManager, 
                          time_window_counter: TimeWindowCounter) -> Tuple[Counter, List[DetectionResult]]:
        """Process frame and return detection results"""
        frame_counts = Counter()
        
        # Run model inference
        detections = self.model.predict(frame)
        
        for detection in detections:
            # Update counters
            frame_counts[detection.class_name] += 1
            time_window_counter.add_detection(detection.class_name)
            
            # Store in sequential memory
            self.sequential_memory.add_detection(
                detection.class_id, 
                detection.class_name, 
                detection.confidence
            )
            
            # Check for alerts
            if (detection.class_id in self.config.class_cooldowns and 
                alert_manager.should_trigger_alert(detection.class_id)):
                alert_manager.play_alert()
        
        return frame_counts, detections

class ObjectDetectorApp:
    """Main application class"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.video_source = VideoSourceFactory.create_video_source(config)
        self.model = YOLOModel(config)
        self.sequential_memory = SequentialMemory(config)  # Updated
        self.alert_manager = AlertManager(config)
        self.visualizer = DetectionVisualizer(config)
        self.time_window_counter = TimeWindowCounter(config.counter_time_window)
        self.detection_processor = DetectionProcessor(config, self.model, self.sequential_memory)
        
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
                # Clear sequence memory when 'c' is pressed
                elif cv2.waitKey(1) & 0xFF == ord('c'):
                    self.sequential_memory.clear_sequence()
                    logger.info("Sequence memory cleared")
                    
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
        
        # Process detections
        frame_counts, detections = self.detection_processor.process_detections(
            frame, self.alert_manager, self.time_window_counter
        )
        
        # Draw individual detections
        for detection in detections:
            self.visualizer.draw_detection(frame, detection)
        
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
        alert_sound_path=Path(r"D:\RaihanFarid\Dokumen\Object Detection\usedAudio\notification-alert-269289.mp3"),
        alert_cooldown=5,
        class_cooldowns={0: 15, 1: 5, 2: 15},
        counter_time_window=10,
        reconnect_delay=5,
        max_consecutive_failures=3,
        # New sequence detection parameters
        sequence_time_window=10.0,
        min_sequence_length=2,
        max_sequence_length=10,
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