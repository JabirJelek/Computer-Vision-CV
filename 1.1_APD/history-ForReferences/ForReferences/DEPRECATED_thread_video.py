
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
    Supports camera devices, RTSP streams, and video files
    """
    
    def __init__(self, source=0, fps=30, processing_interval=0.5, display_width=640, 
                model_path="path/to/your/model.pt", conf_threshold=0.25):
        """
        Args:
            source: Camera index (int), RTSP URL (string), or video file path (string)
            fps: Target FPS for camera (ignored for RTSP/video files)
            processing_interval: Time in seconds between processing frames
            display_width: Width for resizing display frame
            model_path: Path to YOLO model weights
            conf_threshold: Confidence threshold for detections
        """
        self.source = source
        self.processing_interval = processing_interval
        self.display_width = display_width
        self.conf_threshold = conf_threshold
        
        # Auto-detect source type
        if isinstance(source, str):
            if source.startswith(('rtsp://', 'rtmp://', 'http://', 'tcp://')):
                self.source_type = 'rtsp'
            elif source.endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm')):
                self.source_type = 'video_file'
            else:
                self.source_type = 'video_file'  # Assume video file
        else:
            self.source_type = 'camera'  # Integer source means camera index
        
        # Initialize capture first to get video properties quickly
        print(f"Initializing {self.source_type}: {source}")
        self.capture = cv.VideoCapture(source)
        
        # Set source-specific options
        if self.source_type == 'rtsp':
            self.capture.set(cv.CAP_PROP_BUFFERSIZE, 1)
            self.capture.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'H264'))
        elif self.source_type == 'camera':
            self.capture.set(cv.CAP_PROP_FPS, fps)
        # video_file uses default OpenCV settings
        
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open {self.source_type}: {source}")
            
        # Get video properties
        self.frame_width = int(self.capture.get(cv.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.capture.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.actual_fps = self.capture.get(cv.CAP_PROP_FPS)
        self.video_total_frames = int(self.capture.get(cv.CAP_PROP_FRAME_COUNT)) if self.source_type == 'video_file' else 0
        
        # Calculate display dimensions maintaining aspect ratio
        self.display_height = int((self.display_width / self.frame_width) * self.frame_height)
        
        print(f"Video properties: {self.frame_width}x{self.frame_height} at {self.actual_fps:.2f} FPS")
        if self.source_type == 'video_file':
            print(f"Video total frames: {self.video_total_frames}")
        print(f"Display size: {self.display_width}x{self.display_height}")
        
        # Thread synchronization
        self.lock = threading.Lock()
        self.latest_frame = None
        self.frame_counter = 0
        self.running = False
        self.processing_enabled = False  # Start with processing disabled
        
        # Threads
        self.capture_thread = None
        self.processing_thread = None
        
        # Performance monitoring
        self.capture_failures = 0
        self.max_capture_failures = 10
        self.detection_count = 0
        
        # Initialize YOLO model (moved after video setup for faster startup)
        self.model_path = model_path
        self.model = None  # Will be loaded in start() method
        
    def _initialize_model(self):
        """Initialize YOLO model with error handling"""
        try:
            # Replace this path with your actual model path
            placeholder_path = Path(self.model_path)
            
            if not placeholder_path.exists():
                print(f"Warning: Model path '{self.model_path}' does not exist.")
                print("Please update the 'model_path' parameter with your actual model path.")
                print("For now, using a pretrained YOLO11n model as placeholder.")
                model = YOLO("yolo11n.pt")  # Fallback to pretrained model:cite[2]
            else:
                print(f"Loading model from: {self.model_path}")
                # Load model based on file extension
                if self.model_path.endswith('.torchscript'):
                    model = YOLO(self.model_path, task='detect')  # Explicitly specify task for TorchScript:cite[2]
                else:
                    model = YOLO(self.model_path)  # Load your custom model:cite[2]
            
            # Warm up the model with a dummy inference to initialize it faster
            print("Warming up YOLO model...")
            dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = model.predict(dummy_frame, conf=self.conf_threshold, verbose=False)
            
            print(f"YOLO model loaded successfully: {model.__class__.__name__}")
            return model
            
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            print("Falling back to pretrained YOLO11n model")
            try:
                return YOLO("yolo11n.pt")  # Ultimate fallback:cite[2]
            except Exception as e2:
                print(f"Even fallback model failed: {e2}")
                return None
    
    def start(self):
        """Start both capture and processing threads"""
        # Start capture thread first for immediate video playback
        self.running = True
        
        # Start capture thread immediately
        self.capture_thread = threading.Thread(target=self._capture_loop, name="CaptureThread")
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        # Start processing thread but delay YOLO initialization
        self.processing_thread = threading.Thread(target=self._processing_loop, name="ProcessingThread")
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        source_type_map = {
            'camera': 'Camera',
            'rtsp': 'RTSP Stream', 
            'video_file': 'Video File'
        }
        source_type = source_type_map.get(self.source_type, 'Unknown')
        
        print(f"Started SelectiveFrameProcessor with YOLO:")
        print(f"  - Source: {source_type} ({self.source})")
        print(f"  - Capture: Continuous")
        print(f"  - YOLO Processing: Every {self.processing_interval} seconds")
        print(f"  - Display size: {self.display_width}x{self.display_height}")
        print(f"  - Confidence threshold: {self.conf_threshold}")
        
        # Initialize model in background after threads are started
        print("Initializing YOLO model in background...")
        model_init_thread = threading.Thread(target=self._background_model_init, name="ModelInitThread")
        model_init_thread.daemon = True
        model_init_thread.start()
        
    def _background_model_init(self):
        """Initialize model in background thread to avoid blocking video capture"""
        self.model = self._initialize_model()
        if self.model is not None:
            self.processing_enabled = True
            print("YOLO model initialized and processing enabled!")
        else:
            print("Warning: YOLO model failed to initialize. Running in video-only mode.")
        
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
        print(f"Capture thread started - {self.source_type}")
        frames_captured = 0
        self.capture_failures = 0
        
        while self.running:
            ret, frame = self.capture.read()
            
            if not ret:
                # Check if video file has ended
                if self.source_type == 'video_file':
                    print("Video file ended - waiting for processing to complete...")
                    # Allow processing thread to finish current frame
                    time.sleep(2)
                    self.running = False
                    break
                    
                self.capture_failures += 1
                print(f"Warning: Failed to capture frame (failure #{self.capture_failures})")
                
                if self.source_type == 'rtsp' and self.capture_failures >= self.max_capture_failures:
                    print("Multiple RTSP failures - attempting reconnection...")
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
    
    def _run_yolo_detection(self, frame):
        """Run YOLO object detection on a single frame"""
        if self.model is None:
            return frame, 0
            
        try:
            # Run YOLO inference with optimized settings:cite[2]
            results = self.model.predict(
                source=frame,
                conf=self.conf_threshold,
                verbose=False,  # Set to True for detailed inference info
                imgsz=640,     # Standard size for faster inference
                device='cpu'   # Explicitly set device
            )
            
            # Process results
            if results and len(results) > 0:
                # Annotate frame with detections:cite[2]
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
        print("Processing thread started - waiting for YOLO model initialization...")
        frames_processed = 0
        last_processing_time = time.time()
        model_wait_start = time.time()
        model_timeout = 30  # Wait up to 30 seconds for model
        
        while self.running:
            # Wait for model to be initialized with timeout
            if not self.processing_enabled:
                if time.time() - model_wait_start > model_timeout:
                    print("Model initialization timeout - running in video-only mode")
                    self.processing_enabled = True  # Continue without YOLO
                time.sleep(0.1)
                continue
                
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
                    
                    # Run YOLO object detection if model is available
                    if self.model is not None:
                        processed_frame, detections = self._run_yolo_detection(frame_to_process)
                    else:
                        processed_frame = frame_to_process
                        detections = 0
                    
                    # Resize frame to smaller display size
                    resized_frame = self._resize_frame(processed_frame)
                    
                    # Add informational overlay with detection info
                    self._add_info_overlay(resized_frame, frame_num, frames_processed, detections)
                    
                    # Display the resized frame with detections
                    window_name = "YOLO Object Detection - Selective Processing" if self.model else "Video Stream - Selective Processing"
                    cv.imshow(window_name, resized_frame)
                    
                    # Handle key presses - only ESC for exit
                    key = cv.waitKey(1) & 0xFF
                    if key == 27:  # ESC key
                        self.running = False
                        break
                
                last_processing_time = current_time
                
            time.sleep(0.001)
                
        print(f"Processing thread stopped. Total frames processed: {frames_processed}")
    
    def _resize_frame(self, frame):
        """Resize frame to display dimensions maintaining aspect ratio"""
        return cv.resize(frame, (self.display_width, self.display_height))
    
    def _add_info_overlay(self, frame, frame_num, processed_count, detections_count):
        """Add informational text overlay to the frame"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Use the detected source type
        source_type_map = {
            'camera': 'Camera',
            'rtsp': 'RTSP', 
            'video_file': 'Video File'
        }
        source_type = source_type_map.get(self.source_type, 'Unknown')
        
        # Scale font size based on display width
        font_scale = 0.5 if self.display_width <= 640 else 0.7
        thickness = 1 if self.display_width <= 640 else 2
        
        # Model status
        model_status = "YOLO Ready" if self.model is not None else "YOLO Loading..."
        
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
        cv.putText(frame, f"Model: {model_status}", (10, 125), 
                  cv.FONT_HERSHEY_SIMPLEX, font_scale-0.1, (255, 255, 0), 1)
        cv.putText(frame, f"Time: {timestamp}", (10, 140), 
                  cv.FONT_HERSHEY_SIMPLEX, font_scale-0.1, (255, 255, 255), 1)
        cv.putText(frame, f"Interval: {self.processing_interval}s", (10, 155), 
                  cv.FONT_HERSHEY_SIMPLEX, font_scale-0.1, (255, 255, 255), 1)
        cv.putText(frame, "Press ESC to exit", (10, 170), 
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
        if self.model is not None:
            self.model.conf = self.conf_threshold
    
    def get_video_properties(self):
        """Get current video stream properties"""
        source_type_map = {
            'camera': 'Camera',
            'rtsp': 'RTSP', 
            'video_file': 'Video File'
        }
        
        return {
            'width': self.frame_width,
            'height': self.frame_height,
            'fps': self.actual_fps,
            'source_type': source_type_map.get(self.source_type, 'Unknown'),
            'display_size': f"{self.display_width}x{self.display_height}",
            'model': str(self.model_path),
            'confidence_threshold': self.conf_threshold,
            'total_detections': self.detection_count,
            'model_loaded': self.model is not None
        }

def main():
    """
    Demonstration of the SelectiveFrameProcessor with YOLO Object Detection
    """
    print("Selective Frame Processing with YOLO Object Detection")
    print("=" * 60)
    print("Features:")
    print("- Camera, RTSP, and Video File support")
    print("- Multi-threaded architecture")
    print("- Selective frame sampling for CPU efficiency")
    print("- YOLO object detection integration")
    print("- Resizable display output")
    print("- Real-time performance monitoring")
    print("\nControls:")
    print("  ESC: Exit")
    print("=" * 60)
    
    # Choose source type
    while True:
        choice = input("Choose source type:\n1. Camera\n2. RTSP Stream\n3. Video File\nEnter choice (1, 2, or 3): ").strip()
        
        if choice == '1':
            camera_index = int(input("Enter camera index (default 0): ") or "0")
            display_width = int(input("Enter display width (default 640): ") or "640")
            processing_interval = float(input("Enter processing interval in seconds (default 0.5): ") or "0.5")
            model_path = input("Enter YOLO model path (or press Enter for pretrained model): ").strip()
            
            if not model_path:
                model_path = "yolo11n.pt"
                print("Using pretrained YOLO11n model")
            
            processor = SelectiveFrameProcessor(
                source=camera_index,
                fps=30,
                processing_interval=processing_interval,
                display_width=display_width,
                model_path=model_path
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
            
            if not model_path:
                model_path = "yolo11n.pt"
                print("Using pretrained YOLO11n model")
            
            processor = SelectiveFrameProcessor(
                source=rtsp_url,
                processing_interval=processing_interval,
                display_width=display_width,
                model_path=model_path
            )
            break
            
        elif choice == '3':
            video_path = input("Enter video file path: ").strip()
            if not video_path:
                print("Please provide a valid video file path.")
                continue
                
            display_width = int(input("Enter display width (default 640): ") or "640")
            processing_interval = float(input("Enter processing interval in seconds (default 0.5): ") or "0.5")
            model_path = input("Enter YOLO model path (or press Enter for pretrained model): ").strip()
            
            if not model_path:
                model_path = "yolo11n.pt"
                print("Using pretrained YOLO11n model")
            
            processor = SelectiveFrameProcessor(
                source=video_path,
                processing_interval=processing_interval,
                display_width=display_width,
                model_path=model_path
            )
            break
            
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")
    
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