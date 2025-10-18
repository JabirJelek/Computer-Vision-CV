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
import json
from datetime import datetime

class SelectiveFrameProcessor:
    """
    A three-thread system for efficient frame capture with YOLO object detection:
    - Capture Thread: Continuously captures frames, keeping only the latest
    - Processing Thread: Samples frames at fixed intervals for YOLO inference  
    - Cropping Thread: Receives detection results and extracts cropped images
    Supports both camera devices and RTSP streams
    """
    
    def __init__(self, source=0, fps=30, processing_interval=0.5, is_rtsp=False, display_width=640, 
                 model_path=r"D:\SCMA\3-APD\fromAraya\Computer-Vision-CV\3.1_FaceRecog\yolov11n-face.pt", conf_threshold=0.5, 
                 save_crops=True, crop_output_dir="cropped_faces", 
                 max_crops_per_session=1000, crop_quality=95, 
                 enable_crop_preview=True, auto_create_subdirs=True):
        """
        Args:
            source: Camera device index (int) or RTSP URL (string)
            fps: Target frames per second for camera (ignored for RTSP)
            processing_interval: Time in seconds between processing frames
            is_rtsp: Boolean indicating if source is an RTSP stream
            display_width: Width for resizing display frame (maintains aspect ratio)
            model_path: Path to YOLO model weights (.pt file)
            conf_threshold: Confidence threshold for YOLO detections
            save_crops: Whether to save cropped images to disk
            crop_output_dir: Base directory to save cropped images
            max_crops_per_session: Maximum number of crops to save (0 = unlimited)
            crop_quality: JPEG quality for saved crops (1-100)
            enable_crop_preview: Whether to show crop preview windows
            auto_create_subdirs: Whether to create date-based subdirectories
        """
        self.source = source
        self.is_rtsp = is_rtsp
        self.processing_interval = processing_interval
        self.display_width = display_width
        self.conf_threshold = conf_threshold
        
        # Dynamic crop saving configuration
        self.save_crops = save_crops
        self.crop_output_dir = crop_output_dir
        self.max_crops_per_session = max_crops_per_session
        self.crop_quality = max(1, min(100, crop_quality))
        self.enable_crop_preview = enable_crop_preview
        self.auto_create_subdirs = auto_create_subdirs
        
        # Current crop session directory
        self.current_crop_dir = self._get_current_crop_directory()
        
        # Create output directory for cropped images
        if self.save_crops:
            os.makedirs(self.current_crop_dir, exist_ok=True)
            print(f"Cropped images will be saved to: {self.current_crop_dir}")
        
        # Initialize YOLO model
        self.model_path = model_path
        self.model = self._initialize_model()
        
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
        self.crop_lock = threading.Lock()  # Separate lock for crop settings
        self.latest_frame = None
        self.frame_counter = 0
        self.running = False
        
        # Queue for passing detection results to cropping thread
        self.crop_queue = queue.Queue(maxsize=100)
        
        # Threads
        self.capture_thread = None
        self.processing_thread = None
        self.cropping_thread = None
        
        # Performance monitoring
        self.capture_failures = 0
        self.max_capture_failures = 10
        self.detection_count = 0
        self.crop_count = 0
        self.skipped_crops = 0
        
        # Crop preview windows management
        self.crop_windows = set()
        self.max_preview_windows = 6  # Maximum number of preview windows to show
        
    def _get_current_crop_directory(self):
        """Get the current crop directory, creating subdirectories if enabled"""
        base_dir = Path(self.crop_output_dir)
        
        if self.auto_create_subdirs:
            # Create date-based subdirectory
            date_str = datetime.now().strftime("%Y-%m-%d")
            session_str = datetime.now().strftime("%H-%M-%S")
            session_dir = base_dir / date_str / session_str
        else:
            session_dir = base_dir
            
        return str(session_dir)
    
    def _initialize_model(self):
        """Initialize YOLO model with error handling"""
        try:
            # Replace this path with your actual model path
            placeholder_path = Path(self.model_path)
            
            if not placeholder_path.exists():
                print(f"Warning: Model path '{self.model_path}' does not exist.")
                print("Please update the 'model_path' parameter with your actual model path.")
                print("For now, using a pretrained YOLO11n-face model as placeholder.")
                model = YOLO("yolo11n-face.pt")  # Fallback to face detection model
            else:
                model = YOLO(self.model_path)  # Load your custom model
            
            print(f"YOLO model loaded successfully: {model.__class__.__name__}")
            return model
            
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            print("Falling back to pretrained YOLO11n-face model")
            return YOLO("yolo11n-face.pt")  # Ultimate fallback
    
    def start(self):
        """Start capture, processing, and cropping threads"""
        self.running = True
        
        # Start capture thread
        self.capture_thread = threading.Thread(target=self._capture_loop, name="CaptureThread")
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        # Start processing thread  
        self.processing_thread = threading.Thread(target=self._processing_loop, name="ProcessingThread")
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # Start cropping thread
        self.cropping_thread = threading.Thread(target=self._cropping_loop, name="CroppingThread")
        self.cropping_thread.daemon = True
        self.cropping_thread.start()
        
        source_type = "RTSP stream" if self.is_rtsp else "camera"
        print(f"Started SelectiveFrameProcessor with YOLO Face Detection:")
        print(f"  - Source: {source_type} ({self.source})")
        print(f"  - Capture: Continuous")
        print(f"  - YOLO Processing: Every {self.processing_interval} seconds")
        print(f"  - Face Cropping: {'Active' if self.save_crops else 'Inactive'}")
        print(f"  - Display size: {self.display_width}x{self.display_height}")
        print(f"  - Confidence threshold: {self.conf_threshold}")
        if self.save_crops:
            print(f"  - Saving crops to: {self.current_crop_dir}")
            print(f"  - Max crops: {self.max_crops_per_session}")
            print(f"  - Crop quality: {self.crop_quality}%")
        
    def stop(self):
        """Stop all threads and release resources"""
        self.running = False
        
        # Close all crop preview windows
        for window_name in list(self.crop_windows):
            try:
                cv.destroyWindow(window_name)
            except:
                pass
        self.crop_windows.clear()
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
            
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
            
        if self.cropping_thread and self.cropping_thread.is_alive():
            self.cropping_thread.join(timeout=2.0)
            
        self.capture.release()
        cv.destroyAllWindows()
        print(f"Stopped SelectiveFrameProcessor. Total detections: {self.detection_count}, Cropped faces: {self.crop_count}, Skipped: {self.skipped_crops}")
        
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
        
    def _run_yolo_detection(self, frame):
        """Run YOLO face detection and return annotated frame + detection data"""
        try:
            results = self.model.predict(
                source=frame,
                conf=self.conf_threshold,
                verbose=False
            )
            
            if results and len(results) > 0:
                result = results[0]
                annotated_frame = frame.copy()
                detection_data = []
                
                if result.boxes is not None:
                    # Get all detection information
                    boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes [x1, y1, x2, y2]
                    confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)  # Class IDs
                    class_names = result.names  # Dictionary mapping class_id to class name
                    
                    # Process each detection individually
                    for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                        x1, y1, x2, y2 = box
                        class_name = class_names[cls_id]
                        
                        # Store detection data for cropping
                        detection_data.append({
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'confidence': conf,
                            'class_name': class_name,
                            'class_id': cls_id
                        })
                        
                        # Custom bbox drawing
                        color = self._get_color_for_class(cls_id)
                        cv.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        
                        # Custom label
                        label = f"{class_name} {conf:.2f}"
                        cv.putText(annotated_frame, label, (int(x1), int(y1)-10), 
                                cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    self.detection_count += len(boxes)
                    
                    # Send detection data to cropping thread
                    if detection_data and not self.crop_queue.full():
                        try:
                            self.crop_queue.put_nowait({
                                'frame': frame.copy(),
                                'detections': detection_data,
                                'timestamp': time.time(),
                                'frame_number': self.frame_counter
                            })
                        except queue.Full:
                            print("Warning: Crop queue full, skipping detection data")
                
                return annotated_frame, len(boxes), detection_data
                
            return frame, 0, []
            
        except Exception as e:
            print(f"YOLO inference error: {e}")
            return frame, 0, []

    def _cropping_loop(self):
        """Thread to process detection results and extract cropped face images"""
        print("Cropping thread started - processing detected faces")
        
        while self.running:
            try:
                # Get detection data from queue with timeout
                detection_data = self.crop_queue.get(timeout=1.0)
                
                frame = detection_data['frame']
                detections = detection_data['detections']
                timestamp = detection_data['timestamp']
                frame_number = detection_data['frame_number']
                
                # Get current crop settings (thread-safe)
                with self.crop_lock:
                    save_crops = self.save_crops
                    max_crops = self.max_crops_per_session
                    enable_preview = self.enable_crop_preview
                    crop_quality = self.crop_quality
                    current_crop_dir = self.current_crop_dir
                
                # Check if we've reached the maximum crop limit
                if save_crops and max_crops > 0 and self.crop_count >= max_crops:
                    self.skipped_crops += len(detections)
                    self.crop_queue.task_done()
                    continue
                
                # Process each detection
                for i, detection in enumerate(detections):
                    bbox = detection['bbox']
                    confidence = detection['confidence']
                    class_name = detection['class_name']
                    
                    # Check crop limit for each detection
                    if save_crops and max_crops > 0 and self.crop_count >= max_crops:
                        self.skipped_crops += 1
                        continue
                    
                    # Extract region from frame
                    x1, y1, x2, y2 = bbox
                    
                    # Ensure coordinates are within frame boundaries
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame.shape[1], x2)
                    y2 = min(frame.shape[0], y2)
                    
                    # Crop the region
                    if x2 > x1 and y2 > y1:  # Valid bounding box
                        cropped_image = frame[y1:y2, x1:x2]
                        
                        if cropped_image.size > 0:  # Non-empty crop
                            self.crop_count += 1
                            
                            # Save cropped image if enabled
                            if save_crops:
                                self._save_cropped_image(cropped_image, frame_number, i, confidence, class_name, crop_quality, current_crop_dir)
                            
                            # Display cropped image in separate window if enabled
                            if enable_preview and len(self.crop_windows) < self.max_preview_windows:
                                self._display_cropped_image(cropped_image, frame_number, i, confidence, class_name)
                            
                # Mark task as done
                self.crop_queue.task_done()
                
            except queue.Empty:
                # No data in queue, continue waiting
                continue
            except Exception as e:
                print(f"Error in cropping thread: {e}")
                continue
                
        print(f"Cropping thread stopped. Total images cropped: {self.crop_count}")
    
    def _save_cropped_image(self, cropped_image, frame_number, detection_index, confidence, class_name, quality, output_dir):
        """Save cropped image to disk"""
        try:
            timestamp = int(time.time() * 1000)
            filename = f"{class_name}_{timestamp}_f{frame_number}_d{detection_index}_c{confidence:.3f}.jpg"
            filepath = os.path.join(output_dir, filename)
            
            # Resize if too large for storage
            max_dimension = 1024
            h, w = cropped_image.shape[:2]
            if max(h, w) > max_dimension:
                scale = max_dimension / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                cropped_image = cv.resize(cropped_image, (new_w, new_h))
            
            # Save with specified quality
            cv.imwrite(filepath, cropped_image, [cv.IMWRITE_JPEG_QUALITY, quality])
            
            # Only print every 10th save to avoid console spam
            if self.crop_count % 10 == 0:
                print(f"Saved cropped {class_name}: {filename} (total: {self.crop_count})")
            
        except Exception as e:
            print(f"Error saving cropped image: {e}")
    
    def _display_cropped_image(self, cropped_image, frame_number, detection_index, confidence, class_name):
        """Display cropped image in a separate window"""
        try:
            # Resize for consistent display
            display_image = cropped_image.copy()
            max_display_size = 400
            h, w = display_image.shape[:2]
            
            if max(h, w) > max_display_size:
                scale = max_display_size / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                display_image = cv.resize(display_image, (new_w, new_h))
            
            # Add info overlay
            cv.putText(display_image, f"Frame: {frame_number}", (10, 20), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv.putText(display_image, f"Class: {class_name}", (10, 40), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv.putText(display_image, f"Conf: {confidence:.3f}", (10, 60), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv.putText(display_image, f"Total: {self.crop_count}", (10, 80), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            window_name = f"{class_name} {detection_index} - Frame {frame_number}"
            cv.imshow(window_name, display_image)
            self.crop_windows.add(window_name)
            
        except Exception as e:
            print(f"Error displaying cropped image: {e}")

    def _get_color_for_class(self, class_id):
        """Generate consistent color for each class"""
        colors = [(0,255,0), (255,0,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255)]
        return colors[class_id % len(colors)]   
   
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
                    
                    # Run YOLO object detection
                    processed_frame, detections, detection_data = self._run_yolo_detection(frame_to_process)
                    
                    # Resize frame to smaller display size
                    resized_frame = self._resize_frame(processed_frame)
                    
                    # Add informational overlay with detection info
                    self._add_info_overlay(resized_frame, frame_num, frames_processed, detections)
                    
                    # Display the resized frame with detections
                    cv.imshow("YOLO Detection - Selective Processing", resized_frame)
                    
                    # Handle key presses for dynamic control
                    key = cv.waitKey(1) & 0xFF
                    if key == 27:  # ESC key
                        self.running = False
                        break
                    elif key == ord('s'):  # Toggle crop saving
                        self.toggle_crop_saving()
                    elif key == ord('p'):  # Toggle preview
                        self.toggle_crop_preview()
                    elif key == ord('c'):  # Change crop directory
                        self._change_crop_directory()
                    elif key == ord('+'):  # Increase crop quality
                        self.adjust_crop_quality(5)
                    elif key == ord('-'):  # Decrease crop quality
                        self.adjust_crop_quality(-5)
                    elif key == ord('m'):  # Show crop settings
                        self.show_crop_settings()
                
                last_processing_time = current_time
                
            time.sleep(0.001)
                
        print(f"YOLO processing thread stopped. Total frames processed: {frames_processed}")
    
    def _resize_frame(self, frame):
        """Resize frame to display dimensions maintaining aspect ratio"""
        return cv.resize(frame, (self.display_width, self.display_height))
    
    def _add_info_overlay(self, frame, frame_num, processed_count, detections_count):
        """Add informational text overlay to the frame with detection info"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        source_type = "RTSP" if self.is_rtsp else "Camera"
        
        # Get current crop settings (thread-safe)
        with self.crop_lock:
            save_crops = self.save_crops
            enable_preview = self.enable_crop_preview
            crop_quality = self.crop_quality
            max_crops = self.max_crops_per_session
        
        # Scale font size based on display width
        font_scale = 0.5 if self.display_width <= 640 else 0.7
        thickness = 1 if self.display_width <= 640 else 2
        
        y_offset = 25
        line_height = 20
        
        # Add different colored text for better visibility
        texts = [
            (f"Source: {source_type}", (0, 255, 255)),
            (f"Frame: {frame_num}", (0, 255, 0)),
            (f"Processed: {processed_count}", (255, 255, 0)),
            (f"Detections: {detections_count}", (255, 0, 255)),
            (f"Total Detections: {self.detection_count}", (255, 0, 255)),
            (f"Images Cropped: {self.crop_count}", (255, 0, 255)),
            (f"Crop Saving: {'ON' if save_crops else 'OFF'}", (0, 255, 0) if save_crops else (0, 0, 255)),
            (f"Preview: {'ON' if enable_preview else 'OFF'}", (0, 255, 0) if enable_preview else (0, 0, 255)),
            (f"Quality: {crop_quality}%", (255, 255, 0)),
            (f"Max Crops: {max_crops if max_crops > 0 else 'Unlimited'}", (255, 255, 0)),
        ]
        
        if self.skipped_crops > 0:
            texts.append((f"Skipped Crops: {self.skipped_crops}", (255, 0, 0)))
        
        # Add controls help (smaller font)
        help_texts = [
            "Controls: ESC=Exit, s=Toggle Saving, p=Toggle Preview",
            "c=Change Dir, +/-=Quality, m=Show Settings"
        ]
        
        for text, color in texts:
            cv.putText(frame, text, (10, y_offset), 
                      cv.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
            y_offset += line_height
        
        # Add help texts
        help_font_scale = font_scale - 0.1
        for help_text in help_texts:
            cv.putText(frame, help_text, (10, y_offset), 
                      cv.FONT_HERSHEY_SIMPLEX, help_font_scale, (255, 255, 255), 1)
            y_offset += 15
    
    # Dynamic Crop Control Methods
    
    def toggle_crop_saving(self):
        """Toggle crop saving on/off"""
        with self.crop_lock:
            self.save_crops = not self.save_crops
            status = "ENABLED" if self.save_crops else "DISABLED"
            print(f"Crop saving {status}")
    
    def toggle_crop_preview(self):
        """Toggle crop preview windows on/off"""
        with self.crop_lock:
            self.enable_crop_preview = not self.enable_crop_preview
            status = "ENABLED" if self.enable_crop_preview else "DISABLED"
            print(f"Crop preview {status}")
            
            # Close all preview windows if disabling
            if not self.enable_crop_preview:
                for window_name in list(self.crop_windows):
                    try:
                        cv.destroyWindow(window_name)
                    except:
                        pass
                self.crop_windows.clear()
    
    def adjust_crop_quality(self, delta):
        """Adjust crop image quality"""
        with self.crop_lock:
            self.crop_quality = max(1, min(100, self.crop_quality + delta))
            print(f"Crop quality set to {self.crop_quality}%")
    
    def set_max_crops(self, max_crops):
        """Set maximum number of crops to save"""
        with self.crop_lock:
            self.max_crops_per_session = max(0, max_crops)
            print(f"Maximum crops per session set to {self.max_crops_per_session}")
    
    def _change_crop_directory(self):
        """Change crop output directory"""
        try:
            new_dir = input("Enter new crop directory (or press Enter to cancel): ").strip()
            if new_dir:
                with self.crop_lock:
                    self.crop_output_dir = new_dir
                    self.current_crop_dir = self._get_current_crop_directory()
                    os.makedirs(self.current_crop_dir, exist_ok=True)
                    print(f"Crop directory changed to: {self.current_crop_dir}")
        except Exception as e:
            print(f"Error changing crop directory: {e}")
    
    def show_crop_settings(self):
        """Display current crop settings"""
        with self.crop_lock:
            settings = {
                'save_crops': self.save_crops,
                'crop_output_dir': self.current_crop_dir,
                'max_crops_per_session': self.max_crops_per_session,
                'crop_quality': self.crop_quality,
                'enable_crop_preview': self.enable_crop_preview,
                'auto_create_subdirs': self.auto_create_subdirs,
                'total_crops_saved': self.crop_count,
                'skipped_crops': self.skipped_crops
            }
            
            print("\nCurrent Crop Settings:")
            print("-" * 40)
            for key, value in settings.items():
                print(f"  {key}: {value}")
            print("-" * 40)
    
    def reset_crop_counter(self):
        """Reset crop counter and start new session"""
        with self.crop_lock:
            old_count = self.crop_count
            self.crop_count = 0
            self.skipped_crops = 0
            self.current_crop_dir = self._get_current_crop_directory()
            os.makedirs(self.current_crop_dir, exist_ok=True)
            print(f"Reset crop counter (was {old_count}). New session in: {self.current_crop_dir}")
    
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
        with self.crop_lock:
            crop_settings = {
                'save_crops': self.save_crops,
                'crop_output_dir': self.current_crop_dir,
                'max_crops_per_session': self.max_crops_per_session,
                'crop_quality': self.crop_quality,
                'enable_crop_preview': self.enable_crop_preview,
                'total_crops_saved': self.crop_count,
                'skipped_crops': self.skipped_crops
            }
        
        return {
            'width': self.frame_width,
            'height': self.frame_height,
            'fps': self.actual_fps,
            'source_type': 'RTSP' if self.is_rtsp else 'Camera',
            'display_size': f"{self.display_width}x{self.display_height}",
            'model': str(self.model_path),
            'confidence_threshold': self.conf_threshold,
            'total_detections': self.detection_count,
            'crop_settings': crop_settings
        }


def main():
    """
    Demonstration of the SelectiveFrameProcessor with dynamic crop saving
    """
    print("Selective Frame Processing with Dynamic Crop Saving")
    print("=" * 60)
    print("Features:")
    print("- Camera & RTSP support")
    print("- Multi-threaded architecture (Capture, Processing, Cropping)")
    print("- Selective frame sampling for CPU efficiency") 
    print("- YOLO object detection integration")
    print("- Dynamic crop saving with runtime controls")
    print("- Resizable display output")
    print("- Real-time performance monitoring")
    print("\nControls:")
    print("  ESC: Exit")
    print("  s: Toggle crop saving on/off")
    print("  p: Toggle crop preview windows on/off") 
    print("  c: Change crop output directory")
    print("  +: Increase crop quality")
    print("  -: Decrease crop quality")
    print("  m: Show current crop settings")
    print("=" * 60)
    
    # Choose source type
    while True:
        choice = input("Choose source type:\n1. Camera\n2. RTSP Stream\nEnter choice (1 or 2): ").strip()
        
        if choice == '1':
            camera_index = int(input("Enter camera index (default 0): ") or "0")
            display_width = int(input("Enter display width (default 640): ") or "640")
            processing_interval = float(input("Enter processing interval in seconds (default 0.5): ") or "0.5")
            model_path = input("Enter YOLO model path (or press Enter for pretrained model): ").strip()
            
            # Crop settings
            save_crops = input("Enable crop saving initially? (y/n, default y): ").strip().lower() != 'n'
            max_crops = int(input("Max crops to save (0 for unlimited, default 1000): ") or "1000")
            crop_quality = int(input("Crop image quality (1-100, default 95): ") or "95")
            enable_preview = input("Enable crop preview windows? (y/n, default y): ").strip().lower() != 'n'
            
            processor = SelectiveFrameProcessor(
                source=camera_index,
                fps=30,
                processing_interval=processing_interval,
                is_rtsp=False,
                display_width=display_width,
                model_path=model_path,
                save_crops=save_crops,
                max_crops_per_session=max_crops,
                crop_quality=crop_quality,
                enable_crop_preview=enable_preview
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
            
            # Crop settings
            save_crops = input("Enable crop saving initially? (y/n, default y): ").strip().lower() != 'n'
            max_crops = int(input("Max crops to save (0 for unlimited, default 1000): ") or "1000")
            crop_quality = int(input("Crop image quality (1-100, default 95): ") or "95")
            enable_preview = input("Enable crop preview windows? (y/n, default y): ").strip().lower() != 'n'
            
            processor = SelectiveFrameProcessor(
                source=rtsp_url,
                processing_interval=processing_interval,
                is_rtsp=True,
                display_width=display_width,
                model_path=model_path,
                save_crops=save_crops,
                max_crops_per_session=max_crops,
                crop_quality=crop_quality,
                enable_crop_preview=enable_preview
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