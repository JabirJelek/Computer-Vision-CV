# Python 2/3 compatibility
from __future__ import print_function

import os
import sys
import time
import threading
import numpy as np
import cv2 as cv

class SelectiveFrameProcessor:
    """
    A two-thread system for efficient frame capture only:
    - Capture Thread: Continuously captures frames, keeping only the latest
    - Processing Thread: Samples frames at fixed intervals for display
    Supports both camera devices and RTSP streams
    """
    
    def __init__(self, source=0, fps=30, processing_interval=0.5, is_rtsp=False, display_width=640):
        """
        Args:
            source: Camera device index (int) or RTSP URL (string)
            fps: Target frames per second for camera (ignored for RTSP)
            processing_interval: Time in seconds between processing frames
            is_rtsp: Boolean indicating if source is an RTSP stream
            display_width: Width for resizing display frame (maintains aspect ratio)
        """
        self.source = source
        self.is_rtsp = is_rtsp
        self.processing_interval = processing_interval
        self.display_width = display_width
        
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
        print(f"Started SelectiveFrameProcessor:")
        print(f"  - Source: {source_type} ({self.source})")
        print(f"  - Capture: Continuous")
        print(f"  - Processing: Every {self.processing_interval} seconds")
        print(f"  - Display size: {self.display_width}x{self.display_height}")
        
    def stop(self):
        """Stop both threads and release resources"""
        self.running = False
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
            
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
            
        self.capture.release()
        cv.destroyAllWindows()
        print("Stopped SelectiveFrameProcessor")
        
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
        """Display frames at fixed time intervals"""
        print("Processing thread started - sampling frames at fixed intervals")
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
                    
                    # Resize frame to smaller display size
                    resized_frame = self._resize_frame(frame_to_process)
                    
                    # Add informational overlay
                    self._add_info_overlay(resized_frame, frame_num, frames_processed)
                    
                    # Display the resized frame
                    cv.imshow("Video Stream - Selective Processing", resized_frame)
                    
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
    
    def _add_info_overlay(self, frame, frame_num, processed_count):
        """Add informational text overlay to the frame"""
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
        cv.putText(frame, f"Time: {timestamp}", (10, 85), 
                  cv.FONT_HERSHEY_SIMPLEX, font_scale-0.1, (255, 255, 255), 1)
        cv.putText(frame, f"Interval: {self.processing_interval}s", (10, 100), 
                  cv.FONT_HERSHEY_SIMPLEX, font_scale-0.1, (255, 255, 255), 1)
        cv.putText(frame, f"Display: {self.display_width}x{self.display_height}", (10, 115), 
                  cv.FONT_HERSHEY_SIMPLEX, font_scale-0.1, (255, 255, 255), 1)
        cv.putText(frame, "Press ESC to exit", (10, 130), 
                  cv.FONT_HERSHEY_SIMPLEX, font_scale-0.1, (255, 255, 255), 1)
    
    def set_processing_interval(self, interval):
        """Dynamically change the processing interval"""
        self.processing_interval = max(0.01, interval)
        print(f"Processing interval changed to {self.processing_interval} seconds")
    
    def set_display_size(self, width):
        """Dynamically change the display size"""
        self.display_width = max(160, width)  # Minimum 160px width
        self.display_height = int((self.display_width / self.frame_width) * self.frame_height)
        print(f"Display size changed to {self.display_width}x{self.display_height}")
    
    def get_video_properties(self):
        """Get current video stream properties"""
        return {
            'width': self.frame_width,
            'height': self.frame_height,
            'fps': self.actual_fps,
            'source_type': 'RTSP' if self.is_rtsp else 'Camera',
            'display_size': f"{self.display_width}x{self.display_height}"
        }


def main():
    """
    Demonstration of the SelectiveFrameProcessor - Clean Video Capture Only
    """
    print("Selective Frame Processing - Clean Video Capture")
    print("=" * 60)
    print("Features:")
    print("- Camera & RTSP support")
    print("- Multi-threaded architecture")
    print("- Selective frame sampling for CPU efficiency")
    print("- Resizable display output")
    print("- Real-time performance monitoring")
    print("\nControls:")
    print("  ESC: Exit")
    print("=" * 60)
    
    # Choose source type
    while True:
        choice = input("Choose source type:\n1. Camera\n2. RTSP Stream\nEnter choice (1 or 2): ").strip()
        
        if choice == '1':
            camera_index = int(input("Enter camera index (default 0): ") or "0")
            display_width = int(input("Enter display width (default 640): ") or "640")
            processor = SelectiveFrameProcessor(
                source=camera_index,
                fps=30,
                processing_interval=0.5,
                is_rtsp=False,
                display_width=display_width
            )
            break
        elif choice == '2':
            rtsp_url = input("Enter RTSP URL: ").strip()
            if not rtsp_url:
                rtsp_url = "rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4"
                print(f"Using demo URL: {rtsp_url}")
            
            display_width = int(input("Enter display width (default 640): ") or "640")
            processor = SelectiveFrameProcessor(
                source=rtsp_url,
                processing_interval=1.0,
                is_rtsp=True,
                display_width=display_width
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