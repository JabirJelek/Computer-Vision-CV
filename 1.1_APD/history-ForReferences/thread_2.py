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
    A two-thread system for efficient frame processing:
    - Capture Thread: Continuously captures frames, keeping only the latest
    - Processing Thread: Samples frames at fixed intervals for processing
    Supports both camera devices and RTSP streams
    """
    
    def __init__(self, source=0, fps=30, processing_interval=0.5, is_rtsp=False):
        """
        Args:
            source: Camera device index (int) or RTSP URL (string)
            fps: Target frames per second for camera (ignored for RTSP)
            processing_interval: Time in seconds between processing frames
            is_rtsp: Boolean indicating if source is an RTSP stream
        """
        self.source = source
        self.is_rtsp = is_rtsp
        self.processing_interval = processing_interval
        
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
        self.capture_thread.daemon = True  # Daemon thread for cleaner shutdown
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
                    time.sleep(0.1)  # Brief pause before retry
                continue
                
            # Reset failure counter on successful capture
            self.capture_failures = 0
            frames_captured += 1
            
            # Store only the latest frame - previous frames are automatically dropped
            with self.lock:
                self.latest_frame = frame.copy() if frame is not None else None
                self.frame_counter = frames_captured
                
        print(f"Capture thread stopped. Total frames captured: {frames_captured}")
        
    def _reconnect_rtsp(self):
        """Attempt to reconnect to RTSP stream"""
        print("Attempting RTSP reconnection...")
        self.capture.release()
        time.sleep(2)  # Wait before reconnecting
        
        # Reinitialize capture with same RTSP URL
        self.capture = cv.VideoCapture(self.source)
        self.capture.set(cv.CAP_PROP_BUFFERSIZE, 1)
        self.capture.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'H264'))
        
        if self.capture.isOpened():
            print("RTSP reconnection successful")
        else:
            print("RTSP reconnection failed")
        
    def _processing_loop(self):
        """Process frames at fixed time intervals"""
        print("Processing thread started - sampling frames at fixed intervals")
        frames_processed = 0
        last_processing_time = time.time()
        
        while self.running:
            current_time = time.time()
            elapsed = current_time - last_processing_time
            
            # Check if it's time to process a frame
            if elapsed >= self.processing_interval:
                frame_to_process = None
                frame_num = 0
                
                # Safely get the latest frame
                with self.lock:
                    if self.latest_frame is not None:
                        frame_to_process = self.latest_frame.copy()
                        frame_num = self.frame_counter
                
                if frame_to_process is not None:
                    frames_processed += 1
                    
                    # Process the frame (this is where your processing logic goes)
                    self._process_frame(frame_to_process, frame_num, frames_processed)
                    
                    # Display the processed frame
                    cv.imshow("Processed Frames", frame_to_process)
                    
                    # Handle key presses (ESC to exit)
                    key = cv.waitKey(1) & 0xFF
                    if key == 27:  # ESC key
                        self.running = False
                        break
                
                last_processing_time = current_time
                
            # Small sleep to prevent busy waiting
            time.sleep(0.001)
                
        print(f"Processing thread stopped. Total frames processed: {frames_processed}")
        
    def _process_frame(self, frame, frame_num, processed_count):
        """
        Example processing function - replace with your actual processing logic
        """
        # Simulate some processing work
        processing_start = time.time()
        
        # Example: Add timestamp and frame info to the frame
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        source_type = "RTSP" if self.is_rtsp else "Camera"
        
        cv.putText(frame, f"Source: {source_type}", (10, 30), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv.putText(frame, f"Frame: {frame_num}", (10, 60), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv.putText(frame, f"Processed: {processed_count}", (10, 90), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv.putText(frame, f"Time: {timestamp}", (10, 120), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv.putText(frame, f"Interval: {self.processing_interval}s", (10, 150), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Simulate processing time (remove this in production)
        processing_time = 0.05  # 50ms simulated processing
        time.sleep(processing_time)
        
        processing_end = time.time()
        actual_processing_time = processing_end - processing_start
        
        print(f"Processed frame {frame_num} (#{processed_count}) - "
              f"Processing time: {actual_processing_time:.3f}s")
        
    def set_processing_interval(self, interval):
        """Dynamically change the processing interval"""
        self.processing_interval = max(0.01, interval)  # Minimum 10ms
        print(f"Processing interval changed to {self.processing_interval} seconds")


def main():
    """
    Demonstration of the SelectiveFrameProcessor with camera or RTSP
    """
    print("Selective Frame Processing Demo - Camera & RTSP")
    print("=" * 50)
    print("This system supports:")
    print("- Camera devices (integer index)")
    print("- RTSP streams (URL)")
    print("- Continuous frame capture in background thread")
    print("- Selective frame processing at fixed intervals") 
    print("- Automatic dropping of unnecessary frames")
    print("- Reduced CPU usage compared to processing every frame")
    print("\nUsage examples:")
    print("  Camera: processor = SelectiveFrameProcessor(0)")
    print("  RTSP:   processor = SelectiveFrameProcessor('rtsp://url', is_rtsp=True)")
    print("\nPress ESC in the video window to exit")
    print("=" * 50)
    
    # Choose source type
    while True:
        choice = input("Choose source type:\n1. Camera\n2. RTSP Stream\nEnter choice (1 or 2): ").strip()
        
        if choice == '1':
            camera_index = int(input("Enter camera index (default 0): ") or "0")
            processor = SelectiveFrameProcessor(
                source=camera_index,
                fps=30,
                processing_interval=0.5,
                is_rtsp=False
            )
            break
        elif choice == '2':
            rtsp_url = input("Enter RTSP URL: ").strip()
            if not rtsp_url:
                # Example RTSP URL for testing
                rtsp_url = "rtsp://184.72.239.149/vod/mp4:BigBuckBunny_175k.mov"
                print(f"Using demo URL: {rtsp_url}")
            
            processor = SelectiveFrameProcessor(
                source=rtsp_url,
                processing_interval=0.5,
                is_rtsp=True
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


# Example usage patterns:
"""
# Camera usage
camera_processor = SelectiveFrameProcessor(
    source=0,                    # Camera index
    fps=30,
    processing_interval=0.5,     # Process every 500ms
    is_rtsp=False
)

# RTSP usage  
rtsp_processor = SelectiveFrameProcessor(
    source="rtsp://username:password@ip:port/stream",
    processing_interval=1.0,     # Process every second
    is_rtsp=True
)

# RTSP with demo stream (public test stream)
demo_processor = SelectiveFrameProcessor(
    source="rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4",
    processing_interval=0.5,
    is_rtsp=True
)
"""

if __name__ == '__main__':
    main()