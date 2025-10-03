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
    """
    
    def __init__(self, camera_index=0, fps=30, processing_interval=0.5):
        """
        Args:
            camera_index: Camera device index
            fps: Target frames per second for camera
            processing_interval: Time in seconds between processing frames
        """
        self.capture = cv.VideoCapture(camera_index)
        if not self.capture.isOpened():
            raise RuntimeError("Could not open camera")
            
        self.capture.set(cv.CAP_PROP_FPS, fps)
        self.processing_interval = processing_interval
        
        # Thread synchronization
        self.lock = threading.Lock()
        self.latest_frame = None
        self.frame_counter = 0
        self.running = False
        
        # Threads
        self.capture_thread = None
        self.processing_thread = None
        
    def start(self):
        """Start both capture and processing threads"""
        self.running = True
        
        # Start capture thread
        self.capture_thread = threading.Thread(target=self._capture_loop, name="CaptureThread")
        self.capture_thread.start()
        
        # Start processing thread  
        self.processing_thread = threading.Thread(target=self._processing_loop, name="ProcessingThread")
        self.processing_thread.start()
        
        print(f"Started SelectiveFrameProcessor:")
        print(f"  - Capture: Continuous at camera FPS")
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
        print("Capture thread started - continuously capturing frames")
        frames_captured = 0
        
        while self.running:
            ret, frame = self.capture.read()
            if not ret:
                print("Warning: Failed to capture frame")
                continue
                
            frames_captured += 1
            
            # Store only the latest frame - previous frames are automatically dropped
            with self.lock:
                self.latest_frame = frame.copy() if frame is not None else None
                self.frame_counter = frames_captured
                
        print(f"Capture thread stopped. Total frames captured: {frames_captured}")
        
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
        cv.putText(frame, f"Frame: {frame_num}", (10, 30), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv.putText(frame, f"Processed: {processed_count}", (10, 60), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv.putText(frame, f"Time: {timestamp}", (10, 90), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv.putText(frame, f"Interval: {self.processing_interval}s", (10, 120), 
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
    Demonstration of the SelectiveFrameProcessor
    """
    print("Selective Frame Processing Demo")
    print("=" * 40)
    print("This system demonstrates:")
    print("- Continuous frame capture in background thread")
    print("- Selective frame processing at fixed intervals") 
    print("- Automatic dropping of unnecessary frames")
    print("- Reduced CPU usage compared to processing every frame")
    print("\nPress ESC in the video window to exit")
    print("=" * 40)
    
    # Create processor with 500ms processing interval (2 frames per second)
    processor = SelectiveFrameProcessor(
        camera_index=0,
        fps=30,
        processing_interval=0.5  # Process every 500ms
    )
    
    try:
        processor.start()
        
        # Keep main thread alive while threads run
        while processor.running:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        processor.stop()


"""
# For high-frequency processing (10fps)
processor = SelectiveFrameProcessor(processing_interval=0.1)

# For low-frequency processing (1fps)  
processor = SelectiveFrameProcessor(processing_interval=1.0)

# Change interval dynamically
processor.set_processing_interval(0.2)  # Now 5fps
"""



if __name__ == '__main__':
    main()