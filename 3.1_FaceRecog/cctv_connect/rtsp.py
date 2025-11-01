import cv2
import time
import numpy as np
from datetime import datetime
import math

class RTSPAnalyzer:
    def __init__(self):
        self.frame_count = 0
        self.start_time = time.time()
        self.last_frame_time = time.time()
        self.frame_times = []
        self.dropped_frames = 0
        self.total_frames = 0
        self.reconnection_count = 0
        self.quality_metrics = []
        self.resolution_history = []
        self.bitrate_history = []
        self.frame_size_history = []
        
    def calculate_frame_size(self, frame):
        """Calculate the size of the current frame in bytes"""
        if frame is None:
            return 0
        return frame.nbytes
    
    def calculate_quality_metrics(self, frame):
        """Calculate various quality metrics for the frame"""
        if frame is None:
            return {"blur": 0, "brightness": 0, "contrast": 0, "resolution": "0x0"}
        
        # Get resolution
        height, width = frame.shape[:2]
        resolution = f"{width}x{height}"
        
        # Calculate blur using Laplacian variance
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Calculate brightness (mean pixel value)
        brightness = np.mean(gray)
        
        # Calculate contrast (standard deviation)
        contrast = np.std(gray)
        
        return {
            "blur": blur,
            "brightness": brightness,
            "contrast": contrast,
            "resolution": resolution,
            "width": width,
            "height": height
        }
    
    def calculate_connection_stability(self):
        """Calculate connection stability based on frame timing with improved formula"""
        if len(self.frame_times) < 5:  # Need minimum frames for accurate stability
            return 0.8  # Default stability until we have enough data
        
        # Calculate jitter (coefficient of variation of frame intervals)
        intervals = np.diff(self.frame_times)
        if len(intervals) < 2:
            return 0.8
            
        avg_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        
        # Use coefficient of variation instead of raw std for better normalization
        jitter_cv = std_interval / avg_interval if avg_interval > 0 else 1.0
        
        # Calculate frame drop rate (more nuanced calculation)
        expected_frames = (self.frame_times[-1] - self.frame_times[0]) * 30  # Assuming 30fps target
        actual_frames = len(self.frame_times)
        drop_rate = max(0, (expected_frames - actual_frames) / expected_frames) if expected_frames > 0 else 0
        
        # Improved stability formula with better weighting
        stability = max(0, 1.0 - (jitter_cv * 2 + drop_rate * 3 + (self.reconnection_count * 0.1)))
        
        return min(1.0, stability)
    
    def calculate_bitrate(self, frame_size, time_interval):
        """Calculate approximate bitrate with corrected formula"""
        if time_interval == 0 or frame_size == 0:
            return 0
        
        # Original formula was correct but let's verify and add bounds
        # frame_size in bytes, time_interval in seconds
        # bitrate in kbps = (bytes * 8 bits/byte) / (1000 bits/kbit) / time_interval
        bitrate_kbps = (frame_size * 8) / (time_interval * 1000)
        
        # Add sanity checks for realistic bitrate values
        if bitrate_kbps > 50000:  # Unrealistically high bitrate
            # This might indicate calculation error, let's recalculate more carefully
            print(f"Warning: Unrealistic bitrate detected: {bitrate_kbps:.1f} kbps")
            # Alternative calculation using frame dimensions
            if hasattr(self, 'last_frame_dims'):
                width, height = self.last_frame_dims
                # Rough estimate: 0.1 bits per pixel for compressed video
                estimated_size = (width * height * 0.1) / 8  # bytes
                bitrate_kbps = (estimated_size * 8) / (time_interval * 1000)
        
        return bitrate_kbps
    
    def calculate_compression_ratio(self, frame):
        """Calculate compression ratio based on frame dimensions"""
        if frame is None:
            return 0
            
        height, width = frame.shape[:2]
        # Theoretical uncompressed frame size (RGB, 3 bytes per pixel)
        uncompressed_size = width * height * 3
        
        # Actual frame size from OpenCV
        actual_size = frame.nbytes
        
        if uncompressed_size > 0:
            compression_ratio = uncompressed_size / actual_size
            return compression_ratio
        return 1.0
    
    def update_metrics(self, frame):
        """Update all metrics with new frame"""
        current_time = time.time()
        
        # Frame timing
        frame_interval = current_time - self.last_frame_time
        self.frame_times.append(current_time)
        self.last_frame_time = current_time
        
        # Keep only recent frame times (last 30 seconds for better stability calculation)
        self.frame_times = [t for t in self.frame_times if current_time - t < 30]
        
        # Frame counting
        self.frame_count += 1
        self.total_frames += 1
        
        # Store frame dimensions for bitrate verification
        if frame is not None:
            height, width = frame.shape[:2]
            self.last_frame_dims = (width, height)
        
        # Calculate metrics
        frame_size = self.calculate_frame_size(frame)
        quality = self.calculate_quality_metrics(frame)
        compression_ratio = self.calculate_compression_ratio(frame)
        
        self.quality_metrics.append(quality)
        self.frame_size_history.append(frame_size)
        
        # Keep only recent metrics
        self.quality_metrics = self.quality_metrics[-100:]
        self.frame_size_history = self.frame_size_history[-100:]
        
        # Calculate bitrate with additional verification
        bitrate_kbps = self.calculate_bitrate(frame_size, frame_interval)
        self.bitrate_history.append(bitrate_kbps)
        self.bitrate_history = self.bitrate_history[-50:]  # Keep last 50 bitrate readings
        
        # Calculate average bitrate for more stable reading
        avg_bitrate = np.mean(self.bitrate_history) if self.bitrate_history else bitrate_kbps
        
        return {
            "frame_size_bytes": frame_size,
            "frame_size_mb": frame_size / (1024 * 1024),
            "quality": quality,
            "frame_interval": frame_interval,
            "current_fps": 1.0 / frame_interval if frame_interval > 0 else 0,
            "bitrate_kbps": bitrate_kbps,
            "avg_bitrate_kbps": avg_bitrate,
            "compression_ratio": compression_ratio,
            "resolution": quality.get("resolution", "Unknown")
        }
    
    def get_summary_stats(self):
        """Get comprehensive connection statistics"""
        if not self.quality_metrics:
            return {}
        
        # Calculate average quality metrics
        avg_blur = np.mean([q["blur"] for q in self.quality_metrics])
        avg_brightness = np.mean([q["brightness"] for q in self.quality_metrics])
        avg_contrast = np.mean([q["contrast"] for q in self.quality_metrics])
        
        # Calculate overall FPS
        total_time = time.time() - self.start_time
        overall_fps = self.frame_count / total_time if total_time > 0 else 0
        
        # Calculate average frame size and bitrate
        avg_frame_size = np.mean(self.frame_size_history) if self.frame_size_history else 0
        avg_bitrate = np.mean(self.bitrate_history) if self.bitrate_history else 0
        
        # Get most common resolution
        resolutions = [q.get("resolution", "Unknown") for q in self.quality_metrics]
        if resolutions:
            common_resolution = max(set(resolutions), key=resolutions.count)
        else:
            common_resolution = "Unknown"
        
        return {
            "total_frames_processed": self.frame_count,
            "dropped_frames": self.dropped_frames,
            "drop_rate_percent": (self.dropped_frames / max(1, self.total_frames)) * 100,
            "reconnection_count": self.reconnection_count,
            "overall_fps": overall_fps,
            "connection_stability": self.calculate_connection_stability(),
            "average_quality": {
                "blur": avg_blur,
                "brightness": avg_brightness,
                "contrast": avg_contrast
            },
            "average_frame_size_mb": avg_frame_size / (1024 * 1024),
            "average_bitrate_kbps": avg_bitrate,
            "common_resolution": common_resolution,
            "analysis_duration_seconds": total_time
        }

def display_metrics(frame, current_metrics, summary_stats, analyzer):
    """Display metrics on the video frame"""
    overlay = frame.copy()
    height, width = frame.shape[:2]
    
    # Create semi-transparent background for text
    overlay = cv2.rectangle(overlay, (10, 10), (450, 280), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    
    # Text parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (0, 255, 0)  # Green
    thickness = 1
    y_offset = 25
    line_height = 20
    
    # Color coding for values
    def get_color(value, thresholds):
        if value >= thresholds[0]:
            return (0, 255, 0)  # Green - good
        elif value >= thresholds[1]:
            return (0, 255, 255)  # Yellow - warning
        else:
            return (0, 0, 255)  # Red - bad
    
    stability_color = get_color(summary_stats['connection_stability'], [0.7, 0.4])
    blur_color = get_color(current_metrics['quality']['blur'], [100, 50])
    contrast_color = get_color(current_metrics['quality']['contrast'], [30, 15])
    
    metrics_text = [
        f"FPS: {current_metrics['current_fps']:.1f}",
        f"Resolution: {current_metrics['resolution']}",
        f"Frame Size: {current_metrics['frame_size_mb']:.2f} MB",
        f"Bitrate: {current_metrics['avg_bitrate_kbps']:.1f} kbps",
        f"Compression: {current_metrics['compression_ratio']:.1f}x",
        f"Stability: {analyzer.calculate_connection_stability():.3f}",
        f"Blur: {current_metrics['quality']['blur']:.1f}",
        f"Brightness: {current_metrics['quality']['brightness']:.1f}",
        f"Contrast: {current_metrics['quality']['contrast']:.1f}",
        f"Dropped: {analyzer.dropped_frames} ({summary_stats['drop_rate_percent']:.1f}%)",
        f"Reconnects: {analyzer.reconnection_count}",
        f"Total Frames: {analyzer.total_frames}"
    ]
    
    colors = [
        font_color,  # FPS
        font_color,  # Resolution
        font_color,  # Frame Size
        font_color,  # Bitrate
        font_color,  # Compression
        stability_color,  # Stability
        blur_color,  # Blur
        font_color,  # Brightness
        contrast_color,  # Contrast
        font_color,  # Dropped
        font_color,  # Reconnects
        font_color   # Total Frames
    ]
    
    for i, (text, color) in enumerate(zip(metrics_text, colors)):
        y_pos = y_offset + (i * line_height)
        cv2.putText(frame, text, (15, y_pos), font, font_scale, color, thickness)
    
    return frame

def print_periodic_summary(summary_stats):
    """Print periodic summary to console"""
    print("\n" + "="*60)
    print("RTSP STREAM ANALYSIS SUMMARY")
    print("="*60)
    print(f"Analysis Duration: {summary_stats['analysis_duration_seconds']:.1f}s")
    print(f"Total Frames Processed: {summary_stats['total_frames_processed']}")
    print(f"Common Resolution: {summary_stats['common_resolution']}")
    print(f"Dropped Frames: {summary_stats['dropped_frames']} ({summary_stats['drop_rate_percent']:.1f}%)")
    print(f"Reconnection Events: {summary_stats['reconnection_count']}")
    print(f"Overall FPS: {summary_stats['overall_fps']:.1f}")
    print(f"Connection Stability: {summary_stats['connection_stability']:.3f}")
    print(f"Average Frame Size: {summary_stats['average_frame_size_mb']:.2f} MB")
    print(f"Average Bitrate: {summary_stats['average_bitrate_kbps']:.1f} kbps")
    print(f"Average Quality - Blur: {summary_stats['average_quality']['blur']:.1f}, "
          f"Brightness: {summary_stats['average_quality']['brightness']:.1f}, "
          f"Contrast: {summary_stats['average_quality']['contrast']:.1f}")
    
    # Add quality assessment
    print("\nQUALITY ASSESSMENT:")
    if summary_stats['connection_stability'] >= 0.7:
        print("✓ Connection Stability: GOOD")
    elif summary_stats['connection_stability'] >= 0.4:
        print("⚠ Connection Stability: FAIR")
    else:
        print("✗ Connection Stability: POOR")
        
    if summary_stats['average_quality']['blur'] >= 100:
        print("✓ Image Sharpness: GOOD")
    elif summary_stats['average_quality']['blur'] >= 50:
        print("⚠ Image Sharpness: FAIR")
    else:
        print("✗ Image Sharpness: POOR")
        
    if summary_stats['average_quality']['contrast'] >= 30:
        print("✓ Image Contrast: GOOD")
    elif summary_stats['average_quality']['contrast'] >= 15:
        print("⚠ Image Contrast: FAIR")
    else:
        print("✗ Image Contrast: POOR")
    
    print("="*60)

def main():
    rtsp_url = input("Enter the RTSP URL: ")

    if not rtsp_url.startswith('rtsp://'):
        print("That doesn't look like a valid RTSP URL. Please try again.")
        exit()

    # Initialize analyzer and video capture
    analyzer = RTSPAnalyzer()
    
    # Try different OpenCV backends for better compatibility
    backends = [
        cv2.CAP_FFMPEG,
        cv2.CAP_ANY
    ]
    
    cap = None
    for backend in backends:
        cap = cv2.VideoCapture(rtsp_url, backend)
        if cap.isOpened():
            print(f"Connected using backend: {backend}")
            break
        else:
            if cap:
                cap.release()
    
    if not cap or not cap.isOpened():
        print("Error: Could not connect to RTSP stream with any backend")
        exit()

    # Optimize OpenCV settings
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Increase buffer for stability
    cap.set(cv2.CAP_PROP_FPS, 30)        # Set expected FPS

    print("Successfully connected to RTSP stream")
    print("Starting stream analysis...")
    print("Press 'q' to quit, 'r' to reset statistics, 's' to show summary")

    last_summary_time = time.time()
    reset_requested = False

    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            current_time = time.time()
            
            if not ret:
                print("Frame read error, attempting to reconnect...")
                analyzer.dropped_frames += 1
                analyzer.reconnection_count += 1
                cap.release()
                time.sleep(2)
                # Try to reconnect with different backends
                for backend in backends:
                    cap = cv2.VideoCapture(rtsp_url, backend)
                    if cap.isOpened():
                        print(f"Reconnected using backend: {backend}")
                        break
                    else:
                        if cap:
                            cap.release()
                continue
            
            # Update metrics
            current_metrics = analyzer.update_metrics(frame)
            
            # Process frame (resize for better performance if needed)
            frame = cv2.resize(frame, (1024, 576))
            
            # Get current summary stats
            summary_stats = analyzer.get_summary_stats()
            
            # Display metrics on frame
            frame_with_metrics = display_metrics(frame, current_metrics, summary_stats, analyzer)
            
            # Display frame
            cv2.imshow('RTSP Stream Analysis', frame_with_metrics)
            
            # Print periodic summary every 30 seconds
            if current_time - last_summary_time >= 30:
                print_periodic_summary(summary_stats)
                last_summary_time = current_time
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                print("Resetting statistics...")
                analyzer = RTSPAnalyzer()
                reset_requested = True
            elif key == ord('s'):
                print_periodic_summary(summary_stats)
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        # Final summary
        final_summary = analyzer.get_summary_stats()
        print("\nFINAL SUMMARY:")
        print_periodic_summary(final_summary)
        
        if cap:
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()