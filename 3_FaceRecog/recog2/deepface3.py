import cv2
import os
from deepface import DeepFace
import time
from collections import deque

class DeepFaceStream:
    def __init__(self, db_path=None, camera_index=0, enable_analysis=True, overlay_duration=45):
        """
        Initialize DeepFace stream with real-time analysis capabilities
        
        Args:
            db_path: Path to database for face recognition (optional)
            camera_index: Camera device index
            enable_analysis: Enable facial attribute analysis
            overlay_duration: Number of frames to keep text overlay visible
        """
        self.db_path = db_path
        self.camera_index = camera_index
        self.enable_analysis = enable_analysis
        self.overlay_duration = overlay_duration
        self.cap = None
        self.stream_active = False
        
        # Enhanced overlay persistence
        self.current_analysis = None
        self.analysis_frame_count = 0
        self.last_analysis_time = 0
        self.analysis_history = deque(maxlen=10)  # Keep last 10 analyses
        
    def initialize_camera(self):
        """Initialize camera with optimal settings"""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise Exception(f"Could not open camera with index {self.camera_index}")
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Camera initialized successfully")
        return True
    
    def start_stream(self):
        """Start the DeepFace stream with camera input"""
        if not self.initialize_camera():
            return False
        
        print("Starting DeepFace real-time stream...")
        print("Press 'q' to quit, 'a' to toggle analysis, 's' to save frame")
        print("Press '+' to increase overlay duration, '-' to decrease")
        
        self.stream_active = True
        frame_count = 0
        analysis_enabled = self.enable_analysis
        
        try:
            while self.stream_active:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break
                
                frame_count += 1
                
                # Process frame with DeepFace
                processed_frame = self.process_frame_with_deepface(
                    frame, 
                    analysis_enabled,
                    frame_count
                )
                
                # Display the processed frame
                cv2.imshow('DeepFace Real-Time Stream', processed_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('a'):
                    analysis_enabled = not analysis_enabled
                    status = "enabled" if analysis_enabled else "disabled"
                    print(f"Analysis {status}")
                elif key == ord('s'):
                    self.save_frame(frame)
                elif key == ord('+'):
                    self.overlay_duration = min(300, self.overlay_duration + 15)
                    print(f"Overlay duration increased to {self.overlay_duration} frames")
                elif key == ord('-'):
                    self.overlay_duration = max(15, self.overlay_duration - 15)
                    print(f"Overlay duration decreased to {self.overlay_duration} frames")
                    
        except KeyboardInterrupt:
            print("Stream interrupted by user")
        except Exception as e:
            print(f"Error in stream: {e}")
        finally:
            self.cleanup()
    
    def process_frame_with_deepface(self, frame, analysis_enabled, frame_count):
        """Process frame using DeepFace analysis with persistent overlays"""
        try:
            # Convert BGR to RGB for DeepFace
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Update analysis if enabled and it's time for new analysis
            if analysis_enabled and self.should_analyze_frame(frame_count):
                new_analysis = self.analyze_frame(rgb_frame)
                if new_analysis:
                    self.current_analysis = new_analysis
                    self.analysis_frame_count = frame_count
                    self.last_analysis_time = time.time()
                    self.analysis_history.append({
                        'timestamp': self.last_analysis_time,
                        'analysis': new_analysis,
                        'frame_count': frame_count
                    })
            
            # Always draw the latest analysis if we have one and it's still valid
            if (self.current_analysis is not None and 
                analysis_enabled and 
                self.is_analysis_current(frame_count)):
                frame = self.draw_analysis_results(frame, self.current_analysis, frame_count)
            elif not analysis_enabled:
                # Clear analysis when disabled
                self.current_analysis = None
            
            # Add UI information (always visible)
            frame = self.add_ui_overlay(frame, analysis_enabled, frame_count)
            
            return frame
            
        except Exception as e:
            print(f"Frame processing error: {e}")
            return frame
    
    def should_analyze_frame(self, frame_count):
        """Determine if we should analyze this frame"""
        # Analyze every 15th frame OR if we don't have current analysis
        return (frame_count % 15 == 0 or 
                self.current_analysis is None or 
                not self.is_analysis_current(frame_count))
    
    def is_analysis_current(self, frame_count):
        """Check if current analysis is still within overlay duration"""
        if self.current_analysis is None:
            return False
        frames_since_analysis = frame_count - self.analysis_frame_count
        return frames_since_analysis <= self.overlay_duration
    
    def analyze_frame(self, rgb_frame):
        """Analyze frame using DeepFace"""
        try:
            analysis = DeepFace.analyze(
                rgb_frame,
                actions=['age', 'gender', 'emotion', 'race'],
                detector_backend='opencv',
                enforce_detection=False,
                silent=True
            )
            return analysis
        except Exception as e:
            print(f"DeepFace analysis error: {e}")
            return None
    
    def draw_analysis_results(self, frame, analysis, current_frame_count):
        """Draw analysis results on the frame with persistence indicators"""
        if analysis is None or len(analysis) == 0:
            cv2.putText(frame, "No face detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame
        
        # Calculate how fresh the analysis is
        frames_since_analysis = current_frame_count - self.analysis_frame_count
        freshness_ratio = 1.0 - (frames_since_analysis / self.overlay_duration)
        
        # Adjust color based on freshness (green -> yellow -> red)
        box_color = self.get_freshness_color(freshness_ratio)
        
        # Process each detected face
        for i, result in enumerate(analysis):
            frame = self.draw_single_face_analysis(frame, result, i, box_color, freshness_ratio)
        
        return frame
    
    def get_freshness_color(self, freshness_ratio):
        """Get color based on analysis freshness"""
        if freshness_ratio > 0.7:
            return (0, 255, 0)  # Green - fresh
        elif freshness_ratio > 0.3:
            return (0, 255, 255)  # Yellow - medium
        else:
            return (0, 0, 255)  # Red - stale
    
    def draw_single_face_analysis(self, frame, result, face_index, box_color, freshness_ratio):
        """Draw analysis for a single face with freshness indicator"""
        # Get face region
        region = result.get('region', {})
        x = region.get('x', 0)
        y = region.get('y', 0)
        w = region.get('w', 0)
        h = region.get('h', 0)
        
        # Draw bounding box
        if w > 0 and h > 0:
            # Thicker box for fresher analysis
            thickness = max(1, int(3 * freshness_ratio))
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, thickness)
            
            # Prepare analysis text with freshness indicator
            freshness_text = "🟢" if freshness_ratio > 0.7 else "🟡" if freshness_ratio > 0.3 else "🔴"
            texts = [
                f"{freshness_text} Face {face_index + 1}",
                f"Age: {result.get('age', 'N/A')}",
                f"Gender: {result.get('dominant_gender', 'N/A')}",
                f"Emotion: {result.get('dominant_emotion', 'N/A')}",
                f"Race: {result.get('dominant_race', 'N/A')}",
                f"Freshness: {int(freshness_ratio * 100)}%"
            ]
            
            # Draw text background for better readability
            text_x = x + w + 10 if x + w + 180 < frame.shape[1] else max(10, x - 180)
            text_y = max(30, y)  # Ensure text doesn't go off top of screen
            
            # Semi-transparent background
            overlay = frame.copy()
            bg_height = len(texts) * 25 + 10
            cv2.rectangle(overlay, (text_x - 5, text_y - 5), 
                         (text_x + 220, text_y + bg_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            for i, text in enumerate(texts):
                text_y_pos = text_y + i * 25
                color = (255, 255, 255)  # White text
                
                # Highlight freshness text with color
                if i == 0:
                    color = box_color
                elif i == len(texts) - 1:  # Freshness percentage
                    color = self.get_freshness_color(freshness_ratio)
                
                cv2.putText(frame, text, (text_x, text_y_pos), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return frame
    
    def add_ui_overlay(self, frame, analysis_enabled, frame_count):
        """Add UI information overlay with enhanced details"""
        # Status bar with overlay duration info
        status = "ANALYSIS: ON" if analysis_enabled else "ANALYSIS: OFF"
        status_color = (0, 255, 0) if analysis_enabled else (0, 0, 255)
        
        # Main status
        cv2.putText(frame, status, (10, frame.shape[0] - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Overlay duration info
        if analysis_enabled and self.current_analysis:
            frames_since = frame_count - self.analysis_frame_count
            remaining = max(0, self.overlay_duration - frames_since)
            duration_text = f"Overlay: {remaining}/{self.overlay_duration} frames"
            cv2.putText(frame, duration_text, (10, frame.shape[0] - 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Frame counter
        cv2.putText(frame, f"Frame: {frame_count}", (10, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Controls hint
        controls_text = "Q=Quit, A=Toggle, S=Save, +/-=Duration"
        cv2.putText(frame, controls_text, (frame.shape[1] - 300, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Overlay duration setting
        duration_setting = f"Duration: {self.overlay_duration}f"
        cv2.putText(frame, duration_setting, (frame.shape[1] - 150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        return frame
    
    def save_frame(self, frame):
        """Save current frame to file with analysis metadata"""
        timestamp = int(time.time())
        filename = f"capture_{timestamp}.jpg"
        
        # Add metadata to image if analysis exists
        if self.current_analysis:
            analysis_text = f"DeepFace Analysis - {time.ctime()}"
            cv2.putText(frame, analysis_text, (10, frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imwrite(filename, frame)
        print(f"Frame saved as: {filename}")
    
    def cleanup(self):
        """Clean up resources"""
        self.stream_active = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Stream stopped and resources cleaned up")

def main():
    """Main function with enhanced overlay duration"""
    print("=" * 50)
    print("DeepFace Stream with Persistent Overlays")
    print("=" * 50)
    
    # Start with longer overlay duration (45 frames = ~1.5 seconds at 30fps)
    stream = DeepFaceStream(
        camera_index=0,
        enable_analysis=True,
        overlay_duration=45  # Increased from default
    )
    stream.start_stream()

if __name__ == "__main__":
    main()