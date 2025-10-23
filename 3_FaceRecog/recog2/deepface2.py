import cv2
import os
from deepface import DeepFace
import time

class DeepFaceStream:
    def __init__(self, db_path=None, camera_index=0, enable_analysis=True):
        """
        Initialize DeepFace stream with real-time analysis capabilities
        
        Args:
            db_path: Path to database for face recognition (optional)
            camera_index: Camera device index
            enable_analysis: Enable facial attribute analysis
        """
        self.db_path = db_path
        self.camera_index = camera_index
        self.enable_analysis = enable_analysis
        self.cap = None
        self.stream_active = False
        
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
                    print(f"Analysis {'enabled' if analysis_enabled else 'disabled'}")
                elif key == ord('s'):
                    self.save_frame(frame)
                    
        except KeyboardInterrupt:
            print("Stream interrupted by user")
        except Exception as e:
            print(f"Error in stream: {e}")
        finally:
            self.cleanup()
    
    def process_frame_with_deepface(self, frame, analysis_enabled, frame_count):
        """Process frame using DeepFace analysis"""
        try:
            # Convert BGR to RGB for DeepFace
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if analysis_enabled and frame_count % 15 == 0:  # Analyze every 15th frame
                analysis_results = self.analyze_frame(rgb_frame)
                frame = self.draw_analysis_results(frame, analysis_results)
            
            # Add UI information
            frame = self.add_ui_overlay(frame, analysis_enabled, frame_count)
            
            return frame
            
        except Exception as e:
            print(f"Frame processing error: {e}")
            return frame
    
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
    
    def draw_analysis_results(self, frame, analysis):
        """Draw analysis results on the frame"""
        if analysis is None or len(analysis) == 0:
            cv2.putText(frame, "No face detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame
        
        # Process each detected face
        for i, result in enumerate(analysis):
            frame = self.draw_single_face_analysis(frame, result, i)
        
        return frame
    
    def draw_single_face_analysis(self, frame, result, face_index):
        """Draw analysis for a single face"""
        # Get face region
        region = result.get('region', {})
        x = region.get('x', 0)
        y = region.get('y', 0)
        w = region.get('w', 0)
        h = region.get('h', 0)
        
        # Draw bounding box
        if w > 0 and h > 0:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Prepare analysis text
            texts = [
                f"Face {face_index + 1}",
                f"Age: {result.get('age', 'N/A')}",
                f"Gender: {result.get('dominant_gender', 'N/A')}",
                f"Emotion: {result.get('dominant_emotion', 'N/A')}",
                f"Race: {result.get('dominant_race', 'N/A')}"
            ]
            
            # Draw text background for better readability
            text_x = x + w + 10 if x + w + 150 < frame.shape[1] else x - 150
            text_y = y
            
            for i, text in enumerate(texts):
                bg_y = text_y + i * 25
                cv2.rectangle(frame, (text_x - 5, bg_y - 20), 
                            (text_x + 200, bg_y + 5), (0, 0, 0), -1)
                cv2.putText(frame, text, (text_x, bg_y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def add_ui_overlay(self, frame, analysis_enabled, frame_count):
        """Add UI information overlay"""
        # Status bar
        status = "ANALYSIS: ON" if analysis_enabled else "ANALYSIS: OFF"
        status_color = (0, 255, 0) if analysis_enabled else (0, 0, 255)
        
        cv2.putText(frame, status, (10, frame.shape[0] - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Frame counter
        cv2.putText(frame, f"Frame: {frame_count}", (10, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Controls hint
        controls_text = "Controls: Q=Quit, A=Toggle Analysis, S=Save Frame"
        cv2.putText(frame, controls_text, (frame.shape[1] - 400, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame
    
    def save_frame(self, frame):
        """Save current frame to file"""
        timestamp = int(time.time())
        filename = f"capture_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Frame saved as: {filename}")
    
    def cleanup(self):
        """Clean up resources"""
        self.stream_active = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Stream stopped and resources cleaned up")

# Advanced version using DeepFace.stream() directly
class DeepFaceNativeStream:
    def __init__(self, db_path=None, camera_index=0):
        """
        Use DeepFace's native stream function
        
        Args:
            db_path: Database path for face recognition
            camera_index: Camera device index
        """
        self.db_path = db_path
        self.camera_index = camera_index
        
    def start_native_stream(self):
        """Start DeepFace's native stream function"""
        try:
            print("Starting DeepFace native stream...")
            print("This will open a new window with real-time analysis")
            print("Press 'q' to quit the native stream")
            
            if self.db_path and os.path.exists(self.db_path):
                print(f"Using database: {self.db_path}")
                DeepFace.stream(db_path=self.db_path, source=self.camera_index)
            else:
                print("Starting analysis-only stream (no database)")
                DeepFace.stream(source=self.camera_index)
                
        except Exception as e:
            print(f"Native stream error: {e}")

def main():
    """Main function with menu options"""
    print("=" * 50)
    print("DeepFace Real-Time Stream Implementation")
    print("=" * 50)
    print("\nChoose streaming mode:")
    print("1. Custom Stream (More control, OpenCV-based)")
    print("2. Native DeepFace Stream (Built-in functionality)")
    print("3. Stream with Face Recognition Database")
    
    try:
        choice = input("\nEnter your choice (1, 2, or 3): ").strip()
        
        if choice == "1":
            # Custom stream with OpenCV
            stream = DeepFaceStream(
                camera_index=0,
                enable_analysis=True
            )
            stream.start_stream()
            
        elif choice == "2":
            # Native DeepFace stream
            db_path = input("Enter databse path: ").strip()
            if db_path and os.path.exists(db_path):
                native_stream = DeepFaceNativeStream(
                    db_path=db_path,
                    camera_index=0
                    )
                native_stream.start_native_stream()
                
            else:
                print("Invalid database path. Starting analysis-only stream.")
                native_stream = DeepFaceNativeStream(camera_index=0)
                native_stream.start_native_stream()                
            
        elif choice == "3":
            # Stream with face recognition database
            db_path = input("Enter database path (or press Enter for analysis only): ").strip()
            if db_path and os.path.exists(db_path):
                native_stream = DeepFaceNativeStream(
                    db_path=db_path,
                    camera_index=0
                )
                native_stream.start_native_stream()
            else:
                print("Invalid database path. Starting analysis-only stream.")
                native_stream = DeepFaceNativeStream(camera_index=0)
                native_stream.start_native_stream()
                
        else:
            print("Invalid choice. Using custom stream as default.")
            stream = DeepFaceStream(camera_index=0, enable_analysis=True)
            stream.start_stream()
            
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()