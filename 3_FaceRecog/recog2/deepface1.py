import cv2
import numpy as np
from deepface import DeepFace
import time

class DeepFaceCamera:
    def __init__(self, camera_index=0, analysis_interval=30):
        """
        Initialize the DeepFace camera pipeline
        
        Args:
            camera_index: Index of the camera (0 for default built-in camera)
            analysis_interval: Process every Nth frame to reduce computational load
        """
        self.camera_index = camera_index
        self.analysis_interval = analysis_interval
        self.frame_count = 0
        self.last_analysis = None
        self.cap = None
        
    def initialize_camera(self):
        """Initialize the camera"""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise Exception(f"Could not open camera with index {self.camera_index}")
        
        # Set camera resolution (adjust as needed)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("Camera initialized successfully")
        
    def analyze_frame(self, frame):
        """Analyze frame using DeepFace"""
        try:
            # Convert BGR to RGB for DeepFace
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Analyze the frame
            analysis = DeepFace.analyze(
                rgb_frame, 
                actions=['age'],
                detector_backend='opencv',  # Using OpenCV for face detection
                enforce_detection=False,    # Continue even if no face detected
                silent=True                 # Suppress verbose output
            )
            
            return analysis
        except Exception as e:
            print(f"Analysis error: {e}")
            return None
    
    def draw_analysis_results(self, frame, analysis):
        """Draw analysis results on the frame"""
        if analysis is None or len(analysis) == 0:
            cv2.putText(frame, "No face detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return
        
        # Get the first face analysis (assuming single face for simplicity)
        result = analysis[0]
        
        # Draw bounding box around the face
        region = result.get('region', {})
        x = region.get('x', 0)
        y = region.get('y', 0)
        w = region.get('w', 0)
        h = region.get('h', 0)
        
        if w > 0 and h > 0:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Prepare analysis text
        texts = [
            f"Age: {result.get('age', 'N/A')}",
            f"Gender: {result.get('dominant_gender', 'N/A')}",
            f"Emotion: {result.get('dominant_emotion', 'N/A')}",
            f"Race: {result.get('dominant_race', 'N/A')}"
        ]
        
        # Display analysis results
        y_offset = 40
        for i, text in enumerate(texts):
            cv2.putText(frame, text, (10, y_offset + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def run(self):
        """Main pipeline execution"""
        self.initialize_camera()
        
        print("Starting DeepFace camera pipeline...")
        print("Press 'q' to quit, 's' to save current frame")
        
        try:
            while True:
                # Read frame from camera
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Process frame for analysis at specified interval
                self.frame_count += 1
                if self.frame_count % self.analysis_interval == 0:
                    self.last_analysis = self.analyze_frame(frame)
                
                # Draw results on frame
                self.draw_analysis_results(frame, self.last_analysis)
                
                # Display frame info
                cv2.putText(frame, f"Frame: {self.frame_count}", (10, frame.shape[0] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Show the frame
                cv2.imshow('DeepFace Camera Analysis', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save current frame
                    filename = f"frame_{int(time.time())}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Frame saved as {filename}")
                    
        except KeyboardInterrupt:
            print("Pipeline interrupted by user")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Pipeline cleaned up")

# Advanced version with multiple face support
class MultiFaceDeepFaceCamera(DeepFaceCamera):
    def draw_analysis_results(self, frame, analysis):
        """Draw analysis results for multiple faces"""
        if analysis is None or len(analysis) == 0:
            cv2.putText(frame, "No faces detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return
        
        for i, result in enumerate(analysis):
            # Draw bounding box around each face
            region = result.get('region', {})
            x = region.get('x', 0)
            y = region.get('y', 0)
            w = region.get('w', 0)
            h = region.get('h', 0)
            
            if w > 0 and h > 0:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Display basic info near each face
                info_text = f"Face {i+1}: {result.get('dominant_emotion', 'N/A')}"
                cv2.putText(frame, info_text, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def main():
    # Choose which version to run
    print("Choose pipeline mode:")
    print("1. Single face analysis (faster)")
    print("2. Multi-face analysis")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "2":
        pipeline = MultiFaceDeepFaceCamera(
            camera_index=0,
            analysis_interval=20  # Process every 20th frame
        )
    else:
        pipeline = DeepFaceCamera(
            camera_index=0,
            analysis_interval=15  # Process every 15th frame
        )
    
    # Run the pipeline
    pipeline.run()

if __name__ == "__main__":
    main()