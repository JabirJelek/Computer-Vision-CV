import cv2
import time
import numpy as np
from config import *
from face_detector import FaceDetector
from face_recognizer import FaceRecognizer
from camera import Camera

class FaceRecognitionSystem:
    def __init__(self):
        self.camera = Camera()
        self.detector = FaceDetector()
        self.recognizer = FaceRecognizer()
        self.running = False
        
    def draw_results(self, image, faces, results):
        """Draw bounding boxes and recognition results on image"""
        display_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        for face, result in zip(faces, results):
            x1, y1, x2, y2 = face['bbox']
            name, confidence = result
            
            # Draw bounding box
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(display_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            label = f"{name}: {confidence:.2f}" if SHOW_CONFIDENCE else name
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(display_image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(display_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return display_image
    
    def run(self):
        """Main pipeline execution"""
        self.running = True
        fps_time = time.time()
        frame_count = 0
        
        print("🚀 Starting Face Recognition System...")
        print("Press 'q' to quit, 's' to save current frame")
        
        while self.running:
            # Get frame from camera
            frame = self.camera.get_frame()
            if frame is None:
                continue
            
            # Detect faces
            faces = self.detector.detect_faces(frame)
            
            # Recognize each face
            results = []
            for face in faces:
                name, confidence = self.recognizer.recognize_face(face['roi'])
                results.append((name, confidence))
            
            # Draw results on frame
            display_frame = self.draw_results(frame, faces, results)
            
            # Calculate and display FPS
            if DISPLAY_FPS:
                frame_count += 1
                if time.time() - fps_time >= 1.0:
                    fps = frame_count / (time.time() - fps_time)
                    cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    frame_count = 0
                    fps_time = time.time()
            
            # Display frame
            cv2.imshow("Face Recognition System", display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                timestamp = int(time.time())
                cv2.imwrite(f"capture_{timestamp}.jpg", display_frame)
                print(f"📸 Frame saved as capture_{timestamp}.jpg")
        
        self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        self.running = False
        self.camera.release()
        cv2.destroyAllWindows()
        print("🛑 System stopped")

if __name__ == "__main__":
    system = FaceRecognitionSystem()
    system.run()