import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import face_recognition
import numpy as np
import pickle
import os
import time
import logging
from typing import Optional, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class FaceRecognitionSystem:
    def __init__(self, frame_width: int = 640, frame_height: int = 480):
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Input source configuration
        self.input_source = None
        self.input_type = None
        
        # Data storage
        self.known_face_encodings: List[np.ndarray] = []
        self.known_face_names: List[str] = []
        
        # Capture state
        self.capture_new_face = False
        self.new_face_name = ""
        self.new_face_encoding: Optional[np.ndarray] = None
        
        # Performance tracking
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()
        self.processing_times = []
        
        # Video capture object
        self.cap: Optional[cv2.VideoCapture] = None
        
        # Load existing data
        self.load_face_data()
        
        # Initialize input source
        self.setup_input_source()

    def setup_input_source(self) -> None:
        """Setup input source based on user choice with validation"""
        while True:
            print("\n=== Input Source Selection ===")
            print("1. Camera (default device)")
            print("2. RTSP Stream") 
            print("3. USB Camera (specific device index)")
            print("4. Video File")
            choice = input("Choose input source (1/2/3/4) [1]: ").strip()
            
            if choice in ["1", ""]:
                self.input_source = 0
                self.input_type = "camera"
                if self.test_camera_device(0):
                    logging.info("Using default camera (device 0)")
                    break
                else:
                    logging.error("Default camera not available")
                    
            elif choice == "2":
                rtsp_url = input("Enter RTSP URL: ").strip()
                if rtsp_url:
                    self.input_source = rtsp_url
                    self.input_type = "rtsp"
                    logging.info(f"Using RTSP stream: {rtsp_url}")
                    break
                else:
                    print("Invalid RTSP URL")
                    
            elif choice == "3":
                try:
                    device_index = input("Enter camera device index [0]: ").strip()
                    device_index = int(device_index) if device_index else 0
                    self.input_source = device_index
                    self.input_type = "camera"
                    if self.test_camera_device(device_index):
                        logging.info(f"Using USB camera (device {device_index})")
                        break
                    else:
                        logging.error(f"Camera device {device_index} not available")
                except ValueError:
                    print("Invalid device index")
                    
            elif choice == "4":
                video_path = input("Enter video file path: ").strip()
                if os.path.exists(video_path):
                    self.input_source = video_path
                    self.input_type = "video"
                    logging.info(f"Using video file: {video_path}")
                    break
                else:
                    print("Video file not found")
            else:
                print("Invalid choice")

    def test_camera_device(self, device_index: int) -> bool:
        """Test if camera device is available"""
        cap = cv2.VideoCapture(device_index)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            return ret
        return False

    def create_video_capture(self) -> Optional[cv2.VideoCapture]:
        """Create and configure video capture with optimized settings"""
        cap = cv2.VideoCapture(self.input_source)
        
        if not cap.isOpened():
            return None

        # Optimize based on input type
        if self.input_type == "rtsp":
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, 15)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
            
        elif self.input_type == "camera":
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            
        elif self.input_type == "video":
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
        # Common optimizations
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        return cap

    def load_face_data(self) -> None:
        """Load face encodings with better error handling"""
        try:
            if os.path.exists("face_data.pkl"):
                with open("face_data.pkl", "rb") as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data.get('encodings', [])
                    self.known_face_names = data.get('names', [])
                logging.info(f"Loaded {len(self.known_face_names)} known faces")
                
                # Validate encodings
                valid_encodings = []
                valid_names = []
                for enc, name in zip(self.known_face_encodings, self.known_face_names):
                    if isinstance(enc, np.ndarray) and enc.shape[0] == 128:
                        valid_encodings.append(enc)
                        valid_names.append(name)
                
                if len(valid_encodings) != len(self.known_face_encodings):
                    logging.warning("Some face encodings were invalid and removed")
                    self.known_face_encodings = valid_encodings
                    self.known_face_names = valid_names
                    
        except Exception as e:
            logging.error(f"Error loading face data: {e}")
            # Initialize empty data
            self.known_face_encodings = []
            self.known_face_names = []

    def save_face_data(self) -> None:
        """Save face encodings with backup"""
        try:
            data = {
                'encodings': self.known_face_encodings,
                'names': self.known_face_names,
                'timestamp': time.time()
            }
        
            # Create backup if exists
            if os.path.exists("face_data.pkl"):
                os.rename("face_data.pkl", "face_data.pkl.backup")
            
            with open("face_data.pkl", "wb") as f:
                pickle.dump(data, f)
                
            logging.info(f"Saved {len(self.known_face_names)} faces to file")
            
        except Exception as e:
            logging.error(f"Error saving face data: {e}")
            # Restore backup if save failed
            if os.path.exists("face_data.pkl.backup"):
                os.rename("face_data.pkl.backup", "face_data.pkl")

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, bool, float]:
        """Optimized frame processing for single-threaded operation"""
        start_time = time.time()
        
        try:
            # Resize for processing
            scale_factor = 0.25
            small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Face detection
            face_locations = face_recognition.face_locations(
                rgb_small_frame, 
                model="hog"  # Use hog for CPU, cnn for GPU
            )
            
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []
            
            # Face recognition
            for face_encoding in face_encodings:
                if len(self.known_face_encodings) == 0:
                    face_names.append("Unknown")
                    continue
                    
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                name = "Unknown"
                
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                
                face_names.append(name)
            
            # Handle new face capture
            if self.capture_new_face and len(face_encodings) > 0:
                self.new_face_encoding = face_encodings[0]
            
            # Draw results
            frame = self.draw_detections(frame, face_locations, face_names, scale_factor)
            
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            # Keep only recent processing times for average calculation
            if len(self.processing_times) > 30:
                self.processing_times.pop(0)
            
            return frame, len(face_locations) > 0, processing_time
            
        except Exception as e:
            logging.error(f"Error processing frame: {e}")
            processing_time = time.time() - start_time
            # Return original frame on error
            return frame, False, processing_time

    def draw_detections(self, frame: np.ndarray, face_locations: List, 
                       face_names: List[str], scale_factor: float) -> np.ndarray:
        """Draw face detections and information on frame"""
        scale = int(1 / scale_factor)
        
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale coordinates
            top *= scale
            right *= scale
            bottom *= scale
            left *= scale
            
            # Draw bounding box
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Draw label
            label_bg = (left, bottom - 35, right - left, 35)
            cv2.rectangle(frame, (label_bg[0], label_bg[1]), 
                         (label_bg[0] + label_bg[2], label_bg[1] + label_bg[3]), color, cv2.FILLED)
            
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)
        
        # Calculate average processing time
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
        
        # Add system information
        info_lines = [
            f"FPS: {self.fps}",
            f"Processing: {avg_processing_time*1000:.1f}ms",
            f"Source: {self.input_type}",
            f"Faces: {len(self.known_face_names)} known"
        ]
        
        for i, line in enumerate(info_lines):
            y_position = 30 + (i * 25)
            cv2.putText(frame, line, (10, y_position), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Status messages
        status_y = frame.shape[0] - 10
        status_text = "Press 'i'=Input face, 's'=Save, 'q'=Quit"
        cv2.putText(frame, status_text, (10, status_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if self.capture_new_face:
            warning_y = frame.shape[0] - 40
            cv2.putText(frame, "CAPTURE MODE - Position face in frame", 
                       (10, warning_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame

    def start_capture_mode(self) -> None:
        """Start face capture mode"""
        if self.capture_new_face:
            logging.info("Already in capture mode")
            return
        
        self.capture_new_face = True
        name = input("\nEnter name for new face: ").strip()
        
        if name:
            self.new_face_name = name
            logging.info(f"Capture mode activated for: {name}")
            logging.info("Position face in frame and ensure good lighting")
        else:
            self.capture_new_face = False
            logging.info("No name entered, cancelling capture")

    def save_captured_face(self) -> None:
        """Save captured face"""
        if not self.capture_new_face or self.new_face_encoding is None:
            logging.warning("No face to save. Enter capture mode first.")
            return
        
        if not self.new_face_name:
            logging.warning("No name specified for the face")
            return
        
        # Add to known faces
        self.known_face_encodings.append(self.new_face_encoding)
        self.known_face_names.append(self.new_face_name)
        
        name = self.new_face_name
        
        # Reset capture state
        self.capture_new_face = False
        self.new_face_name = ""
        self.new_face_encoding = None
    
        # Save data
        self.save_face_data()
        logging.info(f"Face saved for: {name}")

    def run(self) -> None:
        """Main system loop - single threaded"""
        try:
            self.cap = self.create_video_capture()
            if self.cap is None or not self.cap.isOpened():
                logging.error(f"Cannot open input source: {self.input_source}")
                return

            logging.info("Face recognition system started")
            logging.info("Controls: 'i' - Input new face, 's' - Save face, 'q' - Quit")
            
            last_frame_time = time.time()
            frame_timeout = 5.0
            reconnect_attempts = 0
            max_reconnect_attempts = 3
            
            while True:
                # Update FPS counter
                self.frame_count += 1
                current_time = time.time()
                if current_time - self.last_fps_time >= 1.0:
                    self.fps = self.frame_count
                    self.frame_count = 0
                    self.last_fps_time = current_time
                
                # Read frame
                ret, frame = self.cap.read()
                
                if not ret:
                    if self.input_type == "video":
                        logging.info("Video ended")
                        break
                    else:
                        logging.warning("Failed to grab frame")
                        reconnect_attempts += 1
                        if reconnect_attempts >= max_reconnect_attempts:
                            logging.error("Max reconnection attempts reached")
                            break
                        
                        # Try to reconnect
                        self.cap.release()
                        time.sleep(2)
                        self.cap = self.create_video_capture()
                        if self.cap is None or not self.cap.isOpened():
                            continue
                        else:
                            reconnect_attempts = 0
                            logging.info("Reconnected to input source")
                        continue
                
                reconnect_attempts = 0  # Reset on successful frame
                
                # Process frame
                processed_frame, face_detected, processing_time = self.process_frame(frame)
                
                # Adaptive frame skipping for high processing times
                if processing_time > 0.2:  # If processing takes more than 200ms
                    logging.debug(f"Slow processing: {processing_time:.3f}s - consider optimizing")
                
                # Display frame
                cv2.imshow('Face Recognition System', processed_frame)
                
                # Handle key presses with reduced wait time for responsiveness
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('i'):
                    self.start_capture_mode()
                elif key == ord('s'):
                    self.save_captured_face()
                
                # Check for timeout
                if time.time() - last_frame_time > frame_timeout:
                    logging.warning("Frame processing timeout")
                    break
                    
        except KeyboardInterrupt:
            logging.info("Interrupted by user")
        except Exception as e:
            logging.error(f"Unexpected error in main loop: {e}")
        finally:
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            logging.info("Face recognition system stopped")

def main():
    """Main function with exception handling"""
    try:
        face_system = FaceRecognitionSystem()
        face_system.run()
    except Exception as e:
        logging.error(f"Failed to start system: {e}")
    finally:
        logging.info("Application exited")

if __name__ == "__main__":
    main()
