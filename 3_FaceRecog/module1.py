import face_recognition
import cv2
from multiprocessing import Process, Manager, cpu_count, set_start_method
import time
import numpy as np
import threading
import platform
import os
import pickle
from pathlib import Path

class DynamicFaceEncoder:
    def __init__(self, encodings_file="face_encodings.pkl"):
        self.encodings_file = encodings_file
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_encodings()
    
    def load_encodings(self):
        """Load existing face encodings from file"""
        if os.path.exists(self.encodings_file):
            try:
                with open(self.encodings_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data['encodings']
                    self.known_face_names = data['names']
                print(f"Loaded {len(self.known_face_names)} known faces")
            except Exception as e:
                print(f"Error loading encodings: {e}")
                self.known_face_encodings = []
                self.known_face_names = []
    
    def save_encodings(self):
        """Save current encodings to file"""
        try:
            with open(self.encodings_file, 'wb') as f:
                pickle.dump({
                    'encodings': self.known_face_encodings,
                    'names': self.known_face_names
                }, f)
            print(f"Saved {len(self.known_face_names)} face encodings")
        except Exception as e:
            print(f"Error saving encodings: {e}")
    
    def add_face_from_image(self, image_path, name):
        """Add a new face from an image file"""
        try:
            # Load image
            image = face_recognition.load_image_file(image_path)
            
            # Find face encodings
            face_encodings = face_recognition.face_encodings(image)
            
            if len(face_encodings) == 0:
                print(f"No faces found in {image_path}")
                return False
            
            # Use the first face found
            face_encoding = face_encodings[0]
            
            # Add to known faces
            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(name)
            
            # Save updated encodings
            self.save_encodings()
            print(f"Added face: {name}")
            return True
            
        except Exception as e:
            print(f"Error adding face from {image_path}: {e}")
            return False
    
    def add_face_from_frame(self, frame, name):
        """Add a new face from a video frame"""
        try:
            # Convert BGR to RGB
            rgb_frame = frame[:, :, ::-1]
            
            # Find face encodings
            face_encodings = face_recognition.face_encodings(rgb_frame)
            
            if len(face_encodings) == 0:
                print(f"No faces found in frame")
                return False
            
            # Use the first face found
            face_encoding = face_encodings[0]
            
            # Add to known faces
            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(name)
            
            # Save updated encodings
            self.save_encodings()
            print(f"Added face from frame: {name}")
            return True
            
        except Exception as e:
            print(f"Error adding face from frame: {e}")
            return False

# Get next worker's id
def next_id(current_id, worker_num):
    return 1 if current_id == worker_num else current_id + 1

# Get previous worker's id
def prev_id(current_id, worker_num):
    return worker_num if current_id == 1 else current_id - 1

# A subprocess use to capture frames.
def capture(read_frame_list, Global, worker_num):
    video_capture = cv2.VideoCapture(0)
    print("Width: %d, Height: %d, FPS: %d" % (video_capture.get(3), video_capture.get(4), video_capture.get(5)))

    while not Global.is_exit:
        if Global.buff_num != next_id(Global.read_num, worker_num):
            ret, frame = video_capture.read()
            if ret:
                read_frame_list[Global.buff_num] = frame
                Global.buff_num = next_id(Global.buff_num, worker_num)
            else:
                time.sleep(0.01)
        else:
            time.sleep(0.01)

    video_capture.release()

# Many subprocess use to process frames.
def process(worker_id, read_frame_list, write_frame_list, Global, worker_num):
    face_encoder = Global.face_encoder
    
    while not Global.is_exit:
        # Wait to read
        while Global.read_num != worker_id or Global.read_num != prev_id(Global.buff_num, worker_num):
            if Global.is_exit:
                break
            time.sleep(0.01)

        time.sleep(Global.frame_delay)

        # Read a single frame from frame list
        frame_process = read_frame_list[worker_id]
        Global.read_num = next_id(Global.read_num, worker_num)

        # Convert the image from BGR color to RGB color
        rgb_frame = frame_process[:, :, ::-1]

        # Find all the faces and face encodings in the frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Get current encodings
        current_encodings = face_encoder.known_face_encodings
        current_names = face_encoder.known_face_names

        # Loop through each face in this frame of video
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            name = "Unknown"

            if current_encodings:
                # Compare faces with tolerance
                face_distances = face_recognition.face_distance(current_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                
                # Use a threshold for face recognition
                if face_distances[best_match_index] < 0.6:  # Adjust this threshold as needed
                    name = current_names[best_match_index]
                else:
                    name = "Unknown"
            else:
                name = "Unknown"

            # Draw a box around the face
            cv2.rectangle(frame_process, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame_process, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame_process, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Add frame counter and instructions
        cv2.putText(frame_process, "Press 'a' to add face, 'q' to quit", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Wait to write
        while Global.write_num != worker_id:
            time.sleep(0.01)

        write_frame_list[worker_id] = frame_process
        Global.write_num = next_id(Global.write_num, worker_num)

if __name__ == '__main__':
    # Fix Bug on MacOS
    if platform.system() == 'Darwin':
        set_start_method('forkserver')

    # Initialize face encoder
    face_encoder = DynamicFaceEncoder()

    # Global variables
    Global = Manager().Namespace()
    Global.buff_num = 1
    Global.read_num = 1
    Global.write_num = 1
    Global.frame_delay = 0
    Global.is_exit = False
    Global.face_encoder = face_encoder  # Share the encoder instance
    
    read_frame_list = Manager().dict()
    write_frame_list = Manager().dict()

    # Number of workers
    worker_num = max(2, cpu_count() - 1)

    # Subprocess list
    processes = []

    # Create a thread to capture frames
    processes.append(threading.Thread(target=capture, args=(read_frame_list, Global, worker_num)))
    processes[0].start()

    # Create workers
    for worker_id in range(1, worker_num + 1):
        p = Process(target=process, args=(worker_id, read_frame_list, write_frame_list, Global, worker_num))
        p.start()
        processes.append(p)

    # Main display loop
    last_num = 1
    fps_list = []
    tmp_time = time.time()
    
    # State for adding new faces
    adding_new_face = False
    new_face_name = ""
    name_input_active = False
    
    print("Face Recognition System Started")
    print("Commands:")
    print("- Press 'a' to add a new face")
    print("- Press 'q' to quit")
    
    while not Global.is_exit:
        while Global.write_num != last_num:
            last_num = int(Global.write_num)

            # Calculate fps
            delay = time.time() - tmp_time
            tmp_time = time.time()
            fps_list.append(delay)
            if len(fps_list) > 5 * worker_num:
                fps_list.pop(0)
            fps = len(fps_list) / np.sum(fps_list) if fps_list else 0

            # Calculate frame delay for smooth video
            if fps < 6:
                Global.frame_delay = (1 / fps) * 0.75 if fps > 0 else 0
            elif fps < 20:
                Global.frame_delay = (1 / fps) * 0.5
            elif fps < 30:
                Global.frame_delay = (1 / fps) * 0.25
            else:
                Global.frame_delay = 0

            # Display the resulting image
            current_frame = write_frame_list[prev_id(Global.write_num, worker_num)]
            cv2.imshow('Video', current_frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            Global.is_exit = True
            break
        elif key == ord('a') and not name_input_active:
            # Start process to add new face
            name_input_active = True
            new_face_name = input("Enter name for the new face: ").strip()
            if new_face_name:
                print("Position your face in the frame and press 's' to save, or 'c' to cancel")
                adding_new_face = True
            else:
                name_input_active = False
                adding_new_face = False
        elif key == ord('s') and adding_new_face:
            # Save the current frame as a new face
            current_frame = write_frame_list[prev_id(Global.write_num, worker_num)]
            if face_encoder.add_face_from_frame(current_frame, new_face_name):
                print(f"Successfully added {new_face_name} to known faces!")
            else:
                print(f"Failed to add {new_face_name}. Make sure a face is visible.")
            adding_new_face = False
            name_input_active = False
        elif key == ord('c') and adding_new_face:
            # Cancel adding new face
            print("Cancelled adding new face")
            adding_new_face = False
            name_input_active = False

        time.sleep(0.01)

    # Cleanup
    cv2.destroyAllWindows()
    print("System shutdown complete")