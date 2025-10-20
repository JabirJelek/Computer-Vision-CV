import cv2
import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
from threading import Thread, Lock
from queue import Queue
import time
from typing import Dict, List, Tuple, Optional

class FaceRecognitionSystem:
    def __init__(self, config: Dict):
        self.config = config
        self.detection_model = None
        self.embeddings_db = {}
        self.identity_centroids = {}
        
        # Debug statistics
        self.debug_stats = {
            'total_frames_processed': 0,
            'total_faces_detected': 0,
            'total_faces_recognized': 0,
            'detection_times': [],
            'embedding_times': [],
            'recognition_times': [],
            'last_processing_time': 0
        }
        
        self._load_models()
        self._load_embeddings_database()
        
    def _load_models(self):
        """Load YOLO face detection model from local path"""
        try:
            model_path = Path(self.config['detection_model_path'])
            if not model_path.exists():
                raise FileNotFoundError(f"YOLO model not found at {model_path}")
                
            self.detection_model = YOLO(str(model_path))
            print(f"✅ YOLO model loaded from {model_path}")
            
        except Exception as e:
            print(f"❌ Failed to load YOLO model: {e}")
            raise
            
    def _load_embeddings_database(self):
        """Load pre-computed face embeddings from JSON with your structure"""
        try:
            db_path = Path(self.config['embeddings_db_path'])
            if not db_path.exists():
                print("⚠️  Embeddings database not found, starting fresh")
                self.embeddings_db = {"persons": {}, "metadata": {}}
                return
                
            with open(db_path, 'r') as f:
                self.embeddings_db = json.load(f)
                
            if "persons" in self.embeddings_db:
                for person_id, person_data in self.embeddings_db["persons"].items():
                    display_name = person_data["display_name"]
                    centroid = person_data["centroid_embedding"]
                    self.identity_centroids[display_name] = np.array(centroid)
                    
                print(f"✅ Loaded {len(self.identity_centroids)} identities from database")
                print(f"📊 Available persons: {list(self.identity_centroids.keys())}")
                
            else:
                print("⚠️  No 'persons' key found in JSON database")
                
        except Exception as e:
            print(f"❌ Failed to load embeddings database: {e}")
            raise

    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """Detect faces using YOLO with optimized settings"""
        start_time = time.time()
        try:
            results = self.detection_model(
                frame, 
                conf=self.config['detection_confidence'],
                iou=self.config['detection_iou'],
                verbose=False
            )
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        
                        detections.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(confidence)
                        })
            
            # Update debug stats
            detection_time = (time.time() - start_time) * 1000
            self.debug_stats['detection_times'].append(detection_time)
            self.debug_stats['detection_times'] = self.debug_stats['detection_times'][-100:]  # Keep last 100
            
            return detections
            
        except Exception as e:
            print(f"Detection error: {e}")
            return []

    def extract_embedding(self, face_roi: np.ndarray) -> Optional[np.ndarray]:
        """Extract face embedding using DeepFace"""
        start_time = time.time()
        try:
            if len(face_roi.shape) == 3:
                face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            else:
                return None
                
            if face_roi.shape[0] < 50 or face_roi.shape[1] < 50:
                return None
                
            embedding_obj = DeepFace.represent(
                face_rgb,
                model_name=self.config['embedding_model'],
                enforce_detection=False,
                detector_backend='skip'
            )
            
            if embedding_obj:
                # Update debug stats
                embedding_time = (time.time() - start_time) * 1000
                self.debug_stats['embedding_times'].append(embedding_time)
                self.debug_stats['embedding_times'] = self.debug_stats['embedding_times'][-100:]
                
                return np.array(embedding_obj[0]['embedding'])
                
        except Exception as e:
            print(f"Embedding extraction error: {e}")
            
        return None

    def recognize_face(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """Match embedding against database using cosine similarity"""
        start_time = time.time()
        
        if not self.identity_centroids:
            return None, 0.0
            
        best_similarity = -1.0
        best_identity = None
        
        for identity, centroid in self.identity_centroids.items():
            similarity = cosine_similarity([embedding], [centroid])[0][0]
            
            if similarity > best_similarity and similarity >= self.config['recognition_threshold']:
                best_similarity = similarity
                best_identity = identity
        
        # Update debug stats
        recognition_time = (time.time() - start_time) * 1000
        self.debug_stats['recognition_times'].append(recognition_time)
        self.debug_stats['recognition_times'] = self.debug_stats['recognition_times'][-100:]
                
        return best_identity, best_similarity

    def process_frame(self, frame: np.ndarray) -> List[Dict]:
        """Complete pipeline: detect → extract → recognize"""
        start_time = time.time()
        results = []
        
        # Detect faces
        detections = self.detect_faces(frame)
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            
            padding = self.config.get('roi_padding', 10)
            h, w = frame.shape[:2]
            x1_pad = max(0, x1 - padding)
            y1_pad = max(0, y1 - padding)
            x2_pad = min(w, x2 + padding)
            y2_pad = min(h, y2 + padding)
            
            face_roi = frame[y1_pad:y2_pad, x1_pad:x2_pad]
            
            if face_roi.size == 0:
                continue
                
            embedding = self.extract_embedding(face_roi)
            if embedding is None:
                continue
                
            identity, confidence = self.recognize_face(embedding)
            
            results.append({
                'bbox': detection['bbox'],
                'detection_confidence': detection['confidence'],
                'identity': identity,
                'recognition_confidence': confidence,
                'embedding': embedding.tolist()
            })
        
        # Update overall stats
        self.debug_stats['total_frames_processed'] += 1
        self.debug_stats['total_faces_detected'] += len(detections)
        self.debug_stats['total_faces_recognized'] += len([r for r in results if r['identity']])
        self.debug_stats['last_processing_time'] = (time.time() - start_time) * 1000
            
        return results

    def get_debug_stats(self) -> Dict:
        """Get comprehensive debug statistics"""
        stats = self.debug_stats.copy()
        
        # Calculate averages
        stats['avg_detection_time'] = np.mean(stats['detection_times']) if stats['detection_times'] else 0
        stats['avg_embedding_time'] = np.mean(stats['embedding_times']) if stats['embedding_times'] else 0
        stats['avg_recognition_time'] = np.mean(stats['recognition_times']) if stats['recognition_times'] else 0
        
        # Calculate recognition rate
        if stats['total_faces_detected'] > 0:
            stats['recognition_rate'] = (stats['total_faces_recognized'] / stats['total_faces_detected']) * 100
        else:
            stats['recognition_rate'] = 0
            
        return stats

    def add_new_identity(self, display_name: str, embeddings: List[np.ndarray]):
        """Add new identity to database with proper JSON structure"""
        existing_ids = [pid for pid in self.embeddings_db.get("persons", {}).keys()]
        new_id = f"person_{len(existing_ids) + 1:03d}"
        
        embedding_objects = []
        for i, emb in enumerate(embeddings):
            embedding_objects.append({
                "vector": emb.tolist(),
                "source_file": f"captured_{i}.jpg",
                "file_path": f"captured/{display_name}/captured_{i}.jpg",
                "embedding_length": 128
            })
        
        centroid = np.mean(embeddings, axis=0).tolist()
        
        new_person = {
            "person_id": new_id,
            "folder_name": display_name,
            "display_name": display_name,
            "embeddings": embedding_objects,
            "total_images": len(embeddings),
            "successful_embeddings": len(embeddings),
            "centroid_embedding": centroid
        }
        
        if "persons" not in self.embeddings_db:
            self.embeddings_db["persons"] = {}
            
        self.embeddings_db["persons"][new_id] = new_person
        self.identity_centroids[display_name] = np.array(centroid)
        
        self._update_metadata()
        self._save_embeddings_database()

    def _update_metadata(self):
        """Update metadata in the database"""
        persons = self.embeddings_db.get("persons", {})
        total_embeddings = sum(person["total_images"] for person in persons.values())
        
        self.embeddings_db["metadata"] = {
            "creation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_persons": len(persons),
            "total_embeddings": total_embeddings,
            "average_embeddings_per_person": total_embeddings / len(persons) if persons else 0,
            "description": "Simple dataset - no angle information"
        }

    def _save_embeddings_database(self):
        """Save updated embeddings to JSON file with proper structure"""
        try:
            with open(self.config['embeddings_db_path'], 'w') as f:
                json.dump(self.embeddings_db, f, indent=2)
            print(f"💾 Database saved with {len(self.embeddings_db.get('persons', {}))} persons")
        except Exception as e:
            print(f"❌ Failed to save embeddings database: {e}")

    def get_known_identities(self) -> List[str]:
        """Get list of all known identities"""
        return list(self.identity_centroids.keys())

class DisplayResizer:
    """Handles multiple resizing strategies for output display"""
    
    def __init__(self):
        self.current_scale = 1.0
        self.resize_method = "fit_to_screen"
        self.target_width = 1280
        self.target_height = 720
        self.maintain_aspect_ratio = True
        self.max_display_size = (1920, 1080)
        
    def resize_frame(self, frame: np.ndarray, method: str = None, 
                    target_size: Tuple[int, int] = None, 
                    scale: float = None) -> np.ndarray:
        if method:
            self.resize_method = method
            
        if target_size:
            self.target_width, self.target_height = target_size
            
        if scale:
            self.current_scale = scale
            
        original_height, original_width = frame.shape[:2]
        
        if self.resize_method == "fit_to_screen":
            return self._fit_to_screen(frame)
        elif self.resize_method == "fixed_size":
            return self._resize_fixed(frame, self.target_width, self.target_height)
        elif self.resize_method == "scale":
            return self._resize_scale(frame, self.current_scale)
        elif self.resize_method == "crop":
            return self._resize_crop(frame, self.target_width, self.target_height)
        elif self.resize_method == "letterbox":
            return self._resize_letterbox(frame, self.target_width, self.target_height)
        else:
            return frame
    
    def _fit_to_screen(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        max_w, max_h = self.max_display_size
        
        scale_w = max_w / w
        scale_h = max_h / h
        scale = min(scale_w, scale_h, 1.0)
        
        if scale == 1.0:
            return frame
            
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    def _resize_fixed(self, frame: np.ndarray, width: int, height: int) -> np.ndarray:
        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    
    def _resize_scale(self, frame: np.ndarray, scale: float) -> np.ndarray:
        if scale == 1.0:
            return frame
            
        h, w = frame.shape[:2]
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    def _resize_crop(self, frame: np.ndarray, width: int, height: int) -> np.ndarray:
        h, w = frame.shape[:2]
        
        target_aspect = width / height
        original_aspect = w / h
        
        if original_aspect > target_aspect:
            new_w = int(h * target_aspect)
            start_x = (w - new_w) // 2
            cropped = frame[:, start_x:start_x + new_w]
        else:
            new_h = int(w / target_aspect)
            start_y = (h - new_h) // 2
            cropped = frame[start_y:start_y + new_h, :]
        
        return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_AREA)
    
    def _resize_letterbox(self, frame: np.ndarray, width: int, height: int) -> np.ndarray:
        h, w = frame.shape[:2]
        
        scale = min(width / w, height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        result = np.zeros((height, width, 3), dtype=np.uint8)
        
        pad_x = (width - new_w) // 2
        pad_y = (height - new_h) // 2
        
        result[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        
        return result
    
    def get_resize_info(self) -> Dict:
        return {
            'method': self.resize_method,
            'target_size': (self.target_width, self.target_height),
            'current_scale': self.current_scale,
            'maintain_aspect_ratio': self.maintain_aspect_ratio,
            'max_display_size': self.max_display_size
        }
    
class RealTimeProcessor:
    def __init__(self, face_system, processing_interval: int = 5, buffer_size: int = 10):
        self.face_system = face_system
        self.cap = None
        self.fps = 0
        self.frame_count = 0
        self.processing_count = 0
        self.start_time = time.time()
        
        # Frame processing optimization
        self.processing_interval = processing_interval
        self.last_processed_time = 0
        self.min_processing_delay = 0.1
        
        # Threading for RTSP stability
        self.frame_queue = Queue(maxsize=buffer_size)
        self.latest_frame = None
        self.frame_lock = Lock()
        self.running = False
        self.capture_thread = None
        
        # RTSP configuration
        self.rtsp_url = None
        self.reconnect_delay = 5
        self.max_reconnect_attempts = 5
        
        # Enhanced display resizing
        self.resizer = DisplayResizer()
        self.show_resize_info = False
        self.original_frame_size = None
        
        # Debug controls
        self.debug_mode = False
        self.show_detection_debug = False
        self.show_performance_stats = True
        self.save_debug_frames = False
        self.debug_frame_count = 0
        self.max_debug_frames = 100
        
    def initialize_stream(self, source: str):
        """Initialize camera or RTSP stream with optimized settings"""
        self.rtsp_url = source
        
        if source.startswith('rtsp://') or source.startswith('http://'):
            print(f"📹 Initializing RTSP stream: {source}")
            self._initialize_rtsp_stream(source)
        else:
            try:
                camera_id = int(source)
                print(f"📹 Initializing camera: {camera_id}")
                self._initialize_camera(camera_id)
            except ValueError:
                print(f"📹 Initializing video source: {source}")
                self._initialize_video_source(source)

    def _initialize_camera(self, camera_id: int):
        """Initialize local camera"""
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_id}")

    def _initialize_rtsp_stream(self, rtsp_url: str):
        """Initialize RTSP stream with optimized parameters"""
        optimized_rtsp = self._optimize_rtsp_url(rtsp_url)
        self.cap = cv2.VideoCapture(optimized_rtsp)
        
        self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
        self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open RTSP stream: {rtsp_url}")

    def _initialize_video_source(self, source: str):
        """Initialize video file or other source"""
        self.cap = cv2.VideoCapture(source)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

    def _optimize_rtsp_url(self, rtsp_url: str) -> str:
        """Add optimization parameters to RTSP URL"""
        if '?' in rtsp_url:
            return rtsp_url + '&tcp=True&buffer_size=65535'
        else:
            return rtsp_url + '?tcp=True&buffer_size=65535'
        
    def start_frame_capture(self):
        """Start background thread for frame capture"""
        if self.cap is None:
            raise RuntimeError("Stream not initialized. Call initialize_stream first.")
        
        self.running = True
        self.capture_thread = Thread(target=self._capture_frames, daemon=True)
        self.capture_thread.start()
        print("🎬 Frame capture thread started")

    def _capture_frames(self):
        """Background thread for continuous frame capture"""
        reconnect_attempts = 0
        
        while self.running:
            try:
                ret, frame = self.cap.read()
                
                if not ret:
                    print("⚠️  Frame capture failed, attempting to reconnect...")
                    reconnect_attempts += 1
                    
                    if reconnect_attempts >= self.max_reconnect_attempts:
                        print("❌ Max reconnection attempts reached")
                        break
                    
                    time.sleep(self.reconnect_delay)
                    self._reconnect_stream()
                    continue
                
                reconnect_attempts = 0
                
                with self.frame_lock:
                    self.latest_frame = frame
                
                if not self.frame_queue.full():
                    self.frame_queue.put(frame, block=False)
                else:
                    try:
                        self.frame_queue.get(block=False)
                        self.frame_queue.put(frame, block=False)
                    except:
                        pass
                        
            except Exception as e:
                print(f"🚨 Capture thread error: {e}")
                time.sleep(1)

    def _reconnect_stream(self):
        """Attempt to reconnect to the stream"""
        try:
            if self.cap:
                self.cap.release()
            
            time.sleep(2)
            
            if self.rtsp_url:
                self._initialize_rtsp_stream(self.rtsp_url)
            else:
                print("⚠️  Camera reconnection not implemented")
                
        except Exception as e:
            print(f"❌ Reconnection failed: {e}")
            
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get the latest frame from the capture thread"""
        with self.frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None  

    def should_process_frame(self) -> bool:
        """Determine if current frame should be processed based on interval and timing"""
        current_time = time.time()
        
        if self.frame_count % self.processing_interval != 0:
            return False
        
        if current_time - self.last_processed_time < self.min_processing_delay:
            return False
        
        self.last_processed_time = current_time
        return True           
        
    def calculate_fps(self):
        """Calculate and update FPS"""
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            elapsed = time.time() - self.start_time
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.start_time = time.time()               
                                
        
    def set_display_size(self, width: int, height: int, method: str = "fixed_size"):
        """Set fixed display size"""
        self.resizer.target_width = width
        self.resizer.target_height = height
        self.resizer.resize_method = method
        print(f"🖼️  Display size set to {width}x{height} using {method} method")
    
    def set_display_scale(self, scale: float):
        """Set display scale factor"""
        self.resizer.current_scale = scale
        self.resizer.resize_method = "scale"
        print(f"🔍 Display scale set to {scale:.2f}")
    
    def set_display_method(self, method: str):
        """Set resizing method"""
        valid_methods = ["fit_to_screen", "fixed_size", "scale", "crop", "letterbox"]
        if method in valid_methods:
            self.resizer.resize_method = method
            print(f"🔄 Resize method set to: {method}")
        else:
            print(f"❌ Invalid resize method. Choose from: {valid_methods}")
    
    def set_max_display_size(self, width: int, height: int):
        """Set maximum display size for fit_to_screen method"""
        self.resizer.max_display_size = (width, height)
        print(f"📏 Maximum display size set to {width}x{height}")
    
    def toggle_resize_info(self):
        """Toggle resize information display"""
        self.show_resize_info = not self.show_resize_info
        status = "ON" if self.show_resize_info else "OFF"
        print(f"📊 Resize info display: {status}")

    # Debug control methods
    def toggle_debug_mode(self):
        """Toggle comprehensive debug mode"""
        self.debug_mode = not self.debug_mode
        status = "ON" if self.debug_mode else "OFF"
        print(f"🐛 Debug mode: {status}")
        
    def toggle_detection_debug(self):
        """Toggle detection visualization debug"""
        self.show_detection_debug = not self.show_detection_debug
        status = "ON" if self.show_detection_debug else "OFF"
        print(f"🎯 Detection debug: {status}")
        
    def toggle_performance_stats(self):
        """Toggle performance statistics display"""
        self.show_performance_stats = not self.show_performance_stats
        status = "ON" if self.show_performance_stats else "OFF"
        print(f"📈 Performance stats: {status}")
        
    def toggle_save_debug_frames(self):
        """Toggle saving debug frames"""
        self.save_debug_frames = not self.save_debug_frames
        status = "ON" if self.save_debug_frames else "OFF"
        print(f"💾 Save debug frames: {status}")
        
    def print_detailed_stats(self):
        """Print detailed system statistics"""
        stats = self.face_system.get_debug_stats()
        print("\n" + "="*50)
        print("📊 DETAILED SYSTEM STATISTICS")
        print("="*50)
        print(f"Total Frames Processed: {stats['total_frames_processed']}")
        print(f"Total Faces Detected: {stats['total_faces_detected']}")
        print(f"Total Faces Recognized: {stats['total_faces_recognized']}")
        print(f"Recognition Rate: {stats['recognition_rate']:.1f}%")
        print(f"Last Processing Time: {stats['last_processing_time']:.1f}ms")
        print(f"Avg Detection Time: {stats['avg_detection_time']:.1f}ms")
        print(f"Avg Embedding Time: {stats['avg_embedding_time']:.1f}ms")
        print(f"Avg Recognition Time: {stats['avg_recognition_time']:.1f}ms")
        print(f"Current FPS: {self.fps:.1f}")
        print(f"Processing Interval: 1/{self.processing_interval}")
        print("="*50)
    
    def resize_frame_for_display(self, frame: np.ndarray) -> np.ndarray:
        """Apply resizing to frame for display"""
        if self.original_frame_size is None:
            self.original_frame_size = frame.shape[:2]
        
        return self.resizer.resize_frame(frame)
    
    def draw_resize_info(self, frame: np.ndarray):
        """Display resize information on frame"""
        if not self.show_resize_info:
            return
        
        original_h, original_w = self.original_frame_size or frame.shape[:2]
        display_h, display_w = frame.shape[:2]
        
        info_lines = [
            f"Original: {original_w}x{original_h}",
            f"Display: {display_w}x{display_h}",
            f"Method: {self.resizer.resize_method}",
            f"Scale: {self.resizer.current_scale:.2f}" if self.resizer.resize_method == "scale" else ""
        ]
        
        info_lines = [line for line in info_lines if line.strip()]
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 130), (300, 130 + len(info_lines) * 25 + 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        for i, line in enumerate(info_lines):
            y_position = 150 + (i * 25)
            cv2.putText(frame, line, (20, y_position),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def draw_debug_info(self, frame: np.ndarray, results: List[Dict]):
        """Draw comprehensive debug information on frame"""
        if not self.debug_mode and not self.show_performance_stats:
            return
            
        stats = self.face_system.get_debug_stats()
        
        # Performance metrics
        performance_lines = []
        if self.show_performance_stats:
            performance_lines = [
                f"FPS: {self.fps:.1f}",
                f"Frame: {self.frame_count}",
                f"Processed: {self.processing_count}",
                f"Interval: 1/{self.processing_interval}",
                f"Recognition: {stats['recognition_rate']:.1f}%",
            ]
        
        # Debug information
        debug_lines = []
        if self.debug_mode:
            debug_lines = [
                f"Detection: {stats['avg_detection_time']:.1f}ms",
                f"Embedding: {stats['avg_embedding_time']:.1f}ms",
                f"Recognition: {stats['avg_recognition_time']:.1f}ms",
                f"Total Faces: {stats['total_faces_detected']}",
                f"Recognized: {stats['total_faces_recognized']}",
            ]
        
        all_lines = performance_lines + debug_lines
        if not all_lines:
            return
            
        # Draw background for all info
        overlay = frame.copy()
        start_y = 10
        end_y = start_y + len(all_lines) * 25 + 20
        cv2.rectangle(overlay, (10, start_y), (350, end_y), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw performance stats (green)
        for i, line in enumerate(performance_lines):
            y_position = 30 + (i * 25)
            cv2.putText(frame, line, (20, y_position),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw debug info (cyan)
        debug_start_y = 30 + len(performance_lines) * 25
        for i, line in enumerate(debug_lines):
            y_position = debug_start_y + (i * 25)
            cv2.putText(frame, line, (20, y_position),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    def draw_detection_debug(self, frame: np.ndarray, results: List[Dict]):
        """Draw detailed detection debugging information"""
        if not self.show_detection_debug:
            return
            
        for i, result in enumerate(results):
            x1, y1, x2, y2 = result['bbox']
            
            # Draw detailed information near each detection
            info_text = f"Det: {result['detection_confidence']:.2f}"
            if result['identity']:
                info_text += f" | Rec: {result['identity']} ({result['recognition_confidence']:.2f})"
            
            # Calculate position for debug text (below the bounding box)
            text_y = y2 + 20
            
            # Draw background for text
            text_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(frame, (x1, text_y - text_size[1] - 5), 
                         (x1 + text_size[0], text_y + 5), (0, 0, 0), -1)
            
            # Draw debug text
            cv2.putText(frame, info_text, (x1, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def draw_results(self, frame: np.ndarray, results: List[Dict]):
        """Enhanced visualization with debug support"""
        if self.original_frame_size is None:
            self.original_frame_size = frame.shape[:2]
        
        # Draw bounding boxes and labels
        for result in results:
            x1, y1, x2, y2 = result['bbox']
            identity = result['identity']
            rec_conf = result['recognition_confidence']
            det_conf = result['detection_confidence']
            
            if identity:
                color = (0, 255, 0)
                label = f"{identity} ({rec_conf:.2f})"
            else:
                color = (0, 0, 255)
                label = f"Unknown ({det_conf:.2f})"
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw debug information
        self.draw_debug_info(frame, results)
        self.draw_detection_debug(frame, results)
        
        # Draw resize information if enabled
        self.draw_resize_info(frame)
    
    def save_debug_frame(self, frame: np.ndarray, results: List[Dict]):
        """Save frame with debug information"""
        if not self.save_debug_frames or self.debug_frame_count >= self.max_debug_frames:
            return
            
        debug_frame = frame.copy()
        
        # Add timestamp and frame info
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        info_text = f"Frame_{self.debug_frame_count:04d}_{timestamp}"
        cv2.putText(debug_frame, info_text, (10, debug_frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Save frame
        filename = f"debug_frame_{self.debug_frame_count:04d}.jpg"
        cv2.imwrite(filename, debug_frame)
        self.debug_frame_count += 1
        
        if self.debug_frame_count % 10 == 0:
            print(f"💾 Saved debug frame: {filename}")
    
    def run(self, source: str = "0"):
        """Main processing loop with enhanced debug controls"""
        try:
            self.initialize_stream(source)
            self.start_frame_capture()
            
            print("🚀 Starting optimized real-time face recognition...")
            print(f"📊 Processing configuration:")
            print(f"   - Processing interval: 1 every {self.processing_interval} frames")
            print(f"   - Resize method: {self.resizer.resize_method}")
            print(f"   - Source: {source}")
            print(f"📋 Known identities: {self.face_system.get_known_identities()}")
            print("\n🎛️  CONTROLS:")
            print("  'q' - Quit")
            print("  's' - Save current frame")
            print("  '+' - Increase processing interval")
            print("  '-' - Decrease processing interval")
            print("  'r' - Reset processing counters")
            print("  'i' - Toggle resize info display")
            print("  'd' - Toggle debug mode")
            print("  'p' - Toggle performance stats")
            print("  'b' - Toggle detection debug")
            print("  'f' - Toggle save debug frames")
            print("  'x' - Print detailed statistics")
            print("  '1-8' - Resize methods")
            print("  '0' - Original size")
            
            last_results = []
            
            while self.running:
                frame = self.get_current_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                self.calculate_fps()
                
                original_frame = frame.copy()
                display_frame = self.resize_frame_for_display(frame)
                
                should_process = self.should_process_frame()
                results = last_results
                
                if should_process:
                    results = self.face_system.process_frame(original_frame)
                    self.processing_count += 1
                    last_results = results
                    
                    # Save debug frame if enabled
                    if self.save_debug_frames:
                        self.save_debug_frame(display_frame, results)
                
                self.draw_results(display_frame, results)
                cv2.imshow('Optimized Face Recognition System', display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = int(time.time())
                    cv2.imwrite(f'captured_frame_{timestamp}.jpg', display_frame)
                    print(f"💾 Frame saved: captured_frame_{timestamp}.jpg")
                elif key == ord('+'):
                    self.processing_interval = min(self.processing_interval + 1, 30)
                    print(f"⚙️  Processing interval increased to 1/{self.processing_interval}")
                elif key == ord('-'):
                    self.processing_interval = max(self.processing_interval - 1, 1)
                    print(f"⚙️  Processing interval decreased to 1/{self.processing_interval}")
                elif key == ord('r'):
                    self.frame_count = 0
                    self.processing_count = 0
                    self.start_time = time.time()
                    print("🔄 Processing counters reset")
                elif key == ord('i'):
                    self.toggle_resize_info()
                # Debug controls
                elif key == ord('d'):
                    self.toggle_debug_mode()
                elif key == ord('p'):
                    self.toggle_performance_stats()
                elif key == ord('b'):
                    self.toggle_detection_debug()
                elif key == ord('f'):
                    self.toggle_save_debug_frames()
                elif key == ord('x'):
                    self.print_detailed_stats()
                # Resize controls
                elif key == ord('1'):
                    self.set_display_method("fit_to_screen")
                elif key == ord('2'):
                    self.set_display_size(1280, 720, "fixed_size")
                elif key == ord('3'):
                    self.set_display_scale(0.5)
                elif key == ord('4'):
                    self.set_display_scale(0.75)
                elif key == ord('5'):
                    self.set_display_scale(1.0)
                elif key == ord('6'):
                    self.set_display_scale(1.5)
                elif key == ord('7'):
                    self.set_display_size(1280, 720, "crop")
                elif key == ord('8'):
                    self.set_display_size(1280, 720, "letterbox")
                elif key == ord('0'):
                    self.set_display_method("fit_to_screen")
                    self.set_max_display_size(3840, 2160)
                    print("📺 Displaying original size")
                        
        except Exception as e:
            print(f"❌ Error in main loop: {e}")
        finally:
            self.stop()
            
    def stop(self):
        """Cleanup resources"""
        self.running = False
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        # Print final statistics
        print("\n📊 FINAL STATISTICS:")
        self.print_detailed_stats()
        print("🛑 System stopped gracefully")            
                           
# Configuration with optimized settings
CONFIG = {
    'detection_model_path': r'D:\SCMA\3-APD\fromAraya\Computer-Vision-CV\3.1_FaceRecog\yolov11n-face.pt',
    'embeddings_db_path': r'D:\SCMA\3-APD\fromAraya\Computer-Vision-CV\3.1_FaceRecog\run_py\secondTry\person_folder1.json',
    'detection_confidence': 0.6,
    'detection_iou': 0.5,
    'roi_padding': 10,
    'embedding_model': 'Facenet',
    'recognition_threshold': 0.3,
}

def main():
    # Initialize system
    face_system = FaceRecognitionSystem(CONFIG)
    
    # Create processor with optimization
    processor = RealTimeProcessor(
        face_system=face_system,
        processing_interval=5,
        buffer_size=5
    )
    
        # Choose your input source
    sources = {
        '1': '0',                          # Default camera
        '2': 'rtsp://admin:Admin888@192.168.0.2:554/Streaming/Channels/101',  # RTSP
        '3': 'http://192.168.1.101:8080/video',                   # IP camera
        '4': 'video.mp4'                   # Video file
    }
    
    print("Available sources:")
    for key, source in sources.items():
        print(f"  {key}: {source}")
    
    choice = input("Select source (1-4) or enter custom RTSP URL: ").strip()
    
    if choice in sources:
        source = sources[choice]
    else:
        source = choice  # Custom input
    
    # Configure display
    processor.set_display_size(1024, 768, "fixed_size")
    
    try:
        processor.run(source)  # Use default camera
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        processor.stop()    

if __name__ == "__main__":
    main()