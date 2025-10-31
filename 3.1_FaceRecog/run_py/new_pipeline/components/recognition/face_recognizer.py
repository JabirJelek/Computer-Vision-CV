# components/recognition/face_recognizer.py
from core.base_processor import BaseProcessor
from deepface import DeepFace
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import time
from sklearn.metrics.pairwise import cosine_similarity
import cv2

class FaceRecognizer(BaseProcessor):
    """
    Face recognition component with embedding extraction and identity matching
    Integrates DeepFace with our modular pipeline architecture
    """
    
    def __init__(self, name: str, config: Dict):
        super().__init__(name, config)
        self.embedding_model = None
        self.embeddings_db = {}
        self.identity_centroids = {}
        self.similarity_threshold = 0.6
        
    def _initialize_impl(self) -> bool:
        """Initialize face recognition model and load embeddings database"""
        try:
            # Load embeddings database
            db_path = self.config.get('embeddings_db_path')
            if not db_path:
                print("❌ FaceRecognizer: No embeddings_db_path in config")
                return False
                
            if not self._load_embeddings_database(db_path):
                print("❌ FaceRecognizer: Failed to load embeddings database")
                return False
            
            # Set similarity threshold
            self.similarity_threshold = self.config.get('recognition_threshold', 0.6)
            
            # Initialize embedding model (will be loaded by DeepFace on first use)
            self.embedding_model = self.config.get('embedding_model', 'Facenet')
            
            print(f"✅ FaceRecognizer initialized successfully")
            print(f"   - Database: {len(self.identity_centroids)} identities")
            print(f"   - Model: {self.embedding_model}")
            print(f"   - Threshold: {self.similarity_threshold}")
            
            return True
            
        except Exception as e:
            print(f"❌ FaceRecognizer initialization failed: {e}")
            return False
    
    def _load_embeddings_database(self, db_path: str) -> bool:
        """Load embeddings database from JSON file"""
        try:
            path = Path(db_path)
            if not path.exists():
                print(f"⚠️ FaceRecognizer: Database file not found: {db_path}")
                # Create empty database structure
                self.embeddings_db = {"persons": {}, "metadata": {}}
                self.identity_centroids = {}
                return True
                
            with open(path, 'r') as f:
                self.embeddings_db = json.load(f)
            
            # Extract identity centroids for efficient matching
            self.identity_centroids = {}
            
            if "persons" in self.embeddings_db:
                for person_id, person_data in self.embeddings_db["persons"].items():
                    display_name = person_data.get("display_name", f"Person_{person_id}")
                    centroid = person_data.get("centroid_embedding")
                    if centroid is not None:
                        self.identity_centroids[display_name] = np.array(centroid)
            
            print(f"📊 Loaded {len(self.identity_centroids)} identities from database")
            if self.identity_centroids:
                print(f"   - Identities: {list(self.identity_centroids.keys())}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load embeddings database: {e}")
            return False
    
    def _process_impl(self, data: Dict, context: Dict) -> Dict:
        """Perform face recognition on detected faces"""
        if 'detections' not in data or not data['detections']:
            # No detections to process
            return data
        
        frame = data.get('input_frame')
        if frame is None:
            print("⚠️ FaceRecognizer: No input frame available")
            return data
        
        recognitions = []
        
        for detection in data['detections']:
            try:
                # Extract face ROI
                face_roi = self._extract_face_roi(frame, detection['bbox'])
                if face_roi is None:
                    continue
                
                # Extract embedding
                embedding = self._extract_embedding(face_roi)
                if embedding is None:
                    continue
                
                # Recognize face
                identity, confidence, similarity_scores = self._recognize_face(embedding)
                
                recognition_result = {
                    'bbox': detection['bbox'],
                    'detection_confidence': detection['confidence'],
                    'identity': identity,
                    'recognition_confidence': confidence,
                    'embedding': embedding.tolist(),  # Convert to list for JSON serialization
                    'similarity_scores': similarity_scores,
                    'embedding_model': self.embedding_model
                }
                
                recognitions.append(recognition_result)
                
            except Exception as e:
                print(f"⚠️ FaceRecognizer: Error processing detection: {e}")
                continue
        
        # Add recognitions to pipeline data
        data['recognitions'] = recognitions
        data['recognition_metadata'] = {
            'total_processed': len(recognitions),
            'recognized_count': len([r for r in recognitions if r['identity'] is not None]),
            'timestamp': time.time()
        }
        
        return data
    
    def _extract_face_roi(self, frame: np.ndarray, bbox: List[int]) -> Optional[np.ndarray]:
        """Extract face region of interest with padding and validation"""
        try:
            x1, y1, x2, y2 = bbox
            h, w = frame.shape[:2]
            
            # Add padding
            padding = self.config.get('roi_padding', 20)
            x1_pad = max(0, x1 - padding)
            y1_pad = max(0, y1 - padding)
            x2_pad = min(w, x2 + padding)
            y2_pad = min(h, y2 + padding)
            
            face_roi = frame[y1_pad:y2_pad, x1_pad:x2_pad]
            
            # Validate ROI
            if (face_roi.size == 0 or face_roi.shape[0] < 40 or 
                face_roi.shape[1] < 40 or np.std(face_roi) < 10):
                return None
            
            return face_roi
            
        except Exception as e:
            print(f"⚠️ FaceRecognizer: Error extracting face ROI: {e}")
            return None
    
    def _extract_embedding(self, face_roi: np.ndarray) -> Optional[np.ndarray]:
        """Extract face embedding using DeepFace"""
        try:
            # Convert to RGB and normalize
            if len(face_roi.shape) == 3:
                face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            else:
                face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2RGB)
            
            face_rgb = face_rgb.astype(np.float32) / 255.0
            
            # Extract embedding using DeepFace
            embedding_obj = DeepFace.represent(
                face_rgb,
                model_name=self.embedding_model,
                enforce_detection=False,
                detector_backend='skip',
                align=True
            )
            
            if embedding_obj and len(embedding_obj) > 0:
                embedding_array = np.array(embedding_obj[0]['embedding'])
                
                # Validate embedding quality
                if (np.any(np.isnan(embedding_array)) or 
                    np.any(np.isinf(embedding_array)) or
                    np.linalg.norm(embedding_array) < 0.001):
                    return None
                
                return embedding_array
                
        except Exception as e:
            if self.config.get('verbose', False):
                print(f"⚠️ FaceRecognizer: Embedding extraction failed: {e}")
        
        return None
    
    def _recognize_face(self, embedding: np.ndarray) -> Tuple[Optional[str], float, Dict]:
        """Recognize face by comparing with known identities"""
        if not self.identity_centroids:
            return None, 0.0, {}
        
        best_similarity = -1.0
        best_identity = None
        similarity_scores = {}
        
        embedding = embedding.flatten()
        
        for identity, centroid in self.identity_centroids.items():
            centroid = centroid.flatten()
            
            # Compute cosine similarity
            cosine_sim = cosine_similarity([embedding], [centroid])[0][0]
            
            # Optional: Compute Euclidean distance and convert to similarity
            euclidean_dist = np.linalg.norm(embedding - centroid)
            euclidean_sim = 1 / (1 + euclidean_dist)  # Convert distance to similarity
            
            # Combined similarity score
            final_similarity = 0.8 * cosine_sim + 0.2 * euclidean_sim
            
            similarity_scores[identity] = {
                'cosine': cosine_sim,
                'euclidean': euclidean_sim,
                'combined': final_similarity
            }
            
            if final_similarity > best_similarity and final_similarity >= self.similarity_threshold:
                best_similarity = final_similarity
                best_identity = identity
        
        return best_identity, best_similarity, similarity_scores
    
    def get_known_identities(self) -> List[str]:
        """Get list of all known identities"""
        return list(self.identity_centroids.keys())
    
    def add_identity(self, name: str, embedding: np.ndarray) -> bool:
        """Add new identity to the database (for auto-enrollment)"""
        try:
            if name in self.identity_centroids:
                print(f"⚠️ Identity '{name}' already exists, updating...")
            
            self.identity_centroids[name] = embedding
            
            # Update embeddings database structure
            if "persons" not in self.embeddings_db:
                self.embeddings_db["persons"] = {}
            
            # Generate person ID
            next_id = len(self.embeddings_db["persons"]) + 1
            person_id = f"person_{next_id:03d}"
            
            # Create person entry
            self.embeddings_db["persons"][person_id] = {
                "person_id": person_id,
                "folder_name": "AutoEnrolled",
                "display_name": name,
                "embeddings": [{
                    "vector": embedding.tolist(),
                    "source_file": f"auto_enrolled_{int(time.time())}.jpg",
                    "file_path": f"auto_enrolled/{name}/face_{int(time.time())}.jpg",
                    "embedding_length": len(embedding)
                }],
                "total_images": 1,
                "successful_embeddings": 1,
                "centroid_embedding": embedding.tolist()
            }
            
            print(f"✅ Added new identity: {name}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to add identity '{name}': {e}")
            return False
    
    def save_database(self) -> bool:
        """Save current embeddings database to file"""
        try:
            db_path = self.config.get('embeddings_db_path')
            if not db_path:
                print("❌ No database path configured")
                return False
            
            with open(db_path, 'w') as f:
                json.dump(self.embeddings_db, f, indent=2)
            
            print(f"💾 Database saved: {db_path}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to save database: {e}")
            return False
    
    def get_recognition_stats(self) -> Dict[str, Any]:
        """Get recognition component statistics"""
        return {
            'known_identities': len(self.identity_centroids),
            'similarity_threshold': self.similarity_threshold,
            'embedding_model': self.embedding_model,
            'identity_names': list(self.identity_centroids.keys())
        }