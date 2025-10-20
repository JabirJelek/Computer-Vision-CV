import json
import numpy as np
from deepface import DeepFace
from config import DEEPFACE_MODEL, DATABASE_PATH, THRESHOLD

class FaceRecognizer:
    def __init__(self):
        self.model_name = DEEPFACE_MODEL
        self.database = self.load_database()
        self.threshold = THRESHOLD
        
    def load_database(self):
        """Load pre-computed embeddings from JSON"""
        try:
            with open(DATABASE_PATH, 'r') as f:
                data = json.load(f)
            
            database = {}
            for person_id, person_data in data['persons'].items():
                database[person_id] = {
                    'display_name': person_data['display_name'],
                    'embedding': np.array(person_data['centroid_embedding']),
                    'folder_name': person_data['folder_name']
                }
            
            print(f"✅ Database loaded with {len(database)} persons")
            return database
        except Exception as e:
            print(f"❌ Error loading database: {e}")
            return {}
    
    def recognize_face(self, face_image):
        """Recognize face against database"""
        try:
            # Get embedding for the face
            embedding_obj = DeepFace.represent(
                face_image,
                model_name=self.model_name,
                enforce_detection=False,
                align=False  # YOLO already provides aligned faces
            )
            
            if not embedding_obj:
                return "Unknown", 0.0
            
            query_embedding = np.array(embedding_obj[0]['embedding'])
            
            # Find closest match in database
            best_match = "Unknown"
            best_distance = float('inf')
            
            for person_id, person_data in self.database.items():
                db_embedding = person_data['embedding']
                
                # Calculate cosine distance
                distance = self.cosine_distance(query_embedding, db_embedding)
                
                if distance < best_distance and distance < self.threshold:
                    best_distance = distance
                    best_match = person_data['display_name']
            
            # Convert distance to similarity score (0-1)
            similarity = max(0, 1 - best_distance) if best_match != "Unknown" else 0
            
            return best_match, similarity
            
        except Exception as e:
            print(f"❌ Recognition error: {e}")
            return "Unknown", 0.0
    
    def cosine_distance(self, embedding1, embedding2):
        """Calculate cosine distance between two embeddings"""
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        cosine_similarity = dot_product / (norm1 * norm2)
        return 1 - cosine_similarity