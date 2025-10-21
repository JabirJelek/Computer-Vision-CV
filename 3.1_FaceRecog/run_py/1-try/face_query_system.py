import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from deepface import DeepFace
import os
from typing import List, Dict, Any, Tuple
import time

class FaceQuerySystem:
    def __init__(self, embeddings_json_path: str):
        """
        Initialize the query system with pre-computed embeddings
        """
        self.embeddings_data = self._load_embeddings(embeddings_json_path)
        self.embeddings_array = self._prepare_embeddings_array()
        
    def _load_embeddings(self, json_path: str) -> List[Dict]:
        """Load embeddings from your specific JSON structure"""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        print(f"✅ Loaded JSON with metadata and {len(data.get('embeddings', {}))} embeddings")
        
        # Extract embeddings from your specific structure
        embeddings_dict = data.get('embeddings', {})
        validated_data = []
        
        for filename, embedding_data in embeddings_dict.items():
            if 'embedding' in embedding_data:
                validated_data.append({
                    'file_path': embedding_data.get('file_path', filename),
                    'embedding': embedding_data['embedding'],
                    'filename': filename,
                    'facial_area': embedding_data.get('facial_area', {}),
                    'face_confidence': embedding_data.get('face_confidence', 0.0),
                    'embedding_length': embedding_data.get('embedding_length', 0)
                })
        
        print(f"✅ Processed {len(validated_data)} valid face embeddings")
        return validated_data
    
    def _prepare_embeddings_array(self) -> np.ndarray:
        """Convert embeddings to numpy array"""
        if len(self.embeddings_data) == 0:
            raise ValueError("No embeddings data available!")
        
        embeddings = []
        for item in self.embeddings_data:
            embedding = item['embedding']
            # Ensure embedding is a flat list and convert to numpy
            embedding_array = np.array(embedding, dtype=np.float32).flatten()
            embeddings.append(embedding_array)
        
        print(f"✅ Created embeddings array with shape: {np.array(embeddings).shape}")
        return np.array(embeddings)
    
    def query_by_image(self, query_image_path: str, top_k: int = 5, similarity_threshold: float = 0.8) -> List[Dict]:
        """
        Query similar faces using an image path
        """
        try:
            print(f"🔍 Generating embedding for query image: {query_image_path}")
            query_embedding_obj = DeepFace.represent(query_image_path, model_name='Facenet', enforce_detection=False)  # Use Facenet to match your embeddings
            query_embedding = np.array(query_embedding_obj[0]['embedding']).reshape(1, -1)
            print(f"✅ Query embedding generated: {query_embedding.shape}")
        except Exception as e:
            print(f"❌ Error processing query image: {e}")
            return []
        
        return self._find_similar_faces(query_embedding, top_k, similarity_threshold)
    
    def query_by_embedding(self, query_embedding: np.ndarray, top_k: int = 5, similarity_threshold: float = 0.8) -> List[Dict]:
        """
        Query similar faces using a pre-computed embedding vector
        """
        query_embedding = query_embedding.reshape(1, -1)
        return self._find_similar_faces(query_embedding, top_k, similarity_threshold)
    
    def _find_similar_faces(self, query_embedding: np.ndarray, top_k: int, threshold: float) -> List[Dict]:
        """Core similarity search logic"""
        start_time = time.time()
        
        # Compute cosine similarities
        similarities = cosine_similarity(query_embedding, self.embeddings_array)[0]
        
        # Find top matches
        matches = []
        for i, similarity in enumerate(similarities):
            if similarity >= threshold:
                match_data = self.embeddings_data[i].copy()
                match_data['similarity_score'] = float(similarity)
                match_data['match_rank'] = len(matches) + 1
                matches.append(match_data)
        
        # Sort by similarity score (descending)
        matches.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Limit to top_k results
        top_matches = matches[:top_k]
        
        query_time = (time.time() - start_time) * 1000
        print(f"✅ Query completed in {query_time:.2f}ms - Found {len(top_matches)} matches above threshold {threshold}")
        
        return top_matches
    
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        return {
            "total_faces": len(self.embeddings_data),
            "embedding_dimension": self.embeddings_array.shape[1] if len(self.embeddings_array) > 0 else 0,
            "sample_files": [item['filename'] for item in self.embeddings_data[:3]]
        }
    
    def inspect_embeddings(self):
        """Inspect the loaded embeddings"""
        print("\n🔍 EMBEDDINGS INSPECTION:")
        print(f"Total embeddings: {len(self.embeddings_data)}")
        if len(self.embeddings_data) > 0:
            first = self.embeddings_data[0]
            print(f"First embedding - Filename: {first['filename']}")
            print(f"Embedding length: {len(first['embedding'])}")
            print(f"Embedding sample: {first['embedding'][:5]}...")
            print(f"File path: {first['file_path']}")     
        
        
        
        
        
        
import matplotlib.pyplot as plt
import cv2
from PIL import Image

class EnhancedFaceQuerySystem(FaceQuerySystem):
    def __init__(self, embeddings_json_path: str):
        super().__init__(embeddings_json_path)
    
    def visualize_matches(self, query_image_path: str, matches: List[Dict], max_display: int = 5):
        """
        Visualize query image and top matches
        """
        try:
            # Load query image
            query_img = cv2.imread(query_image_path)
            query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
            
            # Prepare subplots
            fig, axes = plt.subplots(1, min(len(matches), max_display) + 1, figsize=(15, 3))
            fig.suptitle('Face Similarity Search Results', fontsize=16)
            
            # Plot query image
            axes[0].imshow(query_img)
            axes[0].set_title('Query Image')
            axes[0].axis('off')
            
            # Plot matches
            for i, match in enumerate(matches[:max_display]):
                match_path = match.get('file_path', '')
                if os.path.exists(match_path):
                    match_img = cv2.imread(match_path)
                    match_img = cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB)
                    axes[i+1].imshow(match_img)
                    axes[i+1].set_title(f'Match {i+1}\nScore: {match["similarity_score"]:.3f}')
                    axes[i+1].axis('off')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"❌ Visualization error: {e}")
    
    def batch_query(self, query_images_dir: str, output_file: str = "batch_results.json"):
        """
        Process multiple query images and save results
        """
        results = {}
        
        for filename in os.listdir(query_images_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(query_images_dir, filename)
                print(f"🔍 Processing {filename}...")
                
                matches = self.query_by_image(image_path, top_k=3, similarity_threshold=0.8)
                results[filename] = {
                    'query_image': filename,
                    'matches': matches,
                    'timestamp': time.time()
                }
        
        # Save batch results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Batch query completed. Results saved to {output_file}")
        return results