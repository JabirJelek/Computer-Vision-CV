import os
import numpy as np
from test_face_recognition import FaceRecognitionTester
from bias_analyzer import BiasAnalyzer
from typing import Dict, List, Tuple

class BalancedFaceTester(FaceRecognitionTester):
    """
    Enhanced tester with bias correction techniques
    """
    
    def __init__(self, dataset_json: str):
        super().__init__(dataset_json)
        self.bias_weights = self._calculate_bias_weights()
    
    def _calculate_bias_weights(self):
        """Calculate weights to balance over-represented persons"""
        weights = {}
        total_embeddings = self.dataset['metadata']['total_embeddings']
        total_persons = len(self.dataset['persons'])
        
        # Ideal: each person should have equal representation
        ideal_per_person = total_embeddings / total_persons
        
        for person_id, person_data in self.dataset['persons'].items():
            actual_count = person_data['successful_embeddings']
            
            # Weight inversely proportional to over-representation
            # More embeddings = lower weight
            weight = ideal_per_person / actual_count if actual_count > 0 else 0
            weights[person_id] = min(weight, 2.0)  # Cap at 2x to avoid over-compensation
        
        print("⚖️  Calculated bias correction weights")
        return weights
    
    def balanced_similarity_max(self, query_embedding: np.ndarray, 
                              person_embeddings: List[np.ndarray], 
                              person_id: str) -> float:
        """
        Max similarity with bias correction
        """
        similarities = [self.cosine_similarity(query_embedding, emb) 
                       for emb in person_embeddings]
        raw_score = max(similarities) if similarities else 0.0
        
        # Apply bias correction
        corrected_score = raw_score * self.bias_weights.get(person_id, 1.0)
        
        return corrected_score
    
    def balanced_similarity_centroid(self, query_embedding: np.ndarray,
                                   person_centroid: np.ndarray,
                                   person_id: str) -> float:
        """
        Centroid similarity with bias correction
        """
        raw_score = self.cosine_similarity(query_embedding, person_centroid)
        
        # Apply bias correction
        corrected_score = raw_score * self.bias_weights.get(person_id, 1.0)
        
        return corrected_score
    
    def query_person_balanced(self, query_embedding: np.ndarray,
                            strategy: str = "balanced_max") -> List[Tuple[str, float, Dict]]:
        """
        Query with bias-corrected strategies
        """
        results = []
        query_embedding = np.array(query_embedding)
        
        for person_id, person_data in self.dataset["persons"].items():
            if strategy == "balanced_max":
                person_embeddings = [np.array(emb["vector"]) for emb in person_data["embeddings"]]
                score = self.balanced_similarity_max(query_embedding, person_embeddings, person_id)
            
            elif strategy == "balanced_centroid":
                centroid = np.array(person_data["centroid_embedding"])
                score = self.balanced_similarity_centroid(query_embedding, centroid, person_id)
            
            elif strategy == "normalized_max":
                # Alternative: normalize by embedding count
                person_embeddings = [np.array(emb["vector"]) for emb in person_data["embeddings"]]
                raw_score = max([self.cosine_similarity(query_embedding, emb) 
                               for emb in person_embeddings] or [0])
                # Penalize over-representation
                embedding_count = len(person_embeddings)
                avg_embeddings = self.dataset['metadata']['average_embeddings_per_person']
                penalty = min(1.0, avg_embeddings / embedding_count) if embedding_count > avg_embeddings else 1.0
                score = raw_score * penalty
            
            else:
                # Fall back to parent method
                return super().query_person(query_embedding, strategy)
            
            results.append((person_id, score, person_data))
        
        # Sort by corrected score
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def compare_biased_vs_balanced(self, image_path: str):
        """
        Compare original vs balanced matching for a specific image
        """
        from deepface import DeepFace
        
        try:
            # Generate query embedding
            embedding_obj = DeepFace.represent(
                img_path=image_path,
                model_name='Facenet',
                enforce_detection=False
            )
            query_embedding = embedding_obj[0]['embedding']
            
            # Test different strategies
            strategies = [
                ("max_similarity", "Original Max"),
                ("centroid", "Original Centroid"), 
                ("balanced_max", "Balanced Max"),
                ("balanced_centroid", "Balanced Centroid")
            ]
            
            print(f"\n🔄 COMPARISON FOR: {os.path.basename(image_path)}")
            print("=" * 60)
            
            for strategy, description in strategies:
                if "balanced" in strategy:
                    matches = self.query_person_balanced(query_embedding, strategy)
                else:
                    matches = self.query_person(query_embedding, strategy)
                
                print(f"\n{description}:")
                for i, (person_id, score, person_data) in enumerate(matches[:3]):
                    emb_count = person_data['successful_embeddings']
                    print(f"   {i+1}. {person_data['display_name']:20} "
                          f"Score: {score:.4f} (Emb: {emb_count})")
        
        except Exception as e:
            print(f"❌ Error: {e}")

# 🎯 Quick Test for Your False Positive Case
def debug_false_positive(dataset_json: str, query_image_path: str, expected_person: str):
    """
    Debug why a specific false positive occurred
    """
    tester = BalancedFaceTester(dataset_json)
    
    print("🔍 DEBUGGING FALSE POSITIVE CASE")
    print("=" * 50)
    
    # Analyze the dataset first
    analyzer = BiasAnalyzer(dataset_json)
    analyzer.analyze_embedding_distribution()
    
    # Compare strategies
    tester.compare_biased_vs_balanced(query_image_path)
    
    # Show details about the expected person
    expected_data = None
    for person_id, person_data in tester.dataset["persons"].items():
        if person_data["display_name"].lower() == expected_person.lower():
            expected_data = person_data
            break
    
    if expected_data:
        print(f"\n📋 EXPECTED PERSON DETAILS:")
        print(f"   Name: {expected_data['display_name']}")
        print(f"   Embeddings: {expected_data['successful_embeddings']}")
        print(f"   Folder: {expected_data['folder_name']}")

def main():
    dataset_json = input("Enter dataset path: ").strip()
    query_image = input("Enter data path that will be used to compare with: ").strip()  # Update this
    expected_person = input("Enter expected person from the query image: ").strip()  # Update with actual name
    
    debug_false_positive(dataset_json, query_image, expected_person)

if __name__ == "__main__":
    main()