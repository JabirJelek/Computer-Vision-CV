import json
import numpy as np
import os
import time
from typing import List, Dict, Any, Tuple
from deepface import DeepFace
from pathlib import Path

class FaceRecognitionTester:
    def __init__(self, dataset_json: str):
        """
        Flexible tester for your face recognition dataset
        
        Args:
            dataset_json: Path to your enhanced JSON dataset
        """
        self.dataset = self._load_dataset(dataset_json)
        self.results = {}
        self.test_config = {}
        
    def _load_dataset(self, dataset_json: str) -> Dict:
        """Load the enhanced dataset"""
        with open(dataset_json, 'r') as f:
            data = json.load(f)
        
        print(f"✅ Loaded dataset: {data['metadata']['total_persons']} persons, "
              f"{data['metadata']['total_embeddings']} embeddings")
        return data
    
    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def match_strategy_max_similarity(self, query_embedding: np.ndarray, 
                                    person_embeddings: List[np.ndarray]) -> float:
        """
        Strategy 1: Use maximum similarity from any embedding
        Good for: Testing if any angle matches strongly
        """
        similarities = [self.cosine_similarity(query_embedding, emb) 
                       for emb in person_embeddings]
        return max(similarities) if similarities else 0.0
    
    def match_strategy_centroid(self, query_embedding: np.ndarray, 
                              person_centroid: np.ndarray) -> float:
        """
        Strategy 2: Compare against average embedding (centroid)
        Good for: Stable, consistent matching
        """
        return self.cosine_similarity(query_embedding, person_centroid)
    
    def match_strategy_average_similarity(self, query_embedding: np.ndarray, 
                                        person_embeddings: List[np.ndarray]) -> float:
        """
        Strategy 3: Average similarity across all embeddings
        Good for: Overall consistency testing
        """
        similarities = [self.cosine_similarity(query_embedding, emb) 
                       for emb in person_embeddings]
        return np.mean(similarities) if similarities else 0.0
    
    def query_person(self, query_embedding: np.ndarray, 
                    strategy: str = "max_similarity") -> List[Tuple[str, float, Dict]]:
        """
        Query all persons in dataset using specified strategy
        
        Returns: List of (person_id, similarity_score, person_data) sorted by score
        """
        results = []
        query_embedding = np.array(query_embedding)
        
        for person_id, person_data in self.dataset["persons"].items():
            if strategy == "max_similarity":
                person_embeddings = [np.array(emb["vector"]) for emb in person_data["embeddings"]]
                score = self.match_strategy_max_similarity(query_embedding, person_embeddings)
            
            elif strategy == "centroid":
                centroid = np.array(person_data["centroid_embedding"])
                score = self.match_strategy_centroid(query_embedding, centroid)
            
            elif strategy == "average_similarity":
                person_embeddings = [np.array(emb["vector"]) for emb in person_data["embeddings"]]
                score = self.match_strategy_average_similarity(query_embedding, person_embeddings)
            
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            results.append((person_id, score, person_data))
        
        # Sort by similarity score (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def test_single_image(self, image_path: str, 
                         strategy: str = "max_similarity",
                         top_k: int = 5) -> Dict[str, Any]:
        """
        Test a single image against the dataset
        """
        print(f"🔍 Testing image: {os.path.basename(image_path)}")
        
        try:
            # Generate embedding for query image
            embedding_obj = DeepFace.represent(
                img_path=image_path,
                model_name='Facenet',
                enforce_detection=False
            )
            query_embedding = embedding_obj[0]['embedding']
            
            # Query dataset
            start_time = time.time()
            matches = self.query_person(query_embedding, strategy)
            query_time = time.time() - start_time
            
            # Prepare results
            result = {
                "query_image": image_path,
                "strategy": strategy,
                "query_time_ms": round(query_time * 1000, 2),
                "top_matches": [],
                "all_matches": []
            }
            
            # Top K matches
            for i, (person_id, score, person_data) in enumerate(matches[:top_k]):
                match_info = {
                    "rank": i + 1,
                    "person_id": person_id,
                    "similarity_score": round(score, 4),
                    "person_name": person_data["display_name"],
                    "folder_name": person_data["folder_name"],
                    "embedding_count": person_data["successful_embeddings"]
                }
                result["top_matches"].append(match_info)
            
            # All matches (for analysis)
            for person_id, score, person_data in matches:
                result["all_matches"].append({
                    "person_id": person_id,
                    "similarity_score": round(score, 4)
                })
            
            return result
            
        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")
            return {"error": str(e), "query_image": image_path}
    
    def test_known_person(self, person_id: str, 
                         strategy: str = "max_similarity") -> Dict[str, Any]:
        """
        Test self-recognition: Use one of person's own images as query
        Good for: Testing if person recognizes themselves
        """
        if person_id not in self.dataset["persons"]:
            return {"error": f"Person {person_id} not found in dataset"}
        
        person_data = self.dataset["persons"][person_id]
        
        if not person_data["embeddings"]:
            return {"error": f"No embeddings for person {person_id}"}
        
        # Use first embedding as query (simulating one of their own images)
        query_embedding = np.array(person_data["embeddings"][0]["vector"])
        
        # Query dataset (should match themselves highly)
        matches = self.query_person(query_embedding, strategy)
        
        # Find where the actual person ranks
        actual_rank = None
        for rank, (match_id, score, _) in enumerate(matches, 1):
            if match_id == person_id:
                actual_rank = rank
                break
        
        return {
            "test_type": "self_recognition",
            "query_person": person_id,
            "strategy": strategy,
            "actual_rank": actual_rank,
            "top_score": matches[0][1] if matches else 0,
            "self_score": matches[actual_rank-1][1] if actual_rank else 0,
            "top_5_matches": [
                {"person_id": pid, "score": round(score, 4)} 
                for pid, score, _ in matches[:5]
            ]
        }
    
    def compare_strategies(self, image_path: str) -> Dict[str, Any]:
        """
        Compare all matching strategies for a single image
        """
        strategies = ["max_similarity", "centroid", "average_similarity"]
        comparison = {"query_image": image_path, "strategies": {}}
        
        for strategy in strategies:
            result = self.test_single_image(image_path, strategy)
            if "error" not in result:
                comparison["strategies"][strategy] = {
                    "top_match": result["top_matches"][0] if result["top_matches"] else None,
                    "query_time_ms": result["query_time_ms"]
                }
        
        return comparison
    
    def run_comprehensive_test(self, test_images_dir: str = None) -> Dict[str, Any]:
        """
        Run comprehensive tests on the dataset
        """
        print("🧪 Running Comprehensive Tests")
        print("=" * 50)
        
        comprehensive_results = {
            "test_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_info": self.dataset["metadata"],
            "self_recognition_tests": [],
            "strategy_comparisons": [],
            "single_image_tests": []
        }
        
        # Test 1: Self-recognition for each person
        print("\n1. 🤳 Self-Recognition Tests")
        for person_id in self.dataset["persons"].keys():
            result = self.test_known_person(person_id)
            comprehensive_results["self_recognition_tests"].append(result)
            
            status = "✅" if result.get("actual_rank") == 1 else "❌"
            print(f"   {status} {person_id}: Rank {result.get('actual_rank')}, "
                  f"Score {result.get('self_score', 0):.4f}")
        
        # Test 2: Strategy comparison (if test images provided)
        if test_images_dir and os.path.exists(test_images_dir):
            print(f"\n2. 🔄 Strategy Comparison Tests")
            test_images = [f for f in os.listdir(test_images_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for image_file in test_images[:3]:  # Test first 3 images
                image_path = os.path.join(test_images_dir, image_file)
                comparison = self.compare_strategies(image_path)
                comprehensive_results["strategy_comparisons"].append(comparison)
                
                print(f"   📊 {image_file}:")
                for strategy, data in comparison["strategies"].items():
                    top_match = data["top_match"]
                    if top_match:
                        print(f"      {strategy}: {top_match['person_name']} "
                              f"(score: {top_match['similarity_score']})")
        
        # Test 3: Performance benchmark
        print(f"\n3. ⚡ Performance Benchmark")
        if len(self.dataset["persons"]) > 0:
            sample_person = list(self.dataset["persons"].keys())[0]
            sample_embedding = np.array(self.dataset["persons"][sample_person]["embeddings"][0]["vector"])
            
            times = []
            for _ in range(10):
                start = time.time()
                self.query_person(sample_embedding, "max_similarity")
                times.append(time.time() - start)
            
            avg_time = np.mean(times) * 1000
            comprehensive_results["performance_benchmark"] = {
                "avg_query_time_ms": round(avg_time, 2),
                "queries_per_second": round(1000 / avg_time, 2),
                "dataset_size": len(self.dataset["persons"])
            }
            
            print(f"   Avg query time: {avg_time:.2f}ms")
            print(f"   Queries per second: {1000/avg_time:.1f}")
        
        return comprehensive_results
    
    def save_results(self, results: Dict, output_file: str = "test_results.json"):
        """Save test results to JSON"""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Test results saved to: {output_file}")
    
    def print_quick_stats(self):
        """Print quick dataset statistics"""
        stats = self.dataset["metadata"]
        persons = self.dataset["persons"]
        
        print("\n📊 Dataset Quick Stats:")
        print(f"   Total persons: {stats['total_persons']}")
        print(f"   Total embeddings: {stats['total_embeddings']}")
        print(f"   Avg embeddings per person: {stats['average_embeddings_per_person']}")
        
        # Embedding distribution
        counts = [p['successful_embeddings'] for p in persons.values()]
        print(f"   Min embeddings per person: {min(counts)}")
        print(f"   Max embeddings per person: {max(counts)}")

# 🎯 USAGE EXAMPLES
def main():
    # Configuration - UPDATE THESE PATHS
    DATASET_JSON = input("Please enter dataset path: ").strip()  # Your enhanced dataset
    TEST_IMAGES_DIR = input("Please enter test image folder (optional):").strip()  # Optional: folder with test images
    
    # Initialize tester
    tester = FaceRecognitionTester(DATASET_JSON)
    
    # Show quick stats
    tester.print_quick_stats()
    
    while True:
        print("\n🎯 TEST MENU:")
        print("1. Test single image")
        print("2. Test self-recognition for a person")
        print("3. Compare strategies for an image")
        print("4. Run comprehensive tests")
        print("5. Exit")
        
        choice = input("\nChoose option (1-5): ").strip()
        
        if choice == '1':
            image_path = input("Enter image path: ").strip()
            if os.path.exists(image_path):
                strategy = input("Strategy (max_similarity/centroid/average_similarity) [max_similarity]: ").strip() or "max_similarity"
                result = tester.test_single_image(image_path, strategy)
                
                print(f"\n📋 Results for {os.path.basename(image_path)}:")
                for match in result["top_matches"]:
                    print(f"   #{match['rank']}: {match['person_name']} (score: {match['similarity_score']})")
        
        elif choice == '2':
            person_id = input("Enter person ID (e.g., person_001): ").strip()
            result = tester.test_known_person(person_id)
            
            if "error" not in result:
                print(f"\n🤳 Self-Recognition Test for {person_id}:")
                print(f"   Rank: {result['actual_rank']}")
                print(f"   Self-match score: {result['self_score']:.4f}")
                print(f"   Top match score: {result['top_score']:.4f}")
        
        elif choice == '3':
            image_path = input("Enter image path: ").strip()
            if os.path.exists(image_path):
                comparison = tester.compare_strategies(image_path)
                print(f"\n🔄 Strategy Comparison for {os.path.basename(image_path)}:")
                for strategy, data in comparison["strategies"].items():
                    match = data["top_match"]
                    if match:
                        print(f"   {strategy:>18}: {match['person_name']} (score: {match['similarity_score']})")
        
        elif choice == '4':
            print("Running comprehensive tests...")
            results = tester.run_comprehensive_test(TEST_IMAGES_DIR)
            tester.save_results(results, "comprehensive_test_results.json")
        
        elif choice == '5':
            print("👋 Exiting...")
            break
        
        else:
            print("❌ Invalid option")

if __name__ == "__main__":
    main()