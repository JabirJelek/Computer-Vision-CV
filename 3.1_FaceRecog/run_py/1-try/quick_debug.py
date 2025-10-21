# quick_debug.py
import json

# quick_test.py
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def test_json_structure():
    """Test loading your specific JSON structure"""
    json_path = input("Enter the json path: ").strip()
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print("📁 JSON Structure:")
    print(f"Top-level keys: {list(data.keys())}")
    print(f"Metadata keys: {list(data.get('metadata', {}).keys())}")
    
    embeddings = data.get('embeddings', {})
    print(f"Number of embeddings: {len(embeddings)}")
    
    # Show first few filenames
    filenames = list(embeddings.keys())[:3]
    print(f"First 3 filenames: {filenames}")
    
    # Show structure of first embedding
    if filenames:
        first_key = filenames[0]
        first_embedding = embeddings[first_key]
        print(f"\n📊 First embedding structure:")
        print(f"  Keys: {list(first_embedding.keys())}")
        print(f"  Embedding length: {len(first_embedding.get('embedding', []))}")
        print(f"  File path: {first_embedding.get('file_path', 'N/A')}")
        
        # Test embedding conversion
        embedding_array = np.array(first_embedding['embedding'])
        print(f"  Numpy shape: {embedding_array.shape}")
        print(f"  Sample values: {embedding_array[:5]}")

if __name__ == "__main__":
    test_json_structure()