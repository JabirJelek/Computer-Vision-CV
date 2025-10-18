# debug_json_structure.py
import json

def analyze_json_structure(json_path):
    """Analyze the exact structure of your JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print("🔍 JSON STRUCTURE ANALYSIS")
    print(f"Top-level type: {type(data)}")
    
    if isinstance(data, list):
        print(f"List length: {len(data)}")
        if len(data) > 0:
            print("\nFirst item analysis:")
            for i, item in enumerate(data[:2]):  # First 2 items
                print(f"\n--- Item {i} ---")
                print(f"  Type: {type(item)}")
                if isinstance(item, dict):
                    print(f"  Keys: {list(item.keys())}")
                    for key, value in item.items():
                        print(f"    {key}: {type(value)}")
                        if key in ['embedding', 'representation']:
                            if isinstance(value, list):
                                print(f"      Length: {len(value)}")
                                print(f"      Sample first 5: {value[:5]}")
    elif isinstance(data, dict):
        print(f"Dictionary keys: {list(data.keys())}")
        for key, value in data.items():
            print(f"  {key}: {type(value)}")
            if isinstance(value, list):
                print(f"    List length: {len(value)}")
                if len(value) > 0:
                    print(f"    First item type: {type(value[0])}")

# Run the analysis
analyze_json_structure(r"D:\SCMA\3-APD\fromAraya\Computer-Vision-CV\3.1_FaceRecog\face_embeddings_dataset.json")