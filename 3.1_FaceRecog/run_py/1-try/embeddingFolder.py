# Import required libraries
from deepface import DeepFace
import numpy as np
import json
import os
import glob
from pathlib import Path
import time

def get_supported_image_extensions():
    """Return list of supported image file extensions."""
    return ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']

def find_image_files(folder_path):
    """
    Find all image files in the specified folder.
    
    Parameters:
    folder_path (str): Path to the folder containing images
    
    Returns:
    list: List of image file paths
    """
    image_files = []
    supported_extensions = get_supported_image_extensions()
    
    for ext in supported_extensions:
        # Case insensitive search for image files
        pattern = os.path.join(folder_path, f"*{ext}")
        image_files.extend(glob.glob(pattern))
        pattern = os.path.join(folder_path, f"*{ext.upper()}")
        image_files.extend(glob.glob(pattern))
    
    return sorted(image_files)

def create_embedding_for_image(image_path, model_name="Facenet", enforce_detection=False):
    """
    Create embedding for a single image.
    
    Parameters:
    image_path (str): Path to the image file
    model_name (str): Model to use for embedding
    enforce_detection (bool): Whether to require face detection
    
    Returns:
    dict: Embedding results or None if failed
    """
    try:
        print(f"🔄 Processing: {os.path.basename(image_path)}")
        
        # For cropped faces, we might not need face detection
        embedding_objs = DeepFace.represent(
            img_path=image_path,
            model_name=model_name,
            detector_backend="opencv",  # Simple detector for cropped faces
            enforce_detection=enforce_detection,  # Set to False for pre-cropped faces
            align=True
        )
        
        # For cropped faces, we expect exactly one face
        if len(embedding_objs) == 1:
            return embedding_objs[0]
        elif len(embedding_objs) > 1:
            print(f"⚠️  Warning: Multiple faces detected in {os.path.basename(image_path)}")
            # Return the first face (most prominent)
            return embedding_objs[0]
        else:
            print(f"❌ No faces detected in {os.path.basename(image_path)}")
            return None
            
    except Exception as e:
        print(f"❌ Error processing {os.path.basename(image_path)}: {str(e)}")
        return None

def process_folder_embeddings(folder_path, model_name="Facenet", output_file="folder_embeddings.json", enforce_detection=False):
    """
    Process all images in a folder and create embeddings.
    
    Parameters:
    folder_path (str): Path to the folder containing images
    model_name (str): Model to use for embeddings
    output_file (str): JSON file to save results
    enforce_detection (bool): Whether to require face detection
    
    Returns:
    dict: Dictionary containing all embeddings and metadata
    """
    
    print("🚀 STARTING FOLDER EMBEDDING PIPELINE")
    print("=" * 60)
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"❌ Error: Folder '{folder_path}' does not exist.")
        return None
    
    # Find all image files
    image_files = find_image_files(folder_path)
    
    if not image_files:
        print(f"❌ No image files found in '{folder_path}'")
        print(f"   Supported extensions: {', '.join(get_supported_image_extensions())}")
        return None
    
    print(f"📁 Folder: {folder_path}")
    print(f"📊 Found {len(image_files)} image files")
    print(f"🤖 Using model: {model_name}")
    print("-" * 60)
    
    # Results storage
    results = {
        "metadata": {
            "folder_path": folder_path,
            "model_name": model_name,
            "total_images": len(image_files),
            "processed_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "enforce_detection": enforce_detection
        },
        "embeddings": {}
    }
    
    # Statistics
    successful = 0
    failed = 0
    
    # Process each image
    start_time = time.time()
    
    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing: {os.path.basename(image_path)}")
        
        # Create embedding
        embedding_result = create_embedding_for_image(
            image_path, 
            model_name=model_name,
            enforce_detection=enforce_detection
        )
        
        if embedding_result:
            # Store the embedding with filename as key
            filename = os.path.basename(image_path)
            results["embeddings"][filename] = {
                "embedding": [float(x) for x in embedding_result['embedding']],  # Convert to list for JSON
                "embedding_length": len(embedding_result['embedding']),
                "facial_area": embedding_result.get('facial_area', {}),
                "face_confidence": embedding_result.get('face_confidence', None),
                "file_path": image_path
            }
            successful += 1
            print(f"✅ Success: {filename}")
        else:
            failed += 1
            print(f"❌ Failed: {os.path.basename(image_path)}")
    
    # Calculate processing time
    processing_time = time.time() - start_time
    
    # Update metadata with results
    results["metadata"]["successful_embeddings"] = successful
    results["metadata"]["failed_embeddings"] = failed
    results["metadata"]["processing_time_seconds"] = round(processing_time, 2)
    results["metadata"]["average_time_per_image"] = round(processing_time / len(image_files), 2) if image_files else 0
    
    # Save results to JSON file
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {output_file}")
    except Exception as e:
        print(f"❌ Failed to save results: {e}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 PROCESSING SUMMARY")
    print("=" * 60)
    print(f"✅ Successful: {successful}/{len(image_files)}")
    print(f"❌ Failed: {failed}/{len(image_files)}")
    print(f"⏱️  Total time: {processing_time:.2f} seconds")
    print(f"📈 Average per image: {results['metadata']['average_time_per_image']} seconds")
    print(f"💾 Output file: {output_file}")
    
    return results

def load_embeddings_from_file(json_file):
    """
    Load previously saved embeddings from JSON file.
    
    Parameters:
    json_file (str): Path to JSON file with embeddings
    
    Returns:
    dict: Loaded embeddings data
    """
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        print(f"✅ Loaded embeddings from: {json_file}")
        print(f"   Contains {len(data['embeddings'])} embeddings")
        return data
    except Exception as e:
        print(f"❌ Failed to load embeddings: {e}")
        return None

def find_similar_faces(embeddings_data, query_image_path, top_k=5):
    """
    Find the most similar faces in the dataset to a query image.
    
    Parameters:
    embeddings_data (dict): The embeddings data from process_folder_embeddings
    query_image_path (str): Path to the query image
    top_k (int): Number of top matches to return
    
    Returns:
    list: List of similar faces with similarity scores
    """
    print(f"\n🔍 FINDING SIMILAR FACES TO: {os.path.basename(query_image_path)}")
    
    # Create embedding for query image
    query_embedding_obj = create_embedding_for_image(query_image_path)
    if not query_embedding_obj:
        return None
    
    query_embedding = np.array(query_embedding_obj['embedding'])
    
    similarities = []
    
    # Compare with all stored embeddings
    for filename, data in embeddings_data['embeddings'].items():
        stored_embedding = np.array(data['embedding'])
        
        # Calculate cosine similarity
        cosine_sim = np.dot(query_embedding, stored_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding)
        )
        
        similarities.append({
            'filename': filename,
            'similarity': cosine_sim,
            'file_path': data['file_path']
        })
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    
    # Return top_k matches (excluding the query itself if it's in the dataset)
    return similarities[:top_k]

# Example usage and demonstration
if __name__ == "__main__":
    # Configuration
    FOLDER_PATH = input("Enter folder target path:").strip()  # Replace with your folder path
    MODEL_NAME = "Facenet"  # Options: "VGG-Face", "Facenet", "OpenFace", "ArcFace"
    OUTPUT_FILE = "face_embeddings_dataset.json"
    
    # 🎯 MAIN PIPELINE: Process all images in folder
    embeddings_data = process_folder_embeddings(
        folder_path=FOLDER_PATH,
        model_name=MODEL_NAME,
        output_file=OUTPUT_FILE,
        enforce_detection=False  # Set to False for pre-cropped faces
    )
    
    if embeddings_data:
        # 🔍 DEMO: Find similar faces (optional)
        print("\n" + "=" * 60)
        print("🎯 SIMILARITY SEARCH DEMO")
        print("=" * 60)
        
        # You can test with one of your images
        test_image = None
        if embeddings_data['embeddings']:
            # Use the first image as a test query
            first_image = list(embeddings_data['embeddings'].keys())[0]
            first_image_path = embeddings_data['embeddings'][first_image]['file_path']
            
            similar_faces = find_similar_faces(embeddings_data, first_image_path, top_k=3)
            
            if similar_faces:
                print(f"\n📊 Most similar faces to '{first_image}':")
                for i, match in enumerate(similar_faces, 1):
                    print(f"   {i}. {match['filename']} - Similarity: {match['similarity']:.4f}")
    
    # 📁 UTILITY: Load previously created embeddings
    print("\n" + "=" * 60)
    print("🔄 LOADING EMBEDDINGS DEMO")
    print("=" * 60)
    
    if os.path.exists(OUTPUT_FILE):
        loaded_data = load_embeddings_from_file(OUTPUT_FILE)
        
        if loaded_data:
            print("📈 Dataset Statistics:")
            print(f"   • Total images processed: {loaded_data['metadata']['total_images']}")
            print(f"   • Successful embeddings: {loaded_data['metadata']['successful_embeddings']}")
            print(f"   • Model used: {loaded_data['metadata']['model_name']}")
            print(f"   • Processing time: {loaded_data['metadata']['processing_time_seconds']} seconds")