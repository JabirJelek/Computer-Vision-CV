# Import required libraries
from deepface import DeepFace
import numpy as np
import json
import os

def create_face_embedding(image_path, model_name="Facenet", detector_backend="opencv", align=True):
    """
    Generates face embedding for a given image with detailed debugging.
    
    Parameters:
    image_path (str): Path to the input image file.
    model_name (str): Face recognition model to use.
    detector_backend (str): Face detector backend.
    align (bool): Whether to align the face.
    
    Returns:
    list: A list of dictionaries containing embedding and face location data.
    """
    
    # Debug: Print configuration
    print("🔧 CONFIGURATION SETTINGS:")
    print(f"   Image path: {image_path}")
    print(f"   Model: {model_name}")
    print(f"   Detector: {detector_backend}")
    print(f"   Align: {align}")
    print("-" * 60)
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ ERROR: Image file not found: {image_path}")
        return None
    
    try:
        # Generate embedding using DeepFace.represent()
        print("🔄 Processing image...")
        embedding_objs = DeepFace.represent(
            img_path=image_path,
            model_name=model_name,
            detector_backend=detector_backend,
            align=align,
            enforce_detection=False
        )
        
        print("✅ Processing completed successfully!")
        return embedding_objs
        
    except ValueError as ve:
        print(f"❌ Face detection error: {ve}")
        return None
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        return None

def debug_embedding_results(results, show_full_embedding=False):
    """
    Display detailed debugging information about the embedding results.
    
    Parameters:
    results (list): The results from DeepFace.represent()
    show_full_embedding (bool): Whether to display the entire embedding vector
    """
    
    if not results:
        print("❌ No results to debug.")
        return
    
    print("\n" + "=" * 80)
    print("🔍 DEBUG EMBEDDING RESULTS")
    print("=" * 80)
    
    print(f"📊 Total faces detected: {len(results)}")
    
    for i, face in enumerate(results):
        print(f"\n{'=' * 50}")
        print(f"👤 FACE {i+1} DETAILS")
        print(f"{'=' * 50}")
        
        # Face region information
        facial_area = face['facial_area']
        print(f"📍 Face Region: (x:{facial_area['x']}, y:{facial_area['y']}, "
              f"w:{facial_area['w']}, h:{facial_area['h']})")
        
        # Confidence score (if available)
        confidence = face.get('face_confidence', 'Not provided')
        print(f"🎯 Detection Confidence: {confidence}")
        
        # Embedding vector analysis
        embedding = face['embedding']
        print(f"📐 Embedding Vector Length: {len(embedding)} dimensions")
        
        # Convert to numpy array for statistical analysis
        embedding_array = np.array(embedding)
        
        print(f"📈 Embedding Statistics:")
        print(f"   • Min value: {embedding_array.min():.6f}")
        print(f"   • Max value: {embedding_array.max():.6f}")
        print(f"   • Mean value: {embedding_array.mean():.6f}")
        print(f"   • Std deviation: {embedding_array.std():.6f}")
        print(f"   • L2 Norm: {np.linalg.norm(embedding_array):.6f}")
        
        # Show sample of embedding values
        print(f"\n📋 Embedding Vector Sample (first 10 values):")
        sample_values = embedding[:10]
        for j, value in enumerate(sample_values):
            print(f"   [{j:2d}] {value:.6f}")
        
        # Show full embedding if requested (use with caution!)
        if show_full_embedding:
            print(f"\n📜 FULL EMBEDDING VECTOR:")
            for j in range(0, len(embedding), 5):  # Print 5 values per line
                line_values = embedding[j:j+5]
                indices = [f"{j+k:3d}" for k in range(len(line_values))]
                values = [f"{val:.6f}" for val in line_values]
                print(f"   [{','.join(indices)}]: {', '.join(values)}")
        
        # Additional analysis
        print(f"\n🔬 Additional Analysis:")
        # Check for zero or near-zero values
        zero_count = np.sum(np.abs(embedding_array) < 1e-6)
        print(f"   • Near-zero values (< 1e-6): {zero_count}/{len(embedding)}")
        
        # Check value distribution
        positive_count = np.sum(embedding_array > 0)
        negative_count = np.sum(embedding_array < 0)
        print(f"   • Positive values: {positive_count}/{len(embedding)}")
        print(f"   • Negative values: {negative_count}/{len(embedding)}")
        
        # Normalized embedding (for comparison purposes)
        normalized_embedding = embedding_array / np.linalg.norm(embedding_array)
        print(f"   • Normalized vector norm: {np.linalg.norm(normalized_embedding):.6f}")

def save_embedding_to_file(embedding_data, filename="embedding_debug.json"):
    """
    Save embedding results to a JSON file for later analysis.
    
    Parameters:
    embedding_data (list): The embedding results
    filename (str): Output filename
    """
    try:
        # Convert numpy arrays to lists for JSON serialization
        serializable_data = []
        for face in embedding_data:
            serializable_face = face.copy()
            serializable_face['embedding'] = [float(x) for x in face['embedding']]
            serializable_data.append(serializable_face)
        
        with open(filename, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        print(f"\n💾 Embedding data saved to: {filename}")
    except Exception as e:
        print(f"❌ Failed to save embedding data: {e}")

# Example usage with comprehensive debugging
if __name__ == "__main__":
    # Specify the path to your image
    image_path = input("Enter the image path:").strip()  # Replace with your actual image path
    
    print("🚀 STARTING FACE EMBEDDING DEBUG SCRIPT")
    print("=" * 60)
    
    # Generate the embedding
    results = create_face_embedding(
        image_path=image_path,
        model_name="Facenet",  # Try also: "Facenet", "OpenFace", "ArcFace"
        detector_backend="opencv"  # Try also: "mtcnn", "retinaface" for better accuracy
    )
    
    # Debug the results
    if results:
        debug_embedding_results(
            results, 
            show_full_embedding=False  # Set to True to see complete vector (can be very long!)
        )
        
        # Save results to file
        save_embedding_to_file(results, "face_embedding_debug.json")
        
        # Additional: Compare embeddings if multiple faces
        if len(results) > 1:
            print(f"\n{'=' * 60}")
            print("🔍 MULTI-FACE COMPARISON")
            print(f"{'=' * 60}")
            
            for i in range(len(results)):
                for j in range(i + 1, len(results)):
                    emb1 = np.array(results[i]['embedding'])
                    emb2 = np.array(results[j]['embedding'])
                    
                    # Calculate similarity metrics
                    cosine_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                    euclidean_dist = np.linalg.norm(emb1 - emb2)
                    
                    print(f"📊 Face {i+1} vs Face {j+1}:")
                    print(f"   • Cosine Similarity: {cosine_sim:.4f}")
                    print(f"   • Euclidean Distance: {euclidean_dist:.4f}")
                    
                    # Interpretation
                    if cosine_sim > 0.6:
                        similarity_interpretation = "High similarity - possibly same person"
                    elif cosine_sim > 0.3:
                        similarity_interpretation = "Moderate similarity"
                    else:
                        similarity_interpretation = "Low similarity - likely different persons"
                    
                    print(f"   • Interpretation: {similarity_interpretation}")
    else:
        print("❌ No faces detected or processing failed.")
    
    print("\n" + "=" * 60)
    print("✅ DEBUG SCRIPT COMPLETED")