import numpy as np
import os
from face_query_system import FaceQuerySystem
# run_query.py
def main():
    try:
        # Initialize with your embeddings
        json_path = input("Enter the json embedding path: ").strip()
        print("🚀 Initializing Face Query System...")
        query_system = FaceQuerySystem(json_path)
        
        # Inspect the loaded data
        query_system.inspect_embeddings()
        
        # Get statistics
        stats = query_system.get_statistics()
        print(f"\n📊 Database Statistics:")
        print(f"   Total faces: {stats['total_faces']}")
        print(f"   Embedding dimension: {stats['embedding_dimension']}")
        print(f"   Sample files: {stats['sample_files']}")
        
        # Test query with existing embedding
        if stats['total_faces'] > 0:
            print(f"\n🎯 Testing query with first embedding in database...")
            sample_embedding = np.array(query_system.embeddings_data[0]['embedding'])
            matches = query_system.query_by_embedding(
                query_embedding=sample_embedding,
                top_k=5,
                similarity_threshold=0.5
            )
            
            print("\nTop Matches:")
            for i, match in enumerate(matches):
                print(f"  #{match['match_rank']} - Similarity: {match['similarity_score']:.3f}")
                print(f"     File: {match['filename']}")
                if i == 0:
                    print(f"     ✅ This should be an exact match (same image)")
        
        # Test with image file if available
        test_image = input("Enter test image: ").strip()  # Change to your test image path
        if os.path.exists(test_image):
            print(f"\n🔍 Testing query with image: {test_image}")
            matches = query_system.query_by_image(test_image, top_k=3)
            
            print("\nImage Query Results:")
            for i, match in enumerate(matches):
                print(f"  {i+1}. {match['filename']} - Score: {match['similarity_score']:.3f}")
        else:
            print(f"\n⚠️  Test image not found at: {test_image}")
            print("   Using sample embedding query instead.")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()