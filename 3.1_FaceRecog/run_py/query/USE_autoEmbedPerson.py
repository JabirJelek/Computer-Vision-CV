import json
import os
import numpy as np
from deepface import DeepFace
from typing import Dict, List
import time

class SimpleFaceDatasetCreator:
    def __init__(self, dataset_root: str, output_json: str = "simple_face_dataset.json"):
        self.dataset_root = dataset_root
        self.output_json = output_json
        self.dataset = {"persons": {}, "metadata": {}}
    
    def process_person_folder(self, person_folder: str, person_id: str) -> Dict:
        """
        Process all images in a person's folder - SIMPLE VERSION
        """
        person_path = os.path.join(self.dataset_root, person_folder)
        
        if not os.path.isdir(person_path):
            return None
            
        print(f"👤 Processing: {person_folder} -> {person_id}")
        
        person_data = {
            "person_id": person_id,
            "folder_name": person_folder,
            "display_name": person_folder.replace('_', ' ').title(),
            "embeddings": [],
            "total_images": 0,
            "successful_embeddings": 0
        }
        
        # Process each image - NO COMPLEX NAMING NEEDED!
        image_files = [f for f in os.listdir(person_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for image_file in image_files:
            image_path = os.path.join(person_path, image_file)
            
            try:
                print(f"   📷 Processing: {image_file}")
                
                # Generate embedding
                embedding_obj = DeepFace.represent(
                    img_path=image_path,
                    model_name='Facenet',
                    enforce_detection=False
                )
                
                if embedding_obj:
                    embedding_vector = embedding_obj[0]['embedding']
                    
                    # SIMPLE embedding data - no angle detection!
                    embedding_data = {
                        "vector": embedding_vector,
                        "source_file": image_file,
                        "file_path": image_path,
                        "embedding_length": len(embedding_vector)
                    }
                    
                    person_data["embeddings"].append(embedding_data)
                    person_data["successful_embeddings"] += 1
                    print(f"   ✅ Added embedding")
                
            except Exception as e:
                print(f"   ❌ Failed: {image_file} - {e}")
                continue
        
        person_data["total_images"] = len(image_files)
        return person_data if person_data["embeddings"] else None
    
    def calculate_person_centroid(self, embeddings: List[Dict]) -> List[float]:
        """Calculate average embedding for a person"""
        if not embeddings:
            return []
        
        vectors = [np.array(emb["vector"]) for emb in embeddings]
        centroid = np.mean(vectors, axis=0)
        return centroid.tolist()
    
    def create_dataset(self):
        """Main method - SIMPLIFIED"""
        print("🚀 Starting SIMPLE Face Dataset Creation")
        print("=" * 50)
        
        if not os.path.exists(self.dataset_root):
            print(f"❌ Dataset root not found: {self.dataset_root}")
            return False
        
        # Get all person folders
        person_folders = [f for f in os.listdir(self.dataset_root) 
                         if os.path.isdir(os.path.join(self.dataset_root, f))]
        
        if not person_folders:
            print("❌ No person folders found!")
            return False
        
        print(f"📁 Found {len(person_folders)} person folders")
        
        total_embeddings = 0
        successful_persons = 0
        
        # Process each person folder
        for person_folder in person_folders:
            person_id = f"person_{successful_persons + 1:03d}"
            
            person_data = self.process_person_folder(person_folder, person_id)
            
            if person_data:
                # Calculate centroid (average embedding)
                centroid = self.calculate_person_centroid(person_data["embeddings"])
                person_data["centroid_embedding"] = centroid
                
                # Store in dataset
                self.dataset["persons"][person_id] = person_data
                total_embeddings += person_data["successful_embeddings"]
                successful_persons += 1
                
                print(f"✅ Completed: {person_id} - {person_data['successful_embeddings']} embeddings\n")
        
        # Add metadata
        self.dataset["metadata"] = {
            "creation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_persons": successful_persons,
            "total_embeddings": total_embeddings,
            "average_embeddings_per_person": round(total_embeddings / successful_persons, 2) if successful_persons > 0 else 0,
            "description": "Simple dataset - no angle information"
        }
        
        # Save to JSON
        with open(self.output_json, 'w') as f:
            json.dump(self.dataset, f, indent=2)
        
        print("🎉 SIMPLE Dataset Creation Complete!")
        print(f"📊 Summary: {successful_persons} persons, {total_embeddings} embeddings")
        print(f"💾 Saved to: {self.output_json}")
        
        return True

# 🎯 USAGE
def main():
    # CONFIGURATION - JUST POINT TO YOUR FOLDER!
    DATASET_ROOT = input("Please provide Dataset path: ").strip()  # Your folder with person folders
    OUTPUT_JSON = "simple_face_dataset.json"
    
    # Create the simple dataset
    creator = SimpleFaceDatasetCreator(DATASET_ROOT, OUTPUT_JSON)
    success = creator.create_dataset()
    
    if success:
        print(f"\n✅ Success! Your enhanced JSON is ready at: {OUTPUT_JSON}")
        
        # Show sample
        with open(OUTPUT_JSON, 'r') as f:
            data = json.load(f)
        
        print(f"\n📋 Sample person data:")
        first_person_id = list(data["persons"].keys())[0]
        first_person = data["persons"][first_person_id]
        
        print(f"   ID: {first_person_id}")
        print(f"   Folder: {first_person['folder_name']}")
        print(f"   Display Name: {first_person['display_name']}")
        print(f"   Embeddings: {first_person['successful_embeddings']}")
        print(f"   Files: {[emb['source_file'] for emb in first_person['embeddings'][:3]]}")

if __name__ == "__main__":
    main()