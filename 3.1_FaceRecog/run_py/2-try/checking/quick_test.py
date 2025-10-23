from test_face_recognition import FaceRecognitionTester
import os

def quick_test():
    """Run a quick test without the menu"""
    dataset_json = input("Enter dataset json path: ").strip()  # Change this path
    
    if not os.path.exists(dataset_json):
        print(f"❌ Dataset not found: {dataset_json}")
        return
    
    tester = FaceRecognitionTester(dataset_json)
    
    # Quick stats
    tester.print_quick_stats()
    
    # Test self-recognition for all persons
    print("\n🚀 Running Quick Self-Recognition Test...")
    for person_id in list(tester.dataset["persons"].keys())[:5]:  # Test first 5
        result = tester.test_known_person(person_id)
        status = "✅" if result.get("actual_rank") == 1 else "❌"
        print(f"   {status} {person_id}: Rank {result.get('actual_rank')}")

if __name__ == "__main__":
    quick_test()