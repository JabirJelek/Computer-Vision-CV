from dataset_quality_checker import DatasetQualityChecker
import os

def quick_quality_assessment(dataset_json: str):
    """Quick assessment without visualizations"""
    if not os.path.exists(dataset_json):
        print(f"❌ Dataset not found: {dataset_json}")
        return
    
    checker = DatasetQualityChecker(dataset_json)
    report = checker.generate_quality_report()
    
    quality_score = report['summary']['quality_score']
    
    print(f"\n🎯 QUICK ASSESSMENT:")
    if quality_score >= 80:
        print("   ✅ EXCELLENT - Your dataset is well-balanced and sufficient")
    elif quality_score >= 60:
        print("   ✅ GOOD - Your dataset is adequate for research")
    else:
        print("   ⚠️  NEEDS IMPROVEMENT - Consider balancing your dataset")
    
    print(f"   Quality Score: {quality_score}/100")

if __name__ == "__main__":
    quick_quality_assessment("simple_face_dataset.json")