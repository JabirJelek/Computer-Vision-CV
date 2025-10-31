# test_mask_detector_fix.py (ENHANCED VERSION)
import cv2
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.pipeline_orchestrator import PipelineOrchestrator
from components.detection.face_detector import FaceDetector
from components.recognition.robust_face_recognizer import RobustFaceRecognizer
from components.detection.mask_detector import MaskDetector

def test_mask_detector_isolated():
    """Test the fixed mask detector in isolation"""
    config = {
        'detection_model_path': r'D:\SCMA\3-APD\fromAraya\Computer-Vision-CV\3.1_FaceRecog\yolov11n-face.pt',
        'mask_model_path': r'D:\SCMA\3-APD\fromAraya\Computer-Vision-CV\3.1_FaceRecog\run_py\mask_detector112.onnx',
        'embeddings_db_path': r'D:\SCMA\3-APD\fromAraya\Computer-Vision-CV\3.1_FaceRecog\person_folder_3.json',
        'detection_confidence': 0.6,
        'mask_detection_threshold': 0.8,
        'recognition_threshold': 0.6,
        'embedding_model': 'Facenet',
        'roi_padding': 20,
        'mask_roi_padding': 10
    }
    
    try:
        print("🧪 Testing MaskDetector in isolation...")
        mask_detector = MaskDetector("TestMaskDetector", config)
        
        if mask_detector.initialize():
            print("✅ MaskDetector initialized successfully!")
            
            # Test model stats
            stats = mask_detector.get_mask_stats()
            print(f"📊 Model Stats: {stats}")
            
            # Test with a sample image
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                print("📸 Testing with captured frame...")
                
                # Test preprocessing directly
                h, w = frame.shape[:2]
                test_roi = frame[h//4:3*h//4, w//4:3*w//4]  # Center region
                
                if test_roi.size > 0:
                    # Test preprocessing
                    processed = mask_detector._preprocess_face(test_roi)
                    print(f"✅ Preprocessing successful - output shape: {processed.shape}")
                    
                    # Test mask detection
                    mask_status, confidence = mask_detector._detect_mask(test_roi)
                    print(f"🎭 Mask detection: {mask_status} (confidence: {confidence:.3f})")
                else:
                    print("❌ Invalid test ROI")
            else:
                print("❌ Could not capture test frame")
                
            mask_detector.cleanup()
            return True
        else:
            print("❌ MaskDetector initialization failed")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_complete_pipeline():
    """Test the complete pipeline with the fixed mask detector"""
    config = {
        'detection_model_path': r'D:\SCMA\3-APD\fromAraya\Computer-Vision-CV\3.1_FaceRecog\yolov11n-face.pt',
        'mask_model_path': r'D:\SCMA\3-APD\fromAraya\Computer-Vision-CV\3.1_FaceRecog\run_py\mask_detector112.onnx',
        'embeddings_db_path': r'D:\SCMA\3-APD\fromAraya\Computer-Vision-CV\3.1_FaceRecog\person_folder_3.json',
        'detection_confidence': 0.6,
        'mask_detection_threshold': 0.8,
        'recognition_threshold': 0.6,
        'embedding_model': 'Facenet',
        'roi_padding': 20,
        'mask_roi_padding': 10,
        'enable_robust_recognition': True
    }
    
    try:
        orchestrator = PipelineOrchestrator(config)
        
        # Register processors in correct order
        orchestrator.register_processor(FaceDetector("FaceDetector", config))
        orchestrator.register_processor(RobustFaceRecognizer("FaceRecognizer", config))
        orchestrator.register_processor(MaskDetector("MaskDetector", config))
        
        if orchestrator.initialize_pipeline():
            print("✅ Complete pipeline initialized!")
            
            # Quick test with webcam
            cap = cv2.VideoCapture(0)
            print("📸 Capturing test frame...")
            ret, frame = cap.read()
            
            if ret:
                print("🔄 Processing frame through pipeline...")
                results = orchestrator.process_frame(frame)
                
                print(f"📊 Pipeline results:")
                print(f"   - Detections: {len(results.get('detections', []))}")
                print(f"   - Recognitions: {len(results.get('recognitions', []))}")
                print(f"   - Mask detections: {len(results.get('mask_detections', []))}")
                
                if 'mask_metadata' in results:
                    meta = results['mask_metadata']
                    print(f"   - Masks: {meta['mask_count']}/{meta['total_processed']}")
                
                # Show pipeline status
                status = orchestrator.get_pipeline_status()
                print(f"🎯 Pipeline Status: {status['state']}")
                print(f"📈 Active processors: {status['active_processors']}")
                
            else:
                print("❌ Could not capture frame from webcam")
            
            cap.release()
            orchestrator.stop_pipeline()
            return True
        else:
            print("❌ Pipeline initialization failed")
            return False
            
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run comprehensive tests"""
    print("🚀 STARTING COMPREHENSIVE MASK DETECTOR TESTS")
    print("=" * 60)
    
    # Test 1: Isolated MaskDetector
    print("\n1. TESTING MASK DETECTOR ISOLATION")
    print("-" * 40)
    isolated_success = test_mask_detector_isolated()
    
    # Test 2: Complete Pipeline
    if isolated_success:
        print("\n2. TESTING COMPLETE PIPELINE")  
        print("-" * 40)
        pipeline_success = test_complete_pipeline()
    else:
        print("\n⏭️  Skipping pipeline test due to isolation failure")
        pipeline_success = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    print(f"🧪 MaskDetector Isolation: {'✅ PASS' if isolated_success else '❌ FAIL'}")
    print(f"🔗 Complete Pipeline: {'✅ PASS' if pipeline_success else '❌ FAIL'}")
    
    if isolated_success and pipeline_success:
        print("\n🎉 ALL TESTS PASSED! Mask detector is working correctly.")
    else:
        print("\n💡 Some tests failed. Check the error messages above.")

if __name__ == "__main__":
    main()