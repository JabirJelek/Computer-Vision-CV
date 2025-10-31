# test_mask_detector_fix.py
import cv2
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.pipeline_orchestrator import PipelineOrchestrator
from components.detection.face_detector import FaceDetector
from components.recognition.robust_face_recognizer import RobustFaceRecognizer
from components.detection.mask_detector import MaskDetector

def test_mask_detector_fix():
    """Test the fixed mask detector"""
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
        # Test just the mask detector first
        print("🧪 Testing MaskDetector in isolation...")
        mask_detector = MaskDetector("TestMaskDetector", config)
        
        if mask_detector.initialize():
            print("✅ MaskDetector initialized successfully!")
            
            # Test with a sample image
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                # Create a dummy detection for testing
                h, w = frame.shape[:2]
                test_bbox = [w//4, h//4, 3*w//4, 3*h//4]
                
                # Test mask detection
                mask_status, confidence = mask_detector._detect_mask(frame)
                print(f"🎭 Test result: {mask_status} (confidence: {confidence:.3f})")
                
                # Test with pipeline
                print("\n🧪 Testing complete pipeline...")
                test_complete_pipeline()
            else:
                print("❌ Could not capture test frame")
        else:
            print("❌ MaskDetector initialization failed")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

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
        'mask_roi_padding': 10
    }
    
    orchestrator = PipelineOrchestrator(config)
    orchestrator.register_processor(FaceDetector("FaceDetector", config))
    orchestrator.register_processor(RobustFaceRecognizer("FaceRecognizer", config))
    orchestrator.register_processor(MaskDetector("MaskDetector", config))
    
    if orchestrator.initialize_pipeline():
        print("✅ Complete pipeline initialized!")
        
        # Quick test with webcam
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        
        if ret:
            results = orchestrator.process_frame(frame)
            print(f"📊 Pipeline results:")
            print(f"   - Detections: {len(results.get('detections', []))}")
            print(f"   - Recognitions: {len(results.get('recognitions', []))}")
            print(f"   - Mask detections: {len(results.get('mask_detections', []))}")
            
            if 'mask_metadata' in results:
                meta = results['mask_metadata']
                print(f"   - Masks: {meta['mask_count']}/{meta['total_processed']}")
        
        cap.release()
        orchestrator.stop_pipeline()

if __name__ == "__main__":
    test_mask_detector_fix()