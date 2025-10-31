# main_pipeline_app.py
import cv2
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.pipeline_orchestrator import PipelineOrchestrator
from components.detection.face_detector import FaceDetector
from components.recognition.robust_face_recognizer import RobustFaceRecognizer
from components.detection.mask_detector import MaskDetector

def create_pipeline(config):
    """Create and configure the complete processing pipeline"""
    orchestrator = PipelineOrchestrator(config)
    
    # Register processors in correct order
    orchestrator.register_processor(FaceDetector("FaceDetector", config))
    orchestrator.register_processor(RobustFaceRecognizer("FaceRecognizer", config))
    orchestrator.register_processor(MaskDetector("MaskDetector", config))
    
    return orchestrator

def main():
    """Main application using the modular pipeline"""
    config = {
        'detection_model_path': r'D:\SCMA\3-APD\fromAraya\Computer-Vision-CV\3.1_FaceRecog\yolov11n-face.pt',
        'mask_model_path': r'D:\SCMA\3-APD\fromAraya\Computer-Vision-CV\3.1_FaceRecog\run_py\mask_detector112.onnx',
        'embeddings_db_path': r'D:\SCMA\3-APD\fromAraya\Computer-Vision-CV\3.1_FaceRecog\person_folder_3.json',
        'detection_confidence': 0.6,
        'detection_iou': 0.6,
        'mask_detection_threshold': 0.8,
        'recognition_threshold': 0.6,
        'embedding_model': 'Facenet',
        'roi_padding': 20,
        'mask_roi_padding': 10,
        'enable_robust_recognition': True
    }
    
    # Create pipeline
    pipeline = create_pipeline(config)
    
    # Initialize pipeline
    if not pipeline.initialize_pipeline():
        print("❌ Failed to initialize pipeline")
        return
    
    # Start pipeline
    pipeline.start_pipeline()
    
    # Test with webcam
    cap = cv2.VideoCapture(0)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame through pipeline
            results = pipeline.process_frame(frame)
            
            # Display results
            display_frame = frame.copy()
            
            # Draw results if available
            if 'mask_detections' in results:
                for detection in results['mask_detections']:
                    x1, y1, x2, y2 = detection['bbox']
                    identity = detection['identity']
                    mask_status = detection['mask_status']
                    
                    # Simple drawing logic
                    color = (0, 255, 0) if mask_status == "mask" else (0, 0, 255)
                    label = f"{identity if identity else 'Unknown'}: {mask_status}"
                    
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(display_frame, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            cv2.imshow('Face Recognition Pipeline', display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pipeline.stop_pipeline()

if __name__ == "__main__":
    main()