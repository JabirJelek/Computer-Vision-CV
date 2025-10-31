import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.pipeline_orchestrator import PipelineOrchestrator

# test_enhanced_orchestrator.py
def test_enhanced_orchestrator():
    """Test the enhanced PipelineOrchestrator with FaceDetector"""
    config = {
        'detection_model_path': r'D:\SCMA\3-APD\fromAraya\Computer-Vision-CV\3.1_FaceRecog\yolov11n-face.pt',
        'embeddings_db_path': r'D:\SCMA\3-APD\fromAraya\Computer-Vision-CV\3.1_FaceRecog\person_folder_3.json',
        'detection_confidence': 0.6,
        'detection_iou': 0.6,
        'recognition_threshold': 0.7
    }
    
    try:
        # Create orchestrator
        orchestrator = PipelineOrchestrator(config)
        
        # Register FaceDetector
        from components.detection.face_detector import FaceDetector
        face_detector = FaceDetector("FaceDetector", config)
        orchestrator.register_processor(face_detector)
        
        # Initialize pipeline
        if not orchestrator.initialize_pipeline():
            print("❌ Pipeline initialization failed")
            return
        
        print("✅ Pipeline initialized successfully")
        
        # Test with webcam
        import cv2
        cap = cv2.VideoCapture(0)
        
        try:
            orchestrator.start_pipeline()
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame through enhanced pipeline
                results = orchestrator.process_frame(frame)
                
                # Display results
                if 'detections' in results:
                    for detection in results['detections']:
                        x1, y1, x2, y2 = detection['bbox']
                        confidence = detection['confidence']
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f'Face: {confidence:.2f}', (x1, y1-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Show pipeline status on frame
                status = orchestrator.get_pipeline_status()
                cv2.putText(frame, f"Processors: {status['active_processors']}/{status['processor_count']}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"Frames: {status['frame_count']}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imshow('Enhanced Pipeline Test', frame)
                
                # Print status every 30 frames
                if status['frame_count'] % 30 == 0:
                    print(f"📊 Pipeline Status: {status['active_processors']} processors, "
                          f"{status['frame_count']} frames, "
                          f"Avg time: {status['avg_processing_time']:.3f}s")
                
                # Exit on 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()
            orchestrator.stop_pipeline()
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_enhanced_orchestrator()