# test_face_recognizer.py
import cv2
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.pipeline_orchestrator import PipelineOrchestrator

def test_face_recognizer():
    """Test the complete pipeline with FaceDetector and FaceRecognizer"""
    config = {
        'detection_model_path': r'D:\SCMA\3-APD\fromAraya\Computer-Vision-CV\3.1_FaceRecog\yolov11n-face.pt',
        'embeddings_db_path': r'D:\SCMA\3-APD\fromAraya\Computer-Vision-CV\3.1_FaceRecog\person_folder_3.json',
        'detection_confidence': 0.6,
        'detection_iou': 0.6,
        'recognition_threshold': 0.6,
        'embedding_model': 'Facenet',
        'roi_padding': 20,
        'enable_robust_recognition': True,
        'enable_multi_scale': True,
        'enable_temporal_fusion': True,
        'temporal_buffer_size': 10
    }
    
    try:
        # Create orchestrator
        orchestrator = PipelineOrchestrator(config)
        
        # Register FaceDetector
        from components.detection.face_detector import FaceDetector
        face_detector = FaceDetector("FaceDetector", config)
        orchestrator.register_processor(face_detector)    
        
        # Register FaceRecognizer - FIXED IMPORT PATH
        use_robust = config.get('enable_robust_recognition', True)
        if use_robust:
            from components.recognition.robust_face_recognizer import RobustFaceRecognizer
            robust_face_recognizer = RobustFaceRecognizer("FaceRecognizer", config)
        else:
            from components.recognition.face_recognizer import FaceRecognizer
            face_recognizer = FaceRecognizer("FaceRecognizer", config)
        orchestrator.register_processor(robust_face_recognizer)            
        
        # Initialize complete pipeline (detection + recognition)
        if not orchestrator.initialize_pipeline():
            print("❌ Pipeline initialization failed")
            return
        
        print("✅ Pipeline initialized with FaceDetector and FaceRecognizer")
        
        # Test with webcam
        cap = cv2.VideoCapture(0)
        
        try:
            orchestrator.start_pipeline()
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame through complete pipeline
                results = orchestrator.process_frame(frame)
                
                # Display results
                display_frame = frame.copy()
                
                # Draw detections and recognitions
                if 'recognitions' in results:
                    for recognition in results['recognitions']:
                        x1, y1, x2, y2 = recognition['bbox']
                        identity = recognition['identity']
                        confidence = recognition['recognition_confidence']
                        
                        # Color coding based on recognition
                        if identity:
                            color = (0, 255, 0)  # Green for recognized
                            label = f'{identity} ({confidence:.2f})'
                        else:
                            color = (0, 0, 255)  # Red for unknown
                            label = f'Unknown ({confidence:.2f})'
                        
                        # Draw bounding box
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw label
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(display_frame, (x1, y1 - label_size[1] - 10), 
                                    (x1 + label_size[0], y1), color, -1)
                        cv2.putText(display_frame, label, (x1, y1 - 5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Show pipeline status
                status = orchestrator.get_pipeline_status()
                cv2.putText(display_frame, f"Processors: {status['active_processors']}/{status['processor_count']}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(display_frame, f"Frames: {status['frame_count']}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Show recognition stats
                if 'recognition_metadata' in results:
                    meta = results['recognition_metadata']
                    cv2.putText(display_frame, f"Recognized: {meta['recognized_count']}/{meta['total_processed']}", 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imshow('Face Recognition Pipeline Test', display_frame)
                
                # Print detailed status every 30 frames
                if frame_count % 30 == 0:
                    print(f"📊 Pipeline Status:")
                    print(f"   - Active processors: {status['active_processors']}")
                    print(f"   - Total frames: {status['frame_count']}")
                    print(f"   - Avg processing time: {status['avg_processing_time']:.3f}s")
                    
                    # Get recognizer stats if available
                    recognizer = orchestrator.get_processor('FaceRecognizer')
                    if recognizer:
                        if hasattr(recognizer, 'get_robust_stats'):
                            stats = recognizer.get_robust_stats()
                            print(f"   - Known identities: {stats['known_identities']}")
                            print(f"   - Temporal tracks: {stats.get('temporal_tracks', 0)}")
                        else:
                            stats = recognizer.get_recognition_stats()
                            print(f"   - Known identities: {stats['known_identities']}")
                
                frame_count += 1
                
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
    test_face_recognizer()