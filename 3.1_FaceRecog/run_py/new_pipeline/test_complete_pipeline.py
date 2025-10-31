# test_complete_pipeline.py
import cv2
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.pipeline_orchestrator import PipelineOrchestrator

def test_complete_pipeline():
    """Test the complete pipeline with all components"""
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
        'enable_robust_recognition': True,
        'enable_multi_scale': True,
        'enable_temporal_fusion': True,
        'temporal_buffer_size': 10
    }
    
    try:
        # Create orchestrator
        orchestrator = PipelineOrchestrator(config)
        
        # ✅ CORRECT: Register processors in the correct processing order
        from components.detection.face_detector import FaceDetector
        from components.recognition.robust_face_recognizer import RobustFaceRecognizer
        from components.detection.mask_detector import MaskDetector
        
        # Register in processing pipeline order: Detection → Recognition → Mask Detection
        face_detector = FaceDetector("FaceDetector", config)
        robust_face_recognizer = RobustFaceRecognizer("FaceRecognizer", config) 
        mask_detector = MaskDetector("MaskDetector", config)
        
        # Register in the correct order
        orchestrator.register_processor(face_detector)
        orchestrator.register_processor(robust_face_recognizer)
        orchestrator.register_processor(mask_detector)
        
        print("🔄 Processor registration order:")
        for i, processor in enumerate(orchestrator.processors):
            print(f"   {i+1}. {processor.name}")
        
        # Initialize complete pipeline
        if not orchestrator.initialize_pipeline():
            print("❌ Pipeline initialization failed")
            return
        
        print("✅ Complete pipeline initialized with proper component order")
        
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
                
                # Draw mask detections and recognitions
                if 'mask_detections' in results:
                    for detection in results['mask_detections']:
                        x1, y1, x2, y2 = detection['bbox']
                        identity = detection['identity']
                        mask_status = detection['mask_status']
                        mask_confidence = detection['mask_confidence']
                        
                        # Color coding based on mask status and recognition
                        if identity:
                            if mask_status == "mask":
                                color = (0, 255, 0)  # Green for recognized with mask
                            else:
                                color = (0, 255, 255)  # Yellow for recognized without mask
                        else:
                            if mask_status == "mask":
                                color = (255, 255, 0)  # Cyan for unknown with mask
                            else:
                                color = (0, 0, 255)    # Red for unknown without mask
                        
                        # Draw bounding box
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Prepare label
                        if identity:
                            base_label = f"{identity} (Rec:{detection['recognition_confidence']:.2f})"
                        else:
                            base_label = f"Unknown (Det:{detection['detection_confidence']:.2f})"
                        
                        # Add mask status to label
                        mask_label = f" | Mask: {mask_status}({mask_confidence:.2f})"
                        full_label = base_label + mask_label
                        
                        # Draw label
                        label_size = cv2.getTextSize(full_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(display_frame, (x1, y1 - label_size[1] - 10), 
                                    (x1 + label_size[0], y1), color, -1)
                        cv2.putText(display_frame, full_label, (x1, y1 - 5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Show pipeline status
                status = orchestrator.get_pipeline_status()
                cv2.putText(display_frame, f"Processors: {status['active_processors']}/{status['processor_count']}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(display_frame, f"Frames: {status['frame_count']}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Show mask detection stats
                if 'mask_metadata' in results:
                    meta = results['mask_metadata']
                    cv2.putText(display_frame, f"Masks: {meta['mask_count']}/{meta['total_processed']}", 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imshow('Complete Pipeline Test (Detection → Recognition → Mask)', display_frame)
                
                # Print detailed status every 30 frames
                if frame_count % 30 == 0:
                    print(f"📊 Complete Pipeline Status:")
                    print(f"   - Active processors: {status['active_processors']}")
                    print(f"   - Total frames: {status['frame_count']}")
                    print(f"   - Avg processing time: {status['avg_processing_time']:.3f}s")
                    
                    # Get component stats
                    if 'mask_metadata' in results:
                        print(f"   - Mask detection rate: {results['mask_metadata']['mask_count']}/{results['mask_metadata']['total_processed']}")
                    
                    # Get recognizer stats
                    recognizer = orchestrator.get_processor('FaceRecognizer')
                    if recognizer and hasattr(recognizer, 'get_robust_stats'):
                        stats = recognizer.get_robust_stats()
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
        print(f"❌ Complete pipeline test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_complete_pipeline()