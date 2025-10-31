# components/detection/face_detector.py
from core.base_processor import BaseProcessor
from ultralytics import YOLO
from typing import Dict, List, Any
import numpy as np
import time

class FaceDetector(BaseProcessor):
    """Complete face detection component with YOLO integration"""
    
    def __init__(self, name: str, config: Dict):
        super().__init__(name, config)
        self.model = None  # 🆕 ADD THIS LINE
        self.confidence_threshold = config.get('detection_confidence', 0.6)
        self.iou_threshold = config.get('detection_iou', 0.6)
        
    def _initialize_impl(self) -> bool:
        """Initialize YOLO face detection model"""
        try:
            model_path = self.config.get('detection_model_path')
            if not model_path:
                print("❌ FaceDetector: No detection_model_path in config")
                return False
                
            self.model = YOLO(model_path)  # 🆕 Now this will work
            
            # Set detection parameters
            self.confidence_threshold = self.config.get('detection_confidence', 0.6)
            self.iou_threshold = self.config.get('detection_iou', 0.6)
            
            print(f"✅ FaceDetector initialized: {model_path}")
            print(f"   - Confidence: {self.confidence_threshold}")
            print(f"   - IoU: {self.iou_threshold}")
            
            return True
            
        except Exception as e:
            print(f"❌ FaceDetector initialization failed: {e}")
            return False
    
    def _process_impl(self, data: Dict, context: Dict) -> Dict:
        """Perform face detection on input frame"""
        if 'input_frame' not in data:
            print("⚠️ FaceDetector: No input_frame in data")
            return data
            
        frame = data['input_frame']
        
        # Validate frame
        if frame is None or frame.size == 0:
            print("⚠️ FaceDetector: Invalid frame")
            data['detections'] = []
            return data
        
        try:
            # Perform YOLO detection
            results = self.model(
                frame, 
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False
            )
            
            # Parse detections
            detections = self._parse_detections(results)
            
            # Add to pipeline data
            data['detections'] = detections
            data['detection_metadata'] = {
                'count': len(detections),
                'timestamp': time.time()
            }
            
            # Update context for temporal tracking
            context['last_detections'] = detections
            
            return data
            
        except Exception as e:
            print(f"❌ FaceDetector processing error: {e}")
            data['detections'] = []
            return data
    
    def _parse_detections(self, results) -> List[Dict]:
        """Parse YOLO results into standardized detection format"""
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Convert tensor to numpy and extract coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    
                    detection = {
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': confidence,
                        'type': 'face',
                        'track_id': None  # Will be set by tracker if available
                    }
                    
                    detections.append(detection)
        
        return detections
    
    def cleanup(self):
        """Cleanup resources"""
        if self.model is not None:  # 🆕 Check if model exists
            del self.model
            self.model = None
        print("🧹 FaceDetector cleanup completed")