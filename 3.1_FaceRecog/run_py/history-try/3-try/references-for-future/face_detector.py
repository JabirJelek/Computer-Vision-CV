import cv2
import torch
import numpy as np
from config import YOLO_MODEL_PATH

class FaceDetector:
    def __init__(self):
        self.model = self.load_yolo_model()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def load_yolo_model(self):
        """Load YOLO model from local path"""
        try:
            # For YOLO v11, use the YOLO class if it's an official model
            from ultralytics import YOLO
            model = YOLO(YOLO_MODEL_PATH)
            print("✅ YOLO model loaded successfully using ultralytics")
            return model
        except Exception as e:
            print(f"❌ Error loading YOLO model: {e}")
            return None
    
    def detect_faces(self, image):
        """Detect faces in image using YOLO"""
        if self.model is None:
            return []
        
        # Run inference
        results = self.model(image)
        
        # Extract face detections
        faces = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                if conf > 0.5:  # Confidence threshold
                    face_roi = image[int(y1):int(y2), int(x1):int(x2)]
                    faces.append({
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': conf,
                        'roi': face_roi
                    })
        
        return faces