# components/detection/mask_detector.py
from core.base_processor import BaseProcessor
import onnxruntime as ort
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
import time

class MaskDetector(BaseProcessor):
    """Fixed mask detection component with proper ONNX input format handling"""
    
    def __init__(self, name: str, config: Dict):
        super().__init__(name, config)
        self.mask_model = None
        self.input_size = (224, 224)
        self.confidence_threshold = 0.8
        self.roi_padding = 10
        
    def _initialize_impl(self) -> bool:
        """Initialize ONNX mask detection model"""
        try:
            model_path = self.config.get('mask_model_path')
            if not model_path:
                print("❌ MaskDetector: No mask_model_path in config")
                return False
                
            print(f"🔄 Loading mask model from: {model_path}")
            
            # Initialize ONNX Runtime session
            self.mask_model = ort.InferenceSession(str(model_path))
            
            # Get model information
            input_details = self.mask_model.get_inputs()[0]
            
            print(f"📋 Model Input Details:")
            print(f"   - Name: {input_details.name}")
            print(f"   - Shape: {input_details.shape}")
            print(f"   - Type: {input_details.type}")
            
            # Determine input format from shape
            # Common formats: [batch, channels, height, width] or [batch, height, width, channels]
            if len(input_details.shape) == 4:
                # Check if channels are in position 1 (NCHW) or position 3 (NHWC)
                if input_details.shape[1] == 3:  # NCHW format
                    self.input_size = (input_details.shape[2], input_details.shape[3])
                    self.input_format = "NCHW"
                elif input_details.shape[3] == 3:  # NHWC format
                    self.input_size = (input_details.shape[1], input_details.shape[2])
                    self.input_format = "NHWC"
                else:
                    # Default to NHWC for compatibility
                    self.input_size = (224, 224)
                    self.input_format = "NHWC"
                    print("⚠️ Could not determine input format, defaulting to NHWC")
            
            self.confidence_threshold = self.config.get('mask_detection_threshold', 0.8)
            self.roi_padding = self.config.get('mask_roi_padding', 10)
            
            print(f"✅ MaskDetector initialized successfully")
            print(f"   - Input size: {self.input_size}")
            print(f"   - Input format: {self.input_format}")
            print(f"   - Confidence threshold: {self.confidence_threshold}")
            
            return True
            
        except Exception as e:
            print(f"❌ MaskDetector initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _process_impl(self, data: Dict, context: Dict) -> Dict:
        """Perform mask detection on recognized faces"""
        # Check if we have recognitions to process
        if 'recognitions' not in data or not data['recognitions']:
            # No recognitions, return empty mask detections
            data['mask_detections'] = []
            data['mask_metadata'] = {
                'total_processed': 0,
                'mask_count': 0,
                'no_mask_count': 0,
                'timestamp': time.time()
            }
            return data
            
        frame = data.get('input_frame')
        if frame is None:
            print("⚠️ MaskDetector: No input frame available")
            data['mask_detections'] = []
            return data
        
        mask_detections = []
        
        for recognition in data['recognitions']:
            try:
                # Extract face ROI from original frame using bbox
                face_roi = self._extract_face_roi(frame, recognition['bbox'])
                if face_roi is None:
                    continue
                
                # Detect mask
                mask_status, confidence = self._detect_mask(face_roi)
                
                mask_detection = {
                    **recognition,  # Include all recognition data
                    'mask_status': mask_status,
                    'mask_confidence': confidence,
                    'has_mask': mask_status == "mask"
                }
                
                mask_detections.append(mask_detection)
                
            except Exception as e:
                print(f"⚠️ MaskDetector: Error processing recognition: {e}")
                continue
        
        # Add mask detections to pipeline data
        data['mask_detections'] = mask_detections
        data['mask_metadata'] = {
            'total_processed': len(mask_detections),
            'mask_count': len([md for md in mask_detections if md['has_mask']]),
            'no_mask_count': len([md for md in mask_detections if not md['has_mask']]),
            'timestamp': time.time()
        }
        
        return data
    
    def _extract_face_roi(self, frame: np.ndarray, bbox: List[int]) -> Optional[np.ndarray]:
        """Extract face region of interest for mask detection"""
        try:
            x1, y1, x2, y2 = bbox
            h, w = frame.shape[:2]
            
            # Add padding specific to mask detection
            x1_pad = max(0, x1 - self.roi_padding)
            y1_pad = max(0, y1 - self.roi_padding)
            x2_pad = min(w, x2 + self.roi_padding)
            y2_pad = min(h, y2 + self.roi_padding)
            
            # Ensure valid coordinates
            if x2_pad <= x1_pad or y2_pad <= y1_pad:
                return None
                
            face_roi = frame[y1_pad:y2_pad, x1_pad:x2_pad]
            
            # Validate ROI
            if (face_roi.size == 0 or face_roi.shape[0] < 20 or 
                face_roi.shape[1] < 20):
                return None
            
            return face_roi
            
        except Exception as e:
            print(f"⚠️ MaskDetector: Error extracting face ROI: {e}")
            return None
    
    def _detect_mask(self, face_roi: np.ndarray) -> Tuple[str, float]:
        """Perform mask detection on face ROI"""
        if self.mask_model is None:
            return "unknown", 0.0
            
        try:
            # Preprocess face ROI for mask detection
            processed_face = self._preprocess_face(face_roi)
            
            # Run inference
            input_name = self.mask_model.get_inputs()[0].name
            output_name = self.mask_model.get_outputs()[0].name
            
            outputs = self.mask_model.run([output_name], {input_name: processed_face})
            predictions = outputs[0][0]
            
            # Interpret results - assuming output is [mask_prob, no_mask_prob]
            mask_prob = float(predictions[0])
            without_mask_prob = float(predictions[1]) if len(predictions) > 1 else 1.0 - mask_prob
            
            # Apply confidence threshold
            if mask_prob > without_mask_prob and mask_prob >= self.confidence_threshold:
                return "mask", mask_prob
            elif without_mask_prob > mask_prob and without_mask_prob >= self.confidence_threshold:
                return "no_mask", without_mask_prob
            else:
                return "unknown", max(mask_prob, without_mask_prob)
                
        except Exception as e:
            print(f"⚠️ Mask detection error: {e}")
            import traceback
            traceback.print_exc()
            return "unknown", 0.0
    
    def _preprocess_face(self, face_roi: np.ndarray) -> np.ndarray:
        """Preprocess face ROI for mask detection model - FIXED VERSION"""
        try:
            # Resize to model input size
            resized = cv2.resize(face_roi, self.input_size, interpolation=cv2.INTER_AREA)
            
            # Convert to RGB and normalize
            rgb_face = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            normalized_face = rgb_face.astype(np.float32) / 255.0
            
            # Add batch dimension
            input_data = np.expand_dims(normalized_face, axis=0)  # Shape: (1, 224, 224, 3)
            
            # Format based on model requirements - FIXED LOGIC
            if self.input_format == "NCHW":
                # Convert from NHWC to NCHW: (1, 224, 224, 3) -> (1, 3, 224, 224)
                input_data = input_data.transpose(0, 3, 1, 2)
            # If NHWC, no transformation needed
            
            return input_data
            
        except Exception as e:
            print(f"❌ MaskDetector preprocessing error: {e}")
            raise
    
    def get_mask_stats(self) -> Dict[str, Any]:
        """Get mask detection statistics"""
        input_shape = self.mask_model.get_inputs()[0].shape if self.mask_model else None
        return {
            'confidence_threshold': self.confidence_threshold,
            'input_size': self.input_size,
            'input_shape': input_shape,
            'model_loaded': self.mask_model is not None,
            'roi_padding': self.roi_padding,
            'input_format': self.input_format
        }
    
    def cleanup(self):
        """Cleanup resources"""
        if self.mask_model is not None:
            del self.mask_model
            self.mask_model = None
        print("🧹 MaskDetector cleanup completed")