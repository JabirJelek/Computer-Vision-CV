import onnxruntime as ort
import numpy as np
from pathlib import Path
import cv2
import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
from threading import Thread, Lock
from queue import Queue
import time
from typing import Dict, List, Tuple, Optional
import datetime
import csv
from keras.models import load_model

import cv2
import dlib
import numpy as np
from scipy.spatial import distance
from collections import deque
import math

class FixedLandmarkMaskDetector:
    def __init__(self, landmark_model_path="shape_predictor_68_face_landmarks.dat"):
        """
        Fixed landmark-based mask detection
        """
        try:
            # Initialize dlib's face detector and landmark predictor
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(landmark_model_path)
            print(f"✅ Landmark model loaded: {landmark_model_path}")
        except Exception as e:
            print(f"❌ Failed to load landmark model: {e}")
            raise
        
        # Landmark indices for different facial regions
        self.NOSE_BRIDGE = list(range(27, 31))      # Upper nose
        self.NOSE_TIP = [30, 31, 32, 33, 34, 35]    # Nose tip and nostrils
        self.MOUTH_OUTER = list(range(48, 60))      # Outer lip
        self.MOUTH_INNER = list(range(60, 68))      # Inner lip
        self.CHEEKS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        self.JAWLINE = list(range(0, 17))           # Jawline
        
        # Analysis parameters
        self.mask_confidence_threshold = 0.7
        self.min_face_size = 100
        
        # Performance tracking
        self.analysis_history = deque(maxlen=30)
        self.confidence_smoothing = 0.9
        
    def get_landmark_regions(self):
        """Define comprehensive facial regions for analysis"""
        return {
            'nose_bridge': self.NOSE_BRIDGE,
            'nose_tip': self.NOSE_TIP,
            'mouth_outer': self.MOUTH_OUTER,
            'mouth_inner': self.MOUTH_INNER,
            'cheeks': self.CHEEKS,
            'jawline': self.JAWLINE
        }
    
    def detect_faces(self, image):
        """Detect faces using dlib"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 0)  # No upsampling for speed
        return faces, gray
    
    def extract_landmarks(self, image, face_rect):
        """Extract facial landmarks for a detected face"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        landmarks = self.predictor(gray, face_rect)
        return landmarks
    
    def calculate_region_visibility(self, image, landmarks, region_indices):
        """Calculate visibility score for a specific facial region"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Extract region points
        points = []
        for idx in region_indices:
            point = landmarks.part(idx)
            points.append((point.x, point.y))
        
        if not points:
            return 0.0
        
        # Calculate bounding box
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        
        x_min, x_max = max(0, min(x_coords)), min(width, max(x_coords))
        y_min, y_max = max(0, min(y_coords)), min(height, max(y_coords))
        
        if x_max <= x_min or y_max <= y_min:
            return 0.0
        
        # Extract region
        region_roi = gray[int(y_min):int(y_max), int(x_min):int(x_max)]
        
        if region_roi.size == 0:
            return 0.0
        
        # Calculate visibility metrics
        contrast = np.std(region_roi) / 255.0
        
        edges = cv2.Canny(region_roi, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        laplacian_var = cv2.Laplacian(region_roi, cv2.CV_64F).var()
        texture = min(laplacian_var / 1000.0, 1.0)
        
        # Simple weighted combination
        visibility_score = 0.4 * contrast + 0.4 * edge_density + 0.2 * texture
        
        return visibility_score
    
    def detect_mask_simple(self, image, face_rect):
        """
        Simplified mask detection using landmark analysis
        Returns: (is_wearing_mask, confidence, landmarks)
        """
        try:
            landmarks = self.extract_landmarks(image, face_rect)
            regions = self.get_landmark_regions()
            
            # Focus on critical regions
            critical_regions = ['mouth_outer', 'mouth_inner', 'nose_tip']
            mask_evidence = 0.0
            total_weight = 0.0
            
            for region in critical_regions:
                visibility = self.calculate_region_visibility(
                    image, landmarks, regions[region]
                )
                # Low visibility suggests mask
                mask_evidence += (1 - visibility) * 0.33
                total_weight += 0.33
            
            mask_confidence = mask_evidence
            
            # Apply smoothing
            if self.analysis_history:
                prev_confidence = np.mean(list(self.analysis_history)[-5:])
                mask_confidence = (self.confidence_smoothing * prev_confidence + 
                                (1 - self.confidence_smoothing) * mask_confidence)
            
            self.analysis_history.append(mask_confidence)
            
            is_wearing_mask = mask_confidence > self.mask_confidence_threshold
            
            return is_wearing_mask, mask_confidence, landmarks
            
        except Exception as e:
            print(f"Landmark analysis error: {e}")
            return False, 0.0, None
    
    def process_frame(self, frame):
        """Process frame and return consistent results format"""
        faces, gray = self.detect_faces(frame)
        results = []
        
        for face in faces:
            face_size = max(face.width(), face.height())
            if face_size < self.min_face_size:
                continue
            
            is_wearing_mask, confidence, landmarks = self.detect_mask_simple(frame, face)
            
            # Return in consistent format with your existing system
            result = {
                'bbox': (face.left(), face.top(), face.right(), face.bottom()),
                'is_wearing_mask': is_wearing_mask,
                'mask_confidence': confidence,  # This is the key that was missing!
                'landmarks': landmarks,
                'detection_confidence': 1.0,  # Placeholder
                'identity': None,
                'recognition_confidence': 0.0
            }
            results.append(result)
        
        return results

class FixedMaskDetectionAnalyzer:
    """
    Fixed analyzer that works with the corrected results format
    """
    def __init__(self):
        self.detection_history = []
    
    def analyze_detection_patterns(self, results):
        """Fixed analysis method with proper key access"""
        if not results:
            return {
                'total_faces': 0,
                'masked_faces': 0,
                'mask_rate': 0.0,
                'average_mask_confidence': 0.0,
                'confidence_std': 0.0,
                'high_confidence_masks': 0,
                'low_confidence_masks': 0
            }
        
        # Use the correct key 'mask_confidence' 
        confidences = [r.get('mask_confidence', 0.0) for r in results]
        
        analysis = {
            'total_faces': len(results),
            'masked_faces': sum(1 for r in results if r.get('is_wearing_mask', False)),
            'average_mask_confidence': np.mean(confidences) if confidences else 0.0,
            'confidence_std': np.std(confidences) if confidences else 0.0
        }
        
        analysis['mask_rate'] = (
            analysis['masked_faces'] / analysis['total_faces'] 
            if analysis['total_faces'] > 0 else 0.0
        )
        
        # Confidence distribution
        analysis['high_confidence_masks'] = sum(1 for c in confidences if c > 0.8)
        analysis['low_confidence_masks'] = sum(1 for c in confidences if c < 0.3)
        
        self.detection_history.append(analysis)
        return analysis
    
    def draw_simple_visualization(self, image, results, analysis):
        """Simple visualization without complex plots"""
        output = image.copy()
        height, width = output.shape[:2]
        
        # Add analysis overlay
        overlay = output.copy()
        cv2.rectangle(overlay, (0, 0), (300, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, output, 0.3, 0, output)
        
        # Add stats
        stats = [
            f"Faces: {analysis['total_faces']}",
            f"Masked: {analysis['masked_faces']}",
            f"Mask Rate: {analysis['mask_rate']:.1%}",
            f"Avg Conf: {analysis['average_mask_confidence']:.3f}"
        ]
        
        for i, stat in enumerate(stats):
            cv2.putText(output, stat, (10, 30 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Draw bounding boxes
        for result in results:
            bbox = result['bbox']
            is_wearing_mask = result.get('is_wearing_mask', False)
            confidence = result.get('mask_confidence', 0.0)
            
            color = (0, 0, 255) if is_wearing_mask else (0, 255, 0)
            cv2.rectangle(output, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            status = "MASK" if is_wearing_mask else "NO MASK"
            label = f"{status} ({confidence:.2f})"
            cv2.putText(output, label, (bbox[0], bbox[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return output

def test_fixed_landmark_mask_detection():
    """Fixed test function"""
    try:
        # Initialize with error handling
        detector = FixedLandmarkMaskDetector("shape_predictor_68_face_landmarks.dat")
        analyzer = FixedMaskDetectionAnalyzer()
        
        # Test with webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Cannot open webcam")
            return
        
        print("🚀 Starting Fixed Landmark Mask Detection")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to capture frame")
                break
            
            # Process frame
            results = detector.process_frame(frame)
            analysis = analyzer.analyze_detection_patterns(results)  # This should work now
            
            # Draw visualization
            output_frame = analyzer.draw_simple_visualization(frame, results, analysis)
            
            cv2.imshow('Fixed Landmark Mask Detection', output_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Test completed successfully")
        
    except Exception as e:
        print(f"❌ Error in test: {e}")
        import traceback
        traceback.print_exc()

# Alternative: Integration with your existing system
def integrate_with_existing_system():
    """How to integrate landmark detection with your existing system"""
    
    class EnhancedFaceSystem:
        def __init__(self, config):
            self.config = config
            self.face_system = FaceRecognitionSystem(config)
            
            # Add landmark detector
            try:
                self.landmark_detector = FixedLandmarkMaskDetector(
                    config.get('landmark_model_path', 'shape_predictor_68_face_landmarks.dat')
                )
                self.use_landmarks = True
            except:
                print("⚠️  Landmark detector not available, using ONNX only")
                self.use_landmarks = False
        
        def detect_mask_with_landmarks(self, frame, bbox):
            """Use landmarks for mask detection"""
            if not self.use_landmarks:
                return False, 0.0
            
            try:
                x1, y1, x2, y2 = bbox
                dlib_rect = dlib.rectangle(x1, y1, x2, y2)
                
                is_wearing_mask, confidence, _ = self.landmark_detector.detect_mask_simple(
                    frame, dlib_rect
                )
                return is_wearing_mask, confidence
            except Exception as e:
                print(f"Landmark mask detection failed: {e}")
                return False, 0.0
        
        def enhanced_process_frame(self, frame):
            """Enhanced processing with landmark support"""
            # Your existing detection
            detections = self.face_system.detect_faces(frame)
            results = []
            
            for detection in detections:
                bbox = detection['bbox']
                
                # Try landmark-based mask detection first
                is_wearing_mask, mask_confidence = self.detect_mask_with_landmarks(frame, bbox)
                
                # Fallback to ONNX if landmarks fail
                if not self.use_landmarks or mask_confidence < 0.3:
                    if self.face_system.mask_session:
                        # Extract ROI for ONNX
                        x1, y1, x2, y2 = bbox
                        padding = self.config.get('roi_padding', 10)
                        h, w = frame.shape[:2]
                        face_roi = frame[
                            max(0, y1-padding):min(h, y2+padding),
                            max(0, x1-padding):min(w, x2+padding)
                        ]
                        if face_roi.size > 0:
                            is_wearing_mask, onnx_confidence = self.face_system.detect_mask_fixed(face_roi)
                            mask_confidence = max(mask_confidence, onnx_confidence)
                
                # Face recognition (only if no mask)
                identity = None
                rec_confidence = 0.0
                embedding = None
                
                if not is_wearing_mask:
                    embedding = self.face_system.extract_embedding(face_roi)
                    if embedding is not None:
                        identity, rec_confidence = self.face_system.recognize_face(embedding)
                
                results.append({
                    'bbox': bbox,
                    'detection_confidence': detection['confidence'],
                    'identity': identity,
                    'recognition_confidence': rec_confidence,
                    'embedding': embedding.tolist() if embedding is not None else None,
                    'is_wearing_mask': is_wearing_mask,
                    'mask_confidence': mask_confidence,  # This key is now guaranteed!
                    'mask_method': 'landmark' if self.use_landmarks else 'onnx'
                })
            
            return results

    return EnhancedFaceSystem

       
class RealTimeLandmarkMaskDetector(FixedMaskDetectionAnalyzer):
    """
    Optimized version for real-time performance
    """
    def __init__(self, landmark_model_path, processing_interval=5):
        super().__init__(landmark_model_path)
        self.processing_interval = processing_interval
        self.frame_count = 0
        self.last_results = []
        
        # Performance optimizations
        self.min_face_size = 80  # Smaller for faster processing
        self.detection_scale = 0.5  # Scale down for detection
        
    def process_frame_optimized(self, frame):
        """Optimized frame processing for real-time performance"""
        self.frame_count += 1
        
        # Only process every N frames
        if self.frame_count % self.processing_interval != 0:
            return self.last_results
        
        # Resize frame for faster processing
        original_height, original_width = frame.shape[:2]
        processing_frame = cv2.resize(frame, 
                                    (int(original_width * self.detection_scale), 
                                     int(original_height * self.detection_scale)))
        
        # Detect faces on smaller frame
        faces, gray = self.detect_faces(processing_frame)
        results = []
        
        for face in faces:
            # Scale face coordinates back to original
            scaled_face = dlib.rectangle(
                int(face.left() / self.detection_scale),
                int(face.top() / self.detection_scale),
                int(face.right() / self.detection_scale),
                int(face.bottom() / self.detection_scale)
            )
            
            # Analyze on original resolution for accuracy
            result = self.detect_mask_advanced(frame, scaled_face)
            result['bbox'] = (scaled_face.left(), scaled_face.top(), 
                            scaled_face.right(), scaled_face.bottom())
            results.append(result)
        
        self.last_results = results
        return results
    
    def get_performance_stats(self):
        """Get performance statistics"""
        return {
            'frames_processed': self.frame_count,
            'average_confidence': np.mean([r['confidence'] for r in self.last_results]) if self.last_results else 0,
            'mask_detection_rate': np.mean([r['is_wearing_mask'] for r in self.last_results]) if self.last_results else 0
        }
        
class EnhancedFaceRecognitionSystem:
    """
    Enhanced system integrating landmark-based mask detection
    """
    def __init__(self, config):
        self.config = config
        self.face_system = FaceRecognitionSystem(config)
        
        # Initialize landmark-based mask detector
        self.landmark_detector = FixedMaskDetectionAnalyzer(
            config.get('landmark_model_path', 'shape_predictor_68_face_landmarks.dat')
        )
        
        # Detection mode: 'onnx', 'landmark', 'hybrid'
        self.mask_detection_mode = config.get('mask_detection_mode', 'hybrid')
        
    def detect_mask_hybrid(self, face_roi, original_frame, face_bbox):
        """
        Hybrid mask detection using both ONNX and landmark analysis
        """
        results = {}
        
        # Method 1: ONNX model (if available)
        if self.mask_detection_mode in ['onnx', 'hybrid'] and self.face_system.mask_session:
            onnx_result = self.face_system.detect_mask_fixed(face_roi)
            results['onnx'] = onnx_result
        
        # Method 2: Landmark analysis
        if self.mask_detection_mode in ['landmark', 'hybrid']:
            # Convert face bbox to dlib rectangle
            x1, y1, x2, y2 = face_bbox
            dlib_rect = dlib.rectangle(x1, y1, x2, y2)
            
            try:
                landmark_result = self.landmark_detector.detect_mask_advanced(
                    original_frame, dlib_rect
                )
                results['landmark'] = landmark_result
            except Exception as e:
                print(f"Landmark analysis failed: {e}")
                results['landmark'] = {'is_wearing_mask': False, 'confidence': 0.0}
        
        # Combine results based on mode
        if self.mask_detection_mode == 'hybrid':
            # Weighted combination
            onnx_confidence = results.get('onnx', (False, 0.0))[1] if 'onnx' in results else 0.0
            landmark_confidence = results['landmark']['confidence'] if 'landmark' in results else 0.0
            
            # Give more weight to landmark analysis when available
            if landmark_confidence > 0.5:  # Landmark analysis is confident
                final_confidence = 0.3 * onnx_confidence + 0.7 * landmark_confidence
                is_wearing_mask = final_confidence > 0.6
            else:
                final_confidence = onnx_confidence
                is_wearing_mask = results.get('onnx', (False, 0.0))[0]
                
            return is_wearing_mask, final_confidence
        
        elif self.mask_detection_mode == 'landmark':
            landmark_result = results.get('landmark', {'is_wearing_mask': False, 'confidence': 0.0})
            return landmark_result['is_wearing_mask'], landmark_result['confidence']
        
        else:  # onnx
            return results.get('onnx', (False, 0.0))
    
    def process_frame_with_enhanced_mask_detection(self, frame):
        """
        Enhanced processing pipeline with hybrid mask detection
        """
        # Existing face detection
        detections = self.face_system.detect_faces(frame)
        results = []
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            
            # Extract face ROI
            padding = self.config.get('roi_padding', 10)
            h, w = frame.shape[:2]
            x1_pad = max(0, x1 - padding)
            y1_pad = max(0, y1 - padding)
            x2_pad = min(w, x2 + padding)
            y2_pad = min(h, y2 + padding)
            
            face_roi = frame[y1_pad:y2_pad, x1_pad:x2_pad]
            
            if face_roi.size == 0:
                continue
            
            # Enhanced mask detection
            is_wearing_mask, mask_confidence = self.detect_mask_hybrid(
                face_roi, frame, (x1_pad, y1_pad, x2_pad, y2_pad)
            )
            
            # Face recognition (only if no mask)
            identity = None
            recognition_confidence = 0.0
            embedding = None
            
            if not is_wearing_mask:
                embedding = self.face_system.extract_embedding(face_roi)
                if embedding is not None:
                    identity, recognition_confidence = self.face_system.recognize_face(embedding)
            
            results.append({
                'bbox': detection['bbox'],
                'detection_confidence': detection['confidence'],
                'identity': identity,
                'recognition_confidence': recognition_confidence,
                'embedding': embedding.tolist() if embedding is not None else None,
                'is_wearing_mask': is_wearing_mask,
                'mask_confidence': mask_confidence,
                'mask_detection_method': self.mask_detection_mode
            })
        
        return results            

class MaskDetectionAnalyzer:
    """
    Advanced analysis tools for mask detection performance
    """
    def __init__(self):
        self.detection_history = []
        self.performance_metrics = {}
    
    def analyze_detection_patterns(self, results):
        """Analyze patterns in mask detection results"""
        if not results:
            return {}
        
        analysis = {
            'total_faces': len(results),
            'masked_faces': sum(1 for r in results if r['is_wearing_mask']),
            'average_mask_confidence': np.mean([r['mask_confidence'] for r in results]),
            'confidence_std': np.std([r['mask_confidence'] for r in results])
        }
        
        analysis['mask_rate'] = analysis['masked_faces'] / analysis['total_faces']
        
        # Confidence distribution analysis
        confidences = [r['mask_confidence'] for r in results]
        analysis['high_confidence_masks'] = sum(1 for c in confidences if c > 0.8)
        analysis['low_confidence_masks'] = sum(1 for c in confidences if c < 0.3)
        
        self.detection_history.append(analysis)
        return analysis
    
    def generate_visual_analysis_report(self, image, results, analysis):
        """Generate comprehensive visual analysis report"""
        report_image = image.copy()
        height, width = report_image.shape[:2]
        
        # Create sidebar for analytics
        sidebar_width = 300
        sidebar = np.zeros((height, sidebar_width, 3), dtype=np.uint8)
        
        # Add analysis text
        y_offset = 30
        text_lines = [
            "MASK DETECTION ANALYSIS",
            "=" * 20,
            f"Total Faces: {analysis['total_faces']}",
            f"Masked: {analysis['masked_faces']}",
            f"Mask Rate: {analysis['mask_rate']:.1%}",
            f"Avg Confidence: {analysis['average_mask_confidence']:.3f}",
            f"High Confidence: {analysis['high_confidence_masks']}",
            f"Low Confidence: {analysis['low_confidence_masks']}"
        ]
        
        for line in text_lines:
            cv2.putText(sidebar, line, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
        
        # Add confidence distribution
        if self.detection_history:
            confidences = [a['average_mask_confidence'] for a in self.detection_history[-20:]]
            self._plot_confidence_trend(sidebar, confidences, y_offset)
        
        # Combine with original image
        report_image = np.hstack([report_image, sidebar])
        
        return report_image
    
    def _plot_confidence_trend(self, image, confidences, start_y):
        """Plot confidence trend graph"""
        if len(confidences) < 2:
            return
        
        plot_height = 100
        plot_width = 280
        plot_x, plot_y = 10, start_y + 10
        
        # Create plot background
        cv2.rectangle(image, (plot_x, plot_y), 
                     (plot_x + plot_width, plot_y + plot_height), 
                     (50, 50, 50), -1)
        
        # Normalize confidences to plot height
        max_conf = max(confidences) if max(confidences) > 0 else 1.0
        normalized_conf = [c / max_conf * plot_height for c in confidences]
        
        # Plot points
        x_step = plot_width / (len(confidences) - 1)
        points = []
        
        for i, conf in enumerate(normalized_conf):
            x = int(plot_x + i * x_step)
            y = int(plot_y + plot_height - conf)
            points.append((x, y))
            
            # Draw point
            cv2.circle(image, (x, y), 3, (0, 255, 255), -1)
        
        # Draw trend line
        if len(points) > 1:
            for i in range(len(points) - 1):
                cv2.line(image, points[i], points[i + 1], (0, 255, 255), 2)
        
        # Add labels
        cv2.putText(image, "Confidence Trend", (plot_x, plot_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
class CorrectedEnhancedFaceSystem:
    """Corrected integration with landmark detection"""
    
    def __init__(self, config):
        self.config = config
        self.face_system = FaceRecognitionSystem(config)
        
        # Initialize landmark detector
        try:
            landmark_path = config.get('landmark_model_path', 'shape_predictor_68_face_landmarks.dat')
            self.landmark_detector = FixedLandmarkMaskDetector(landmark_path)
            self.use_landmarks = True
            print("✅ Landmark mask detection enabled")
        except Exception as e:
            print(f"❌ Landmark detector failed: {e}")
            self.use_landmarks = False
    
    def enhanced_mask_detection(self, frame, face_bbox):
        """Use landmarks for mask detection when available"""
        if not self.use_landmarks:
            # Fallback to ONNX
            if self.face_system.mask_session:
                x1, y1, x2, y2 = face_bbox
                padding = self.config.get('roi_padding', 10)
                h, w = frame.shape[:2]
                face_roi = frame[
                    max(0, y1-padding):min(h, y2+padding),
                    max(0, x1-padding):min(w, x2+padding)
                ]
                if face_roi.size > 0:
                    return self.face_system.detect_mask_fixed(face_roi)
            return False, 0.0
        
        try:
            # Convert bbox to dlib rectangle
            x1, y1, x2, y2 = face_bbox
            dlib_rect = dlib.rectangle(x1, y1, x2, y2)
            
            # Use landmark detection
            is_wearing_mask, confidence, _ = self.landmark_detector.detect_mask_simple(frame, dlib_rect)
            return is_wearing_mask, confidence
            
        except Exception as e:
            print(f"Landmark detection failed: {e}")
            return False, 0.0
    
    def process_frame_enhanced(self, frame):
        """Enhanced processing with landmark support"""
        # Detect faces using YOLO
        detections = self.face_system.detect_faces(frame)
        results = []
        
        for detection in detections:
            bbox = detection['bbox']
            
            # Enhanced mask detection
            is_wearing_mask, mask_confidence = self.enhanced_mask_detection(frame, bbox)
            
            # Face recognition (only if no mask)
            identity = None
            recognition_confidence = 0.0
            embedding = None
            
            if not is_wearing_mask:
                # Extract face ROI for recognition
                x1, y1, x2, y2 = bbox
                padding = self.config.get('roi_padding', 10)
                h, w = frame.shape[:2]
                face_roi = frame[
                    max(0, y1-padding):min(h, y2+padding),
                    max(0, x1-padding):min(w, x2+padding)
                ]
                
                if face_roi.size > 0:
                    embedding = self.face_system.extract_embedding(face_roi)
                    if embedding is not None:
                        identity, recognition_confidence = self.face_system.recognize_face(embedding)
            
            results.append({
                'bbox': bbox,
                'detection_confidence': detection['confidence'],
                'identity': identity,
                'recognition_confidence': recognition_confidence,
                'embedding': embedding.tolist() if embedding is not None else None,
                'is_wearing_mask': is_wearing_mask,
                'mask_confidence': mask_confidence,
                'mask_method': 'landmark' if self.use_landmarks else 'onnx'
            })
        
        return results
            

def test_landmark_mask_detection():
    """Test function for landmark-based mask detection"""
    
    try:
        # Initialize the DETECTOR (not analyzer)
        detector = FixedLandmarkMaskDetector(r"D:\SCMA\3-APD\fromAraya\Computer-Vision-CV\3.1_FaceRecog\run_py\shape_predictor_68_face_landmarks.dat")
        
        # Test with webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Cannot open webcam")
            return
        
        print("🚀 Starting CORRECTED Landmark-Based Mask Detection")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame using the DETECTOR
            results = detector.process_frame(frame)
            
            # Draw results directly (simplified)
            output_frame = frame.copy()
            
            for result in results:
                bbox = result['bbox']
                is_wearing_mask = result.get('is_wearing_mask', False)
                confidence = result.get('mask_confidence', 0.0)
                
                # Color coding
                color = (0, 0, 255) if is_wearing_mask else (0, 255, 0)  # Red for mask, Green for no mask
                
                # Draw bounding box
                x1, y1, x2, y2 = bbox
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                status = "MASK" if is_wearing_mask else "NO MASK"
                label = f"{status} ({confidence:.2f})"
                cv2.putText(output_frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Show frame count and stats
            cv2.putText(output_frame, f"Faces detected: {len(results)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('CORRECTED Landmark Mask Detection', output_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Test completed successfully")
        
    except Exception as e:
        print(f"❌ Error in test: {e}")
        import traceback
        traceback.print_exc()
        
# Update your main function to use the corrected system
def main_enhanced():
    """Main function with enhanced landmark detection"""
    
    # Enhanced config with landmark model
    ENHANCED_CONFIG = {
        'detection_model_path': r'D:\SCMA\3-APD\fromAraya\Computer-Vision-CV\3.1_FaceRecog\yolov11n-face.pt',
        'embeddings_db_path': r'D:\SCMA\3-APD\fromAraya\Computer-Vision-CV\3.1_FaceRecog\person_folder1.4.json',
        'detection_confidence': 0.6,
        'detection_iou': 0.5,
        'roi_padding': 10,
        'embedding_model': 'Facenet',
        'recognition_threshold': 0.5,
        
        # Mask detection config
        'mask_model_path': r'D:\SCMA\3-APD\fromAraya\Computer-Vision-CV\3.1_FaceRecog\run_py\mask_detector2.onnx',
        'mask_confidence_threshold': 0.4,
        'enable_mask_detection': True,
        
        # Landmark model path
        'landmark_model_path': r'D:\SCMA\3-APD\fromAraya\Computer-Vision-CV\3.1_FaceRecog\run_py\shape_predictor_68_face_landmarks.dat'
    }
    
    try:
        # Initialize enhanced system
        enhanced_system = CorrectedEnhancedFaceSystem(ENHANCED_CONFIG)
        
        # Create processor
        processor = RealTimeProcessor(
            face_system=enhanced_system.face_system,  # Use the underlying face system
            processing_interval=5,
            buffer_size=5
        )
        
        # Override the process_frame method to use enhanced detection
        original_process_frame = processor.face_system.process_frame
        
        def enhanced_process_frame(frame):
            return enhanced_system.process_frame_enhanced(frame)
        
        processor.face_system.process_frame = enhanced_process_frame
        
        # Choose source
        source = '0'  # Default camera
        
        # Configure display
        processor.set_display_size(1024, 768, "fixed_size")
        
        print("🚀 Starting Enhanced System with Landmark Detection")
        processor.run(source)
        
    except Exception as e:
        print(f"❌ Error in enhanced system: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # You can choose which test to run:
    
    # Option 1: Run the corrected landmark-only test
    test_landmark_mask_detection()
    
    # Option 2: Run the enhanced system with landmark integration
    # main_enhanced()

class FaceRecognitionSystem:
    def __init__(self, config: Dict):
        self.config = config
        self.detection_model = None
        self.embeddings_db = {}
        self.identity_centroids = {}
        self.mask_session = None
        self.mask_detector_enabled = config.get('enable_mask_detection', True)
        
        # Debug statistics
        self.debug_stats = {
            'total_frames_processed': 0,
            'total_faces_detected': 0,
            'total_faces_recognized': 0,
            'total_masked_faces': 0,
            'detection_times': [],
            'embedding_times': [],
            'recognition_times': [],
            'mask_detection_times': [],
            'last_processing_time': 0
        }
        
        self._load_models()
        self._load_embeddings_database()
        self._load_mask_detector_onnx_fixed()
        
    def add_landmark_detection(self, landmark_model_path):
        """Add landmark-based mask detection to existing system"""
        try:
            self.landmark_detector = FixedLandmarkMaskDetector(landmark_model_path)
            self.use_landmark_masks = True
            print("✅ Landmark mask detection enabled")
        except Exception as e:
            print(f"❌ Landmark detection failed: {e}")
            self.use_landmark_masks = False

    def enhanced_mask_detection(self, frame, bbox):
        """Enhanced mask detection using landmarks + ONNX"""
        if hasattr(self, 'use_landmark_masks') and self.use_landmark_masks:
            # Try landmarks first
            x1, y1, x2, y2 = bbox
            dlib_rect = dlib.rectangle(x1, y1, x2, y2)
            landmark_mask, landmark_conf = self.landmark_detector.detect_mask_simple(frame, dlib_rect)
            
            if landmark_conf > 0.5:  # Confident landmark detection
                return landmark_mask, landmark_conf
        
        # Fallback to ONNX
        if self.mask_session:
            # Your existing ONNX detection code here
            pass
        
        return False, 0.0        
        
    def _load_mask_detector_onnx_fixed(self):
        """Load mask detector with NHWC format fix"""
        if not self.mask_detector_enabled:
            print("😷 Mask detection disabled in config")
            return
            
        try:
            mask_model_path = self.config.get('mask_model_path', '')
            if not mask_model_path or not Path(mask_model_path).exists():
                print(f"❌ Mask model not found at: {mask_model_path}")
                return
        
            print(f"🔄 Loading ONNX mask detection model...")
            self.mask_session = ort.InferenceSession(
                mask_model_path,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            
            model_input = self.mask_session.get_inputs()[0]
            self.mask_input_name = model_input.name
            self.mask_output_name = self.mask_session.get_outputs()[0].name
            
            print(f"✅ ONNX mask detection model loaded")
            print(f"   - Input: {self.mask_input_name}, Output: {self.mask_output_name}")
            print(f"   - Input shape: {model_input.shape}")
            print("   - Using NHWC format: [batch, 224, 224, 3]")
            
            # Test the model
            self._test_onnx_model_fixed()
            
        except Exception as e:
            print(f"❌ Failed to load mask detection model: {e}")
            self.mask_session = None
            
    def _test_onnx_model_fixed(self):
        """Test ONNX model with sample data"""
        if self.mask_session is None:
            return
            
        try:
            print("🧪 Testing ONNX model with sample input...")
            # Create test input
            test_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
            
            # Run inference
            outputs = self.mask_session.run(
                [self.mask_output_name], 
                {self.mask_input_name: test_input}
            )
            
            predictions = outputs[0]
            print(f"✅ ONNX test successful - Output shape: {predictions.shape}")
            
        except Exception as e:
            print(f"❌ ONNX model test failed: {e}")            
                   
    def _load_mask_detector_onnx(self):
        """Load the mask detection model in ONNX format - FIXED VERSION"""
        if not self.mask_detector_enabled:
            print("😷 Mask detection disabled in config")
            return
            
        try:
            mask_model_path = self.config.get('mask_model_path', '')
            if not mask_model_path:
                print("⚠️  No mask model path specified in config")
                return
                
            # Check if file exists
            if not Path(mask_model_path).exists():
                print(f"❌ Mask model not found at: {mask_model_path}")
                return
            
            print(f"🔄 Loading ONNX mask detection model from: {mask_model_path}")
            
            # Create ONNX Runtime session
            self.mask_session = ort.InferenceSession(
                mask_model_path,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            
            # Get model input information
            model_input = self.mask_session.get_inputs()[0]
            self.mask_input_name = model_input.name
            self.mask_output_name = self.mask_session.get_outputs()[0].name
            self.mask_input_shape = model_input.shape
            
            print(f"✅ ONNX mask detection model loaded successfully")
            print(f"   - Input name: {self.mask_input_name}")
            print(f"   - Output name: {self.mask_output_name}")
            print(f"   - Input shape: {self.mask_input_shape}")
            
            # Based on your error, the model expects [?, 224, 224, 3] (NHWC)
            # So we'll use NHWC format
            self.mask_input_format = 'nhwc'
            print("   - Using NHWC input format: [batch, 224, 224, 3]")
            
            # Test the model with the correct format
            self._test_onnx_model_fixed()
            
        except Exception as e:
            print(f"❌ Failed to load ONNX mask detection model: {e}")
            import traceback
            traceback.print_exc()
            print("⚠️  Continuing without mask detection")
            self.mask_session = None                
                                                                 
    def _load_models(self):
        """Load YOLO face detection model from local path"""
        try:
            model_path = Path(self.config['detection_model_path'])
            if not model_path.exists():
                raise FileNotFoundError(f"YOLO model not found at {model_path}")
                
            self.detection_model = YOLO(str(model_path))
            print(f"✅ YOLO model loaded from {model_path}")
            
        except Exception as e:
            print(f"❌ Failed to load YOLO model: {e}")
            raise
            
    def _load_embeddings_database(self):
        """Load pre-computed face embeddings from JSON with your structure"""
        try:
            db_path = Path(self.config['embeddings_db_path'])
            if not db_path.exists():
                print("⚠️  Embeddings database not found, starting fresh")
                self.embeddings_db = {"persons": {}, "metadata": {}}
                return
                
            with open(db_path, 'r') as f:
                self.embeddings_db = json.load(f)
                
            if "persons" in self.embeddings_db:
                for person_id, person_data in self.embeddings_db["persons"].items():
                    display_name = person_data["display_name"]
                    centroid = person_data["centroid_embedding"]
                    self.identity_centroids[display_name] = np.array(centroid)
                    
                print(f"✅ Loaded {len(self.identity_centroids)} identities from database")
                print(f"📊 Available persons: {list(self.identity_centroids.keys())}")
                
            else:
                print("⚠️  No 'persons' key found in JSON database")
                
        except Exception as e:
            print(f"❌ Failed to load embeddings database: {e}")
            raise

    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """Detect faces using YOLO with optimized settings"""
        start_time = time.time()
        try:
            results = self.detection_model(
                frame, 
                conf=self.config['detection_confidence'],
                iou=self.config['detection_iou'],
                verbose=False
            )
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        
                        detections.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(confidence)
                        })
            
            # Update debug stats
            detection_time = (time.time() - start_time) * 1000
            self.debug_stats['detection_times'].append(detection_time)
            self.debug_stats['detection_times'] = self.debug_stats['detection_times'][-100:]
            
            return detections
            
        except Exception as e:
            print(f"Detection error: {e}")
            return []

    def extract_embedding(self, face_roi: np.ndarray) -> Optional[np.ndarray]:
        """Optimized embedding extraction with better error handling"""
        start_time = time.time()
        
        # Validate ROI dimensions more thoroughly
        if (face_roi.size == 0 or face_roi.shape[0] < 50 or face_roi.shape[1] < 50 or 
            np.max(face_roi) - np.min(face_roi) < 10):  # Check for low contrast
            return None
            
        try:
            # Convert to RGB and ensure proper data type
            if len(face_roi.shape) == 3:
                face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            else:
                face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2RGB)
            
            # Normalize pixel values
            face_rgb = face_rgb.astype(np.float32) / 255.0
            
            embedding_obj = DeepFace.represent(
                face_rgb,
                model_name=self.config['embedding_model'],
                enforce_detection=False,
                detector_backend='skip',
                align=True  # Add face alignment for better accuracy
            )
            
            if embedding_obj and len(embedding_obj) > 0:
                embedding_time = (time.time() - start_time) * 1000
                self.debug_stats['embedding_times'].append(embedding_time)
                self.debug_stats['embedding_times'] = self.debug_stats['embedding_times'][-100:]
                
                return np.array(embedding_obj[0]['embedding'])
                
        except Exception as e:
            if self.config.get('verbose', False):
                print(f"Embedding extraction error: {e}")
                
        return None

    def recognize_face(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """Enhanced matching with multiple similarity strategies"""
        start_time = time.time()
        
        if not self.identity_centroids:
            return None, 0.0
            
        best_similarity = -1.0
        best_identity = None
        
        embedding = embedding.flatten()
        
        for identity, centroid in self.identity_centroids.items():
            centroid = centroid.flatten()
            
            # Cosine similarity (primary)
            cosine_sim = cosine_similarity([embedding], [centroid])[0][0]
            
            # Optional: Euclidean distance (normalized to 0-1)
            euclidean_dist = np.linalg.norm(embedding - centroid)
            euclidean_sim = 1 / (1 + euclidean_dist)  # Convert distance to similarity
            
            # Combine strategies (weighted)
            final_similarity = 0.8 * cosine_sim + 0.2 * euclidean_sim
            
            if final_similarity > best_similarity and final_similarity >= self.config['recognition_threshold']:
                best_similarity = final_similarity
                best_identity = identity
        
        recognition_time = (time.time() - start_time) * 1000
        self.debug_stats['recognition_times'].append(recognition_time)
        self.debug_stats['recognition_times'] = self.debug_stats['recognition_times'][-100:]
                
        return best_identity, best_similarity
            
    def detect_mask_onnx_fixed(self, face_roi: np.ndarray) -> Tuple[bool, float]:
        """
        Fixed ONNX mask detection with enhanced debugging
        """
        if self.mask_session is None:
            print("❌ Mask session is None - skipping mask detection")
            return False, 0.0
            
        if face_roi.size == 0:
            print("❌ Empty face ROI - skipping mask detection")
            return False, 0.0
            
        start_time = time.time()
        try:
            # Preprocess image for ONNX model
            resized_face = cv2.resize(face_roi, (224, 224))
            rgb_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB)
            normalized_face = rgb_face.astype(np.float32) / 255.0
            
            # MODEL EXPECTS: [batch, 224, 224, 3] - NHWC format
            input_tensor = np.expand_dims(normalized_face, axis=0)
            
            # Enhanced debugging
            print(f"🔍 Mask detection - Input shape: {input_tensor.shape}, range: [{input_tensor.min():.3f}, {input_tensor.max():.3f}]")
            
            # Run inference
            outputs = self.mask_session.run(
                [self.mask_output_name], 
                {self.mask_input_name: input_tensor}
            )
            
            predictions = outputs[0]
            print(f"🔍 Mask detection - Output shape: {predictions.shape}, values: {predictions}")
            
            # Interpret predictions
            mask_confidence = self._interpret_predictions_simple(predictions)
            is_wearing_mask = mask_confidence > self.config.get('mask_confidence_threshold', 0.7)
            
            print(f"😷 Mask detection result: {is_wearing_mask} (confidence: {mask_confidence:.3f}, threshold: {self.config.get('mask_confidence_threshold', 0.7)})")
            
            # Update stats
            mask_time = (time.time() - start_time) * 1000
            self.debug_stats['mask_detection_times'].append(mask_time)
            self.debug_stats['mask_detection_times'] = self.debug_stats['mask_detection_times'][-100:]
            
            return is_wearing_mask, float(mask_confidence)
            
        except Exception as e:
            print(f"❌ ONNX mask detection error: {e}")
            import traceback
            traceback.print_exc()
            return False, 0.0
                
    def _interpret_predictions_simple(self, predictions: np.ndarray) -> float:
        """Interpret ONNX model predictions"""
        try:
            # Handle different output formats
            if predictions.shape[1] == 2:
                # [no_mask, mask] format - return mask probability
                return float(predictions[0][1])
            elif predictions.shape[1] == 1:
                # Single probability output
                return float(predictions[0][0])
            else:
                # Raw logits or other format
                if len(predictions.shape) == 2 and predictions.shape[1] > 2:
                    # Apply softmax if needed
                    exp_pred = np.exp(predictions - np.max(predictions))
                    softmax_pred = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)
                    return float(softmax_pred[0][1])  # Return mask probability
                else:
                    return float(predictions[0][0])
        except Exception as e:
            print(f"❌ Prediction interpretation error: {e}")
            return 0.0        
        
    def process_frame_with_mask(self, frame: np.ndarray) -> List[Dict]:
        """Enhanced pipeline with FIXED ONNX mask detection"""
        start_time = time.time()
        results = []
        
        # Detect faces (existing)
        detections = self.detect_faces(frame)
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            
            # Apply padding and extract ROI (existing)
            padding = self.config.get('roi_padding', 10)
            h, w = frame.shape[:2]
            x1_pad = max(0, x1 - padding)
            y1_pad = max(0, y1 - padding)
            x2_pad = min(w, x2 + padding)
            y2_pad = min(h, y2 + padding)
            
            face_roi = frame[y1_pad:y2_pad, x1_pad:x2_pad]
            
            if face_roi.size == 0:
                continue
                
            # FIXED ONNX MASK DETECTION
            is_wearing_mask = False
            mask_confidence = 0.0
            if self.mask_session is not None:
                is_wearing_mask, mask_confidence = self.detect_mask_onnx_fixed(face_roi)
            
            identity = None
            recognition_confidence = 0.0
            embedding = None
            
            # Only run recognition if no mask detected
            if not is_wearing_mask:
                embedding = self.extract_embedding(face_roi)
                if embedding is not None:
                    identity, recognition_confidence = self.recognize_face(embedding)
            else:
                # Count masked faces
                self.debug_stats['total_masked_faces'] += 1
        
            results.append({
                'bbox': detection['bbox'],
                'detection_confidence': detection['confidence'],
                'identity': identity,
                'recognition_confidence': recognition_confidence,
                'embedding': embedding.tolist() if embedding is not None else None,
                # MASK DETECTION RESULTS
                'is_wearing_mask': is_wearing_mask,
                'mask_confidence': mask_confidence
            })
        
        # Update stats
        self.debug_stats['total_frames_processed'] += 1
        self.debug_stats['total_faces_detected'] += len(detections)
        self.debug_stats['total_faces_recognized'] += len([r for r in results if r['identity']])
        self.debug_stats['last_processing_time'] = (time.time() - start_time) * 1000
            
        return results
   
    def detect_mask_fixed(self, face_roi: np.ndarray) -> Tuple[bool, float]:
        """Fixed mask detection using NHWC format"""
        if self.mask_session is None or face_roi.size == 0:
            return False, 0.0
        
        try:
            # Preprocess - resize to 224x224 and convert to RGB
            resized_face = cv2.resize(face_roi, (224, 224))
            rgb_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB)
            normalized_face = rgb_face.astype(np.float32) / 255.0
            
            # NHWC format: [1, 224, 224, 3] - NO TRANSPOSE
            input_tensor = np.expand_dims(normalized_face, axis=0)
            
            # Run inference
            outputs = self.mask_session.run(
                [self.mask_output_name], 
                {self.mask_input_name: input_tensor}
            )
            
            predictions = outputs[0]
            
            # Interpret predictions
            if predictions.shape[1] == 2:
                mask_confidence = predictions[0][1]  # [no_mask, mask] format
            else:
                mask_confidence = predictions[0][0]  # single probability
            
            is_wearing_mask = mask_confidence > self.config.get('mask_confidence_threshold', 0.7)
            return is_wearing_mask, float(mask_confidence)
            
        except Exception as e:
            print(f"❌ Mask detection error: {e}")
            return False, 0.0    

    def process_frame(self, frame: np.ndarray) -> List[Dict]:
        """Main processing with fixed mask detection"""
        start_time = time.time()
        results = []
        detections = self.detect_faces(frame)
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            padding = self.config.get('roi_padding', 10)
            h, w = frame.shape[:2]
            
            x1_pad = max(0, x1 - padding)
            y1_pad = max(0, y1 - padding)
            x2_pad = min(w, x2 + padding)
            y2_pad = min(h, y2 + padding)
            
            face_roi = frame[y1_pad:y2_pad, x1_pad:x2_pad]
            
            if face_roi.size == 0:
                continue
                
            # Mask detection
            is_wearing_mask, mask_confidence = self.detect_mask_fixed(face_roi)
            
            identity = None
            recognition_confidence = 0.0
            embedding = None
            
            # Only recognize if no mask
            if not is_wearing_mask:
                embedding = self.extract_embedding(face_roi)
                if embedding is not None:
                    identity, recognition_confidence = self.recognize_face(embedding)
            else:
                self.debug_stats['total_masked_faces'] += 1
        
            results.append({
                'bbox': detection['bbox'],
                'detection_confidence': detection['confidence'],
                'identity': identity,
                'recognition_confidence': recognition_confidence,
                'embedding': embedding.tolist() if embedding is not None else None,
                'is_wearing_mask': is_wearing_mask,
                'mask_confidence': mask_confidence
            })
    
        # Update stats
        self.debug_stats['total_frames_processed'] += 1
        self.debug_stats['total_faces_detected'] += len(detections)
        self.debug_stats['total_faces_recognized'] += len([r for r in results if r['identity']])
        self.debug_stats['last_processing_time'] = (time.time() - start_time) * 1000
            
        return results

    def get_debug_stats(self) -> Dict:
        """Enhanced performance statistics with mask detection"""
        stats = self.debug_stats.copy()
        
        # Existing calculations...
        stats['avg_detection_time'] = np.mean(stats['detection_times']) if stats['detection_times'] else 0
        stats['p95_detection_time'] = np.percentile(stats['detection_times'], 95) if stats['detection_times'] else 0
        stats['avg_embedding_time'] = np.mean(stats['embedding_times']) if stats['embedding_times'] else 0
        stats['avg_recognition_time'] = np.mean(stats['recognition_times']) if stats['recognition_times'] else 0
        
        # Mask detection stats
        if stats['mask_detection_times']:
            stats['avg_mask_detection_time'] = np.mean(stats['mask_detection_times'])
            stats['p95_mask_detection_time'] = np.percentile(stats['mask_detection_times'], 95)
        else:
            stats['avg_mask_detection_time'] = 0
            stats['p95_mask_detection_time'] = 0
        
        # Calculate rates and efficiencies
        if stats['total_faces_detected'] > 0:
            stats['recognition_rate'] = (stats['total_faces_recognized'] / stats['total_faces_detected']) * 100
            stats['mask_rate'] = (stats['total_masked_faces'] / stats['total_faces_detected']) * 100
        else:
            stats['recognition_rate'] = 0
            stats['mask_rate'] = 0
            
        if stats['total_frames_processed'] > 0:
            stats['faces_per_frame'] = stats['total_faces_detected'] / stats['total_frames_processed']
        else:
            stats['faces_per_frame'] = 0
        
        # Memory usage (approximate)
        try:
            import psutil
            process = psutil.Process()
            stats['memory_mb'] = process.memory_info().rss / 1024 / 1024
        except ImportError:
            stats['memory_mb'] = 0
        
        return stats
    
    def get_known_identities(self) -> List[str]:
        """Get list of all known identities"""
        return list(self.identity_centroids.keys())

class DisplayResizer:
    """Handles multiple resizing strategies for output display"""
    
    def __init__(self):
        self.current_scale = 1.0
        self.resize_method = "fit_to_screen"
        self.target_width = 1280
        self.target_height = 720
        self.maintain_aspect_ratio = True
        self.max_display_size = (1920, 1080)
        
    def resize_frame(self, frame: np.ndarray, method: str = None, 
                    target_size: Tuple[int, int] = None, 
                    scale: float = None) -> np.ndarray:
        if method:
            self.resize_method = method
            
        if target_size:
            self.target_width, self.target_height = target_size
            
        if scale:
            self.current_scale = scale
            
        if self.resize_method == "fit_to_screen":
            return self._fit_to_screen(frame)
        elif self.resize_method == "fixed_size":
            return self._resize_fixed(frame, self.target_width, self.target_height)
        elif self.resize_method == "scale":
            return self._resize_scale(frame, self.current_scale)
        elif self.resize_method == "crop":
            return self._resize_crop(frame, self.target_width, self.target_height)
        elif self.resize_method == "letterbox":
            return self._resize_letterbox(frame, self.target_width, self.target_height)
        else:
            return frame
    
    def _fit_to_screen(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        max_w, max_h = self.max_display_size
        
        scale_w = max_w / w
        scale_h = max_h / h
        scale = min(scale_w, scale_h, 1.0)
        
        if scale == 1.0:
            return frame
            
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    def _resize_fixed(self, frame: np.ndarray, width: int, height: int) -> np.ndarray:
        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    
    def _resize_scale(self, frame: np.ndarray, scale: float) -> np.ndarray:
        if scale == 1.0:
            return frame
            
        h, w = frame.shape[:2]
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    def _resize_crop(self, frame: np.ndarray, width: int, height: int) -> np.ndarray:
        h, w = frame.shape[:2]
        
        target_aspect = width / height
        original_aspect = w / h
        
        if original_aspect > target_aspect:
            new_w = int(h * target_aspect)
            start_x = (w - new_w) // 2
            cropped = frame[:, start_x:start_x + new_w]
        else:
            new_h = int(w / target_aspect)
            start_y = (h - new_h) // 2
            cropped = frame[start_y:start_y + new_h, :]
        
        return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_AREA)
    
    def _resize_letterbox(self, frame: np.ndarray, width: int, height: int) -> np.ndarray:
        h, w = frame.shape[:2]
        
        scale = min(width / w, height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        result = np.zeros((height, width, 3), dtype=np.uint8)
        
        pad_x = (width - new_w) // 2
        pad_y = (height - new_h) // 2
        
        result[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        
        return result
    
    def get_resize_info(self) -> Dict:
        return {
            'method': self.resize_method,
            'target_size': (self.target_width, self.target_height),
            'current_scale': self.current_scale,
            'maintain_aspect_ratio': self.maintain_aspect_ratio,
            'max_display_size': self.max_display_size
        }
           
class RealTimeProcessor:
    def __init__(self, face_system, processing_interval: int = 5, buffer_size: int = 10):
        self.face_system = face_system
        self.cap = None
        self.fps = 0
        self.frame_count = 0
        self.processing_count = 0
        self.start_time = time.time()
        
        # Frame processing optimization
        self.processing_interval = processing_interval
        self.last_processed_time = 0
        self.min_processing_delay = 0.1
        
        # Threading for RTSP stability
        self.frame_queue = Queue(maxsize=buffer_size)
        self.latest_frame = None
        self.frame_lock = Lock()
        self.running = False
        self.capture_thread = None
        
        # RTSP configuration
        self.rtsp_url = None
        self.reconnect_delay = 5
        self.max_reconnect_attempts = 5
        
        # Enhanced display resizing - NOW APPLIED TO INPUT STREAM
        self.resizer = DisplayResizer()
        self.show_resize_info = False
        self.original_frame_size = None
        
        # Processing resolution - resize input stream for processing
        self.processing_width = 1000  # Default processing width
        self.processing_height = 500  # Default processing height
        self.processing_scale = 1.0   # Scale factor for processing
        
        # Debug controls
        self.debug_mode = False
        self.show_detection_debug = False
        self.show_performance_stats = True
        self.save_debug_frames = False
        self.debug_frame_count = 0
        self.max_debug_frames = 100
        
        # Dynamic Resolution Adjustment System
        self.dynamic_adjustment_enabled = True
        self.adaptive_check_interval = 30  # Check every 30 frames
        self.performance_history = []  # Track performance metrics
        self.max_history_size = 50
        
        # Resolution adjustment parameters
        self.min_processing_scale = 0.5   # Minimum scale (50% of original)
        self.max_processing_scale = 1.5   # Maximum scale (150% of original)
        self.current_processing_scale = 1.0
        self.scale_adjustment_step = 0.1
        
        # Performance thresholds
        self.target_detection_rate = 0.7   # Aim for 70% detection rate
        self.target_face_size = 80         # Target face size in pixels
        self.min_face_size = 40            # Minimum acceptable face size
        
        # Detection quality tracking
        self.consecutive_poor_detections = 0
        self.consecutive_good_detections = 0
        self.adjustment_cooldown = 0
        
        print("🎯 Dynamic resolution adjustment ENABLED")
            
        # Enhanced control attributes
        self.face_tracking_enabled = False
        self.logging_enabled = False
        self.current_preset_index = 0
        
        print("🎮 Enhanced keyboard controls LOADED")        
        
        # Initialize tracker
        self.face_tracker = SimpleFaceTracker(confidence_frames=15, cooldown_seconds=11)        
        
    def set_processing_resolution(self, width: int, height: int):
        """Set the resolution for processing (face detection/recognition)"""
        self.processing_width = width
        self.processing_height = height
        print(f"⚙️  Processing resolution set to {width}x{height}")
    
    def set_processing_scale(self, scale: float):
        """Set scale factor for processing resolution"""
        self.processing_scale = scale
        print(f"⚙️  Processing scale set to {scale:.2f}")
    
    def analyze_detection_performance(self, results: List[Dict], processing_frame_shape: Tuple[int, int]) -> Dict:
        """Comprehensive analysis of detection performance for dynamic adjustment"""
        performance = {
            'detection_count': len(results),
            'face_sizes': [],
            'detection_confidences': [],
            'recognition_rates': [],
            'avg_face_size': 0,
            'detection_quality': 0,
            'needs_adjustment': False,
            'adjustment_direction': 0  # -1: decrease, 0: maintain, +1: increase
        }
        
        if not results:
            performance['detection_quality'] = 0
            performance['needs_adjustment'] = True
            performance['adjustment_direction'] = 1  # Increase resolution if no detections
            return performance
        
        # Analyze each detection
        for result in results:
            x1, y1, x2, y2 = result['bbox']
            face_width = x2 - x1
            face_height = y2 - y1
            face_size = min(face_width, face_height)  # Use smallest dimension
            
            performance['face_sizes'].append(face_size)
            performance['detection_confidences'].append(result['detection_confidence'])
            
            if result['identity']:
                performance['recognition_rates'].append(result['recognition_confidence'])
        
        # Calculate metrics
        if performance['face_sizes']:
            performance['avg_face_size'] = np.mean(performance['face_sizes'])
            performance['detection_quality'] = self.calculate_detection_quality(performance)
            performance['needs_adjustment'] = self.should_adjust_resolution(performance)
            performance['adjustment_direction'] = self.get_adjustment_direction(performance)
        
        return performance
        
    def scale_bbox_to_display(self, bbox: List[int], original_shape: Tuple[int, int], display_shape: Tuple[int, int]) -> List[int]:
        """Scale bounding box coordinates from original frame to display frame"""
        x1, y1, x2, y2 = bbox
        orig_h, orig_w = original_shape
        disp_h, disp_w = display_shape
        
        scale_x = disp_w / orig_w
        scale_y = disp_h / orig_h
        
        return [
            int(x1 * scale_x),
            int(y1 * scale_y), 
            int(x2 * scale_x),
            int(y2 * scale_y)
        ]    
    

    def calculate_detection_quality(self, performance: Dict) -> float:
        """Calculate overall detection quality score (0-1)"""
        quality_factors = []
        
        # Face size factor (normalized to target)
        if performance['avg_face_size'] > 0:
            size_factor = min(performance['avg_face_size'] / self.target_face_size, 1.0)
            quality_factors.append(size_factor * 0.6)  # 60% weight
        
        # Detection confidence factor
        if performance['detection_confidences']:
            conf_factor = np.mean(performance['detection_confidences'])
            quality_factors.append(conf_factor * 0.7)  # 70% weight
        
        # Recognition rate factor (if applicable)
        if performance['recognition_rates']:
            recog_factor = np.mean(performance['recognition_rates'])
            quality_factors.append(recog_factor * 0.5)  # 50% weight
        else:
            # If no recognitions but detections exist, use medium weight
            quality_factors.append(0.15)
        
        return sum(quality_factors)

    def should_adjust_resolution(self, performance: Dict) -> bool:
        """Determine if resolution adjustment is needed"""
        # Always adjust if no detections
        if performance['detection_count'] == 0:
            return True
        
        # Check face size thresholds
        if performance['avg_face_size'] < self.min_face_size:
            return True
        
        # Check detection quality
        if performance['detection_quality'] < self.target_detection_rate:
            return True
        
        # Check if we're in cooldown period
        if self.adjustment_cooldown > 0:
            return False
        
        return False

    def get_adjustment_direction(self, performance: Dict) -> int:
        """Determine which direction to adjust resolution"""
        if performance['detection_count'] == 0:
            return 1  # Increase resolution if no detections
        
        if performance['avg_face_size'] < self.min_face_size:
            return 1  # Increase resolution for small faces
        
        if performance['detection_quality'] < self.target_detection_rate:
            return 1  # Increase resolution for poor quality
        
        # If quality is good and faces are large, consider decreasing resolution
        if (performance['detection_quality'] > self.target_detection_rate + 0.2 and 
            performance['avg_face_size'] > self.target_face_size + 20):
            return -1  # Decrease resolution
        
        return 0  # Maintain current resolution
    
    
    def resize_for_processing(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame for processing (face detection/recognition)"""
        if self.processing_scale != 1.0:
            # Scale-based resizing
            h, w = frame.shape[:2]
            new_w = int(w * self.processing_scale)
            new_h = int(h * self.processing_scale)
            return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            # Fixed size resizing
            return cv2.resize(frame, (self.processing_width, self.processing_height), interpolation=cv2.INTER_AREA)
    
    def scale_bbox_to_original(self, bbox: List[int], original_shape: Tuple[int, int], processed_shape: Tuple[int, int]) -> List[int]:
        """Scale bounding box coordinates from processed frame back to original frame"""
        x1, y1, x2, y2 = bbox
        orig_h, orig_w = original_shape
        proc_h, proc_w = processed_shape
        
        scale_x = orig_w / proc_w
        scale_y = orig_h / proc_h
        
        return [
            int(x1 * scale_x),
            int(y1 * scale_y),
            int(x2 * scale_x),
            int(y2 * scale_y)
        ]

    def initialize_stream(self, source: str):
        """Initialize camera or RTSP stream with optimized settings"""
        self.rtsp_url = source
        
        if source.startswith('rtsp://') or source.startswith('http://'):
            print(f"📹 Initializing RTSP stream: {source}")
            self._initialize_rtsp_stream(source)
        else:
            try:
                camera_id = int(source)
                print(f"📹 Initializing camera: {camera_id}")
                self._initialize_camera(camera_id)
            except ValueError:
                print(f"📹 Initializing video source: {source}")
                self._initialize_video_source(source)

    def _initialize_camera(self, camera_id: int):
        """Initialize local camera"""
        self.cap = cv2.VideoCapture(camera_id)
        # Set camera to highest resolution for best processing
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_id}")

    def _initialize_rtsp_stream(self, rtsp_url: str):
        """Initialize RTSP stream with optimized parameters"""
        optimized_rtsp = self._optimize_rtsp_url(rtsp_url)
        self.cap = cv2.VideoCapture(optimized_rtsp)
        
        # Set RTSP properties for better stability
        self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
        self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open RTSP stream: {rtsp_url}")

    def _initialize_video_source(self, source: str):
        """Initialize video file or other source"""
        self.cap = cv2.VideoCapture(source)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

    def _optimize_rtsp_url(self, rtsp_url: str) -> str:
        """Add optimization parameters to RTSP URL"""
        if '?' in rtsp_url:
            return rtsp_url + '&tcp=True&buffer_size=65535'
        else:
            return rtsp_url + '?tcp=True&buffer_size=65535'
        
    def start_frame_capture(self):
        """Start background thread for frame capture"""
        if self.cap is None:
            raise RuntimeError("Stream not initialized. Call initialize_stream first.")
        
        self.running = True
        self.capture_thread = Thread(target=self._capture_frames, daemon=True)
        self.capture_thread.start()
        print("🎬 Frame capture thread started")
        
    def apply_dynamic_adjustment(self, performance: Dict):
        """Apply resolution adjustment based on performance analysis"""
        if not self.dynamic_adjustment_enabled or self.adjustment_cooldown > 0:
            return
        
        direction = performance['adjustment_direction']
        
        if direction == 0:
            self.consecutive_good_detections += 1
            self.consecutive_poor_detections = 0
            return
        
        # Track consecutive adjustments
        if direction == 1:  # Need to increase resolution
            self.consecutive_poor_detections += 1
            self.consecutive_good_detections = 0
        else:  # Need to decrease resolution
            self.consecutive_good_detections += 1
            self.consecutive_poor_detections = 0
        
        # Calculate new scale with momentum
        momentum = self.calculate_adjustment_momentum()
        new_scale = self.current_processing_scale + (direction * self.scale_adjustment_step * momentum)
        
        # Apply bounds
        new_scale = max(self.min_processing_scale, min(self.max_processing_scale, new_scale))
        
        # Only adjust if change is significant
        if abs(new_scale - self.current_processing_scale) >= self.scale_adjustment_step * 0.5:
            old_scale = self.current_processing_scale
            self.current_processing_scale = new_scale
            self.adjustment_cooldown = 10  # Cooldown period
            
            # Log the adjustment
            direction_symbol = "🔼" if direction > 0 else "🔽"
            reason = self.get_adjustment_reason(performance, direction)
            print(f"{direction_symbol} Dynamic adjustment: {old_scale:.2f} → {new_scale:.2f} | {reason}")

    def calculate_adjustment_momentum(self) -> float:
        """Calculate adjustment momentum based on consecutive performance"""
        if self.consecutive_poor_detections > 3:
            return 2.0  # Double step size for persistent issues
        elif self.consecutive_poor_detections > 1:
            return 1.5  # 50% larger step
        elif self.consecutive_good_detections > 5:
            return 0.5  # Smaller step when optimizing from good state
        else:
            return 1.0  # Normal step

    def get_adjustment_reason(self, performance: Dict, direction: int) -> str:
        """Generate human-readable reason for adjustment"""
        if performance['detection_count'] == 0:
            return "No faces detected"
        elif performance['avg_face_size'] < self.min_face_size:
            return f"Faces too small ({performance['avg_face_size']:.0f}px < {self.min_face_size}px)"
        elif performance['detection_quality'] < self.target_detection_rate:
            return f"Poor detection quality ({performance['detection_quality']:.2f})"
        else:
            return f"Optimizing performance ({performance['detection_quality']:.2f})"     
            
    def enhanced_resize_for_processing(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame for processing using dynamic scale"""
        if self.current_processing_scale == 1.0:
            return frame
            
        h, w = frame.shape[:2]
        new_w = int(w * self.current_processing_scale)
        new_h = int(h * self.current_processing_scale)
        
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def update_dynamic_system(self):
        """Update dynamic adjustment system state"""
        if self.adjustment_cooldown > 0:
            self.adjustment_cooldown -= 1
        
        # Trim performance history
        if len(self.performance_history) > self.max_history_size:
            self.performance_history.pop(0)           

    def _capture_frames(self):
        """Background thread for continuous frame capture"""
        reconnect_attempts = 0
        
        while self.running:
            try:
                ret, frame = self.cap.read()
                
                if not ret:
                    print("⚠️  Frame capture failed, attempting to reconnect...")
                    reconnect_attempts += 1
                    
                    if reconnect_attempts >= self.max_reconnect_attempts:
                        print("❌ Max reconnection attempts reached")
                        break
                    
                    time.sleep(self.reconnect_delay)
                    self._reconnect_stream()
                    continue
                
                reconnect_attempts = 0
                
                with self.frame_lock:
                    self.latest_frame = frame
                
                if not self.frame_queue.full():
                    self.frame_queue.put(frame, block=False)
                else:
                    try:
                        self.frame_queue.get(block=False)
                        self.frame_queue.put(frame, block=False)
                    except:
                        pass
                        
            except Exception as e:
                print(f"🚨 Capture thread error: {e}")
                time.sleep(1)

    def _reconnect_stream(self):
        """Enhanced stream reconnection with exponential backoff"""
        max_attempts = 10
        base_delay = 2
        
        for attempt in range(max_attempts):
            try:
                if self.cap:
                    self.cap.release()
                
                # Exponential backoff
                delay = base_delay * (2 ** attempt)
                print(f"🔄 Reconnection attempt {attempt + 1}/{max_attempts}, waiting {delay}s...")
                time.sleep(min(delay, 30))  # Cap at 30 seconds
                
                if self.rtsp_url:
                    self._initialize_rtsp_stream(self.rtsp_url)
                    
                    # Test connection
                    ret, test_frame = self.cap.read()
                    if ret and test_frame is not None:
                        print("✅ Stream reconnected successfully!")
                        return True
                else:
                    print("❌ No RTSP URL available for reconnection")
                    return False
                    
            except Exception as e:
                print(f"❌ Reconnection attempt {attempt + 1} failed: {e}")
        
        print("🚨 Maximum reconnection attempts reached")
        return False

            
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get the latest frame from the capture thread"""
        with self.frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None  

    def should_process_frame(self) -> bool:
        """Adaptive frame processing based on system load"""
        current_time = time.time()
        
        # Base interval check
        if self.frame_count % self.processing_interval != 0:
            return False
        
        # Timing protection
        if current_time - self.last_processed_time < self.min_processing_delay:
            return False
        
        # Adaptive interval based on FPS
        if self.fps < 10:  # Low FPS - process fewer frames
            adaptive_interval = max(1, self.processing_interval + 2)
            if self.frame_count % adaptive_interval != 0:
                return False
        elif self.fps > 30:  # High FPS - can process more frames
            adaptive_interval = max(1, self.processing_interval - 1)
            if self.frame_count % adaptive_interval != 0:
                return False
        
        self.last_processed_time = current_time
        return True
        
    def calculate_fps(self):
        """Calculate and update FPS"""
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            elapsed = time.time() - self.start_time
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.start_time = time.time()
            
    def cycle_processing_preset(self):
        """Cycle through different processing presets"""
        presets = [
            {"name": "SPEED", "interval": 10, "scale": 0.6, "width": 640, "height": 480},
            {"name": "BALANCED", "interval": 5, "scale": 1.0, "width": 1280, "height": 720},
            {"name": "QUALITY", "interval": 2, "scale": 1.3, "width": 1600, "height": 900},
            {"name": "MAX QUALITY", "interval": 1, "scale": 1.5, "width": 1920, "height": 1080}
        ]
        
        self.current_preset_index = getattr(self, 'current_preset_index', -1) + 1
        if self.current_preset_index >= len(presets):
            self.current_preset_index = 0
            
        preset = presets[self.current_preset_index]
        
        self.processing_interval = preset["interval"]
        self.current_processing_scale = preset["scale"]
        self.processing_width = preset["width"]
        self.processing_height = preset["height"]
        
        print(f"🎛️  Preset: {preset['name']}")
        print(f"   - Interval: 1/{preset['interval']}")
        print(f"   - Scale: {preset['scale']:.1f}")
        print(f"   - Resolution: {preset['width']}x{preset['height']}")

    def toggle_face_tracking(self):
        """Toggle face tracking between frames (placeholder for implementation)"""
        self.face_tracking_enabled = not getattr(self, 'face_tracking_enabled', False)
        status = "ENABLED" if self.face_tracking_enabled else "DISABLED"
        print(f"👤 Face tracking: {status}")

    def toggle_logging(self):
        """Toggle performance logging to file"""
        self.logging_enabled = not getattr(self, 'logging_enabled', False)
        status = "ENABLED" if self.logging_enabled else "DISABLED"
        
        if self.logging_enabled:
            self.setup_logging()
        
        print(f"📝 Performance logging: {status}")

    def setup_logging(self):
        """Setup performance logging to CSV file"""
        import csv
        from datetime import datetime
        
        if not hasattr(self, 'log_file'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_file = f"performance_log_{timestamp}.csv"
            
            # Write header
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'fps', 'processing_interval', 'processing_scale',
                    'faces_detected', 'faces_recognized', 'detection_time_avg',
                    'embedding_time_avg', 'recognition_time_avg', 'memory_usage'
                ])
            print(f"📊 Logging to: {self.log_file}")

    def log_performance_data(self):
        """Log current performance data to file"""
        if getattr(self, 'logging_enabled', False):
            stats = self.face_system.get_debug_stats()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp, self.fps, self.processing_interval, 
                    self.current_processing_scale, stats['total_faces_detected'],
                    stats['total_faces_recognized'], stats['avg_detection_time'],
                    stats['avg_embedding_time'], stats['avg_recognition_time'],
                    stats.get('memory_mb', 0)
                ])

    def take_annotated_snapshot(self, frame: np.ndarray):
        """Take snapshot with overlay information"""
        timestamp = int(time.time())
        filename = f"snapshot_{timestamp}.jpg"
        
        # Create annotated frame
        annotated_frame = frame.copy()
        
        # Add timestamp and system info
        cv2.putText(annotated_frame, f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"FPS: {self.fps:.1f}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Scale: {self.current_processing_scale:.2f}", 
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Save frame
        cv2.imwrite(filename, annotated_frame)
        print(f"📸 Annotated snapshot saved: {filename}")            
            
                           
                                
        
    def set_display_size(self, width: int, height: int, method: str = "fixed_size"):
        """Set fixed display size"""
        self.resizer.target_width = width
        self.resizer.target_height = height
        self.resizer.resize_method = method
        print(f"🖼️  Display size set to {width}x{height} using {method} method")
    
    def set_display_scale(self, scale: float):
        """Set display scale factor"""
        self.resizer.current_scale = scale
        self.resizer.resize_method = "scale"
        print(f"🔍 Display scale set to {scale:.2f}")
    
    def set_display_method(self, method: str):
        """Set resizing method"""
        valid_methods = ["fit_to_screen", "fixed_size", "scale", "crop", "letterbox"]
        if method in valid_methods:
            self.resizer.resize_method = method
            print(f"🔄 Resize method set to: {method}")
        else:
            print(f"❌ Invalid resize method. Choose from: {valid_methods}")
    
    def set_max_display_size(self, width: int, height: int):
        """Set maximum display size for fit_to_screen method"""
        self.resizer.max_display_size = (width, height)
        print(f"📏 Maximum display size set to {width}x{height}")
    
    def toggle_resize_info(self):
        """Toggle resize information display"""
        self.show_resize_info = not self.show_resize_info
        status = "ON" if self.show_resize_info else "OFF"
        print(f"📊 Resize info display: {status}")

    # Debug control methods
    def toggle_debug_mode(self):
        """Toggle comprehensive debug mode"""
        self.debug_mode = not self.debug_mode
        status = "ON" if self.debug_mode else "OFF"
        print(f"🐛 Debug mode: {status}")
        
    def toggle_detection_debug(self):
        """Toggle detection visualization debug"""
        self.show_detection_debug = not self.show_detection_debug
        status = "ON" if self.show_detection_debug else "OFF"
        print(f"🎯 Detection debug: {status}")
        
    def toggle_performance_stats(self):
        """Toggle performance statistics display"""
        self.show_performance_stats = not self.show_performance_stats
        status = "ON" if self.show_performance_stats else "OFF"
        print(f"📈 Performance stats: {status}")
        
    def toggle_save_debug_frames(self):
        """Toggle saving debug frames"""
        self.save_debug_frames = not self.save_debug_frames
        status = "ON" if self.save_debug_frames else "OFF"
        print(f"💾 Save debug frames: {status}")
        
    def print_detailed_stats(self):
        """Print detailed system statistics"""
        stats = self.face_system.get_debug_stats()
        print("\n" + "="*50)
        print("📊 DETAILED SYSTEM STATISTICS")
        print("="*50)
        print(f"Total Frames Processed: {stats['total_frames_processed']}")
        print(f"Total Faces Detected: {stats['total_faces_detected']}")
        print(f"Total Faces Recognized: {stats['total_faces_recognized']}")
        print(f"Recognition Rate: {stats['recognition_rate']:.1f}%")
        print(f"Last Processing Time: {stats['last_processing_time']:.1f}ms")
        print(f"Avg Detection Time: {stats['avg_detection_time']:.1f}ms")
        print(f"Avg Embedding Time: {stats['avg_embedding_time']:.1f}ms")
        print(f"Avg Recognition Time: {stats['avg_recognition_time']:.1f}ms")
        print(f"Current FPS: {self.fps:.1f}")
        print(f"Processing Interval: 1/{self.processing_interval}")
        print(f"Processing Resolution: {self.processing_width}x{self.processing_height}")
        print(f"Display Method: {self.resizer.resize_method}")
        print("="*50)
    
    def resize_frame_for_display(self, frame: np.ndarray) -> np.ndarray:
        """Apply resizing to frame for display"""
        if self.original_frame_size is None:
            self.original_frame_size = frame.shape[:2]
        
        return self.resizer.resize_frame(frame)
    
    def draw_resize_info(self, frame: np.ndarray):
        """Display resize information on frame"""
        if not self.show_resize_info:
            return
        
        original_h, original_w = self.original_frame_size or frame.shape[:2]
        display_h, display_w = frame.shape[:2]
        
        info_lines = [
            f"Original: {original_w}x{original_h}",
            f"Display: {display_w}x{display_h}",
            f"Method: {self.resizer.resize_method}",
            f"Scale: {self.resizer.current_scale:.2f}" if self.resizer.resize_method == "scale" else "",
            f"Processing: {self.processing_width}x{self.processing_height}"
        ]
        
        info_lines = [line for line in info_lines if line.strip()]
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 130), (350, 130 + len(info_lines) * 25 + 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        for i, line in enumerate(info_lines):
            y_position = 150 + (i * 25)
            cv2.putText(frame, line, (20, y_position),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def draw_debug_info(self, frame: np.ndarray, results: List[Dict]):
        """Enhanced debug info with mask statistics"""
        if not self.debug_mode and not self.show_performance_stats:
            return
            
        stats = self.face_system.get_debug_stats()
        
        # Performance metrics
        performance_lines = []
        if self.show_performance_stats:
            performance_lines = [
                f"FPS: {self.fps:.1f}",
                f"Frame: {self.frame_count}",
                f"Processed: {self.processing_count}",
                f"Interval: 1/{self.processing_interval}",
                f"Recognition: {stats['recognition_rate']:.1f}%",
                f"Masked: {stats.get('mask_rate', 0):.1f}%",  # NEW
            ]
        
        # Debug information
        debug_lines = []
        if self.debug_mode:
            debug_lines = [
                f"Detection: {stats['avg_detection_time']:.1f}ms",
                f"Embedding: {stats['avg_embedding_time']:.1f}ms",
                f"Recognition: {stats['avg_recognition_time']:.1f}ms",
                f"Mask Detect: {stats.get('avg_mask_detection_time', 0):.1f}ms",  # NEW
                f"Total Faces: {stats['total_faces_detected']}",
                f"Recognized: {stats['total_faces_recognized']}",
                f"Masked: {stats.get('total_masked_faces', 0)}",  # NEW
            ]
        
        all_lines = performance_lines + debug_lines
        if not all_lines:
            return
            
        # Draw background for all info
        overlay = frame.copy()
        start_y = 10
        end_y = start_y + len(all_lines) * 25 + 20
        cv2.rectangle(overlay, (10, start_y), (350, end_y), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw performance stats (green)
        for i, line in enumerate(performance_lines):
            y_position = 30 + (i * 25)
            cv2.putText(frame, line, (20, y_position),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw debug info (cyan)
        debug_start_y = 30 + len(performance_lines) * 25
        for i, line in enumerate(debug_lines):
            y_position = debug_start_y + (i * 25)
            cv2.putText(frame, line, (20, y_position),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
               
    def draw_detection_debug(self, frame: np.ndarray, results: List[Dict]):
        """Draw detailed detection debugging information"""
        if not self.show_detection_debug:
            return
            
        for i, result in enumerate(results):
            x1, y1, x2, y2 = result['bbox']
            
            # Draw detailed information near each detection
            info_text = f"Det: {result['detection_confidence']:.2f}"
            if result['identity']:
                info_text += f" | Rec: {result['identity']} ({result['recognition_confidence']:.2f})"
            
            # Calculate position for debug text (below the bounding box)
            text_y = y2 + 20
            
            # Draw background for text
            text_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(frame, (x1, text_y - text_size[1] - 5), 
                         (x1 + text_size[0], text_y + 5), (0, 0, 0), -1)
            
            # Draw debug text
            cv2.putText(frame, info_text, (x1, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def draw_results(self, frame: np.ndarray, results: List[Dict]):
        """Enhanced visualization with debug support"""
        if self.original_frame_size is None:
            self.original_frame_size = frame.shape[:2]
        
        # Draw bounding boxes and labels
        for result in results:
            x1, y1, x2, y2 = result['bbox']
            identity = result['identity']
            rec_conf = result['recognition_confidence']
            det_conf = result['detection_confidence']
            
            if identity:
                color = (0, 255, 0)
                label = f"{identity} ({rec_conf:.2f})"
            else:
                color = (0, 0, 255)
                label = f"Unknown ({det_conf:.2f})"
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw debug information
        self.draw_debug_info(frame, results)
        self.draw_detection_debug(frame, results)
        
        # Draw resize information if enabled
        self.draw_resize_info(frame)
    
    def save_debug_frame(self, frame: np.ndarray, results: List[Dict]):
        """Save frame with debug information"""
        if not self.save_debug_frames or self.debug_frame_count >= self.max_debug_frames:
            return
            
        debug_frame = frame.copy()
        
        # Add timestamp and frame info
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        info_text = f"Frame_{self.debug_frame_count:04d}_{timestamp}"
        cv2.putText(debug_frame, info_text, (10, debug_frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Save frame
        filename = f"debug_frame_{self.debug_frame_count:04d}.jpg"
        cv2.imwrite(filename, debug_frame)
        self.debug_frame_count += 1
        
        if self.debug_frame_count % 10 == 0:
            print(f"💾 Saved debug frame: {filename}")
                
    def draw_enhanced_results(self, frame: np.ndarray, results: List[Dict], performance: Dict):
        """Draw results with dynamic adjustment information"""
        # Existing drawing logic
        self.draw_results(frame, results)
        
        # Add dynamic adjustment info if available
        if performance and self.show_performance_stats:
            self.draw_dynamic_adjustment_info(frame, performance)
                
    def draw_enhanced_results_with_mask(self, frame: np.ndarray, results: List[Dict], performance: Dict):
        """Draw results with mask status visualization and ONNX debug info"""
        if self.original_frame_size is None:
            self.original_frame_size = frame.shape[:2]
        
        # Draw bounding boxes and labels with mask status
        for result in results:
            x1, y1, x2, y2 = result['bbox']
            identity = result['identity']
            rec_conf = result['recognition_confidence']
            det_conf = result['detection_confidence']
            is_wearing_mask = result.get('is_wearing_mask', False)
            mask_conf = result.get('mask_confidence', 0.0)
            
            # COLOR CODING BASED ON MASK STATUS
            if is_wearing_mask:
                color = (255, 255, 0)  # CYAN for masked faces
                if identity:
                    label = f"{identity} MASKED ({mask_conf:.2f})"
                else:
                    label = f"Masked ({mask_conf:.2f})"
            elif identity:
                color = (0, 255, 0)  # GREEN for recognized unmasked
                label = f"{identity} ({rec_conf:.2f})"
            else:
                color = (0, 0, 255)  # RED for unknown unmasked
                label = f"Unknown ({det_conf:.2f})"
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                        (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # ADD MASK STATUS INDICATOR
            status_text = "MASK" if is_wearing_mask else "NO MASK"
            status_color = (255, 255, 0) if is_wearing_mask else (0, 255, 255)
            cv2.putText(frame, status_text, (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
            
            # DEBUG: Show mask confidence for all faces
            if self.debug_mode:
                conf_text = f"Mask Conf: {mask_conf:.2f}"
                cv2.putText(frame, conf_text, (x1, y2 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Existing debug and performance drawing
        self.draw_debug_info(frame, results)
        self.draw_detection_debug(frame, results)
        self.draw_resize_info(frame)
        
        if performance and self.show_performance_stats:
            self.draw_dynamic_adjustment_info(frame, performance)
            
        # NEW: Show ONNX model status
        self.draw_onnx_status(frame)

    def draw_onnx_status(self, frame: np.ndarray):
        """Display ONNX model status on frame"""
        if not self.debug_mode:
            return
            
        status = "ONNX: " + ("✅ LOADED" if self.face_system.mask_session is not None else "❌ NOT LOADED")
        cv2.putText(frame, status, (frame.shape[1] - 200, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
    def draw_dynamic_adjustment_info(self, frame: np.ndarray, performance: Dict):
        """Display dynamic adjustment metrics"""
        info_lines = [
            f"Dynamic Scale: {self.current_processing_scale:.2f}",
            f"Faces: {performance.get('detection_count', 0)}",
            f"Avg Size: {performance.get('avg_face_size', 0):.0f}px",
            f"Quality: {performance.get('detection_quality', 0):.2f}",
        ]
        
        if performance.get('needs_adjustment', False):
            direction = performance.get('adjustment_direction', 0)
            if direction > 0:
                info_lines.append("Status: NEEDS INCREASE ↗")
            elif direction < 0:
                info_lines.append("Status: CAN DECREASE ↘")
            else:
                info_lines.append("Status: OPTIMAL ✓")
        
        # Draw background
        overlay = frame.copy()
        start_y = frame.shape[0] - len(info_lines) * 25 - 20
        end_y = frame.shape[0] - 10
        cv2.rectangle(overlay, (10, start_y), (300, end_y), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw text
        for i, line in enumerate(info_lines):
            y_position = start_y + 20 + (i * 20)
            color = (0, 255, 255) if "NEEDS" in line else (255, 255, 255)
            cv2.putText(frame, line, (20, y_position),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
    def handle_key_controls(self, key: int, display_frame: np.ndarray = None):
        """Comprehensive keyboard controls for all system features"""
        if key == ord('q'):
            self.running = False
            print("🛑 Quitting application...")
            
        elif key == ord('s'):
            # Save current frame
            timestamp = int(time.time())
            filename = f'captured_frame_{timestamp}.jpg'
            cv2.imwrite(filename, display_frame)
            print(f"💾 Frame saved: {filename}")
            
        elif key == ord('+'):  # Increase processing interval (process fewer frames)
            old_interval = self.processing_interval
            self.processing_interval = min(self.processing_interval + 1, 30)
            print(f"⏱️  Processing interval: 1/{old_interval} → 1/{self.processing_interval}")
            
        elif key == ord('-'):  # Decrease processing interval (process more frames)
            old_interval = self.processing_interval
            self.processing_interval = max(self.processing_interval - 1, 1)
            print(f"⏱️  Processing interval: 1/{old_interval} → 1/{self.processing_interval}")
            
        elif key == ord('r'):  # Reset processing counters
            self.frame_count = 0
            self.processing_count = 0
            self.start_time = time.time()
            print("🔄 Processing counters reset")
            
        elif key == ord('i'):  # Toggle resize info display
            self.toggle_resize_info()
            
        elif key == ord('d'):  # Toggle debug mode
            self.toggle_debug_mode()
            
        elif key == ord('p'):  # Toggle performance stats
            self.toggle_performance_stats()
            
        elif key == ord('b'):  # Toggle detection debug
            self.toggle_detection_debug()
            
        elif key == ord('f'):  # Toggle save debug frames
            self.toggle_save_debug_frames()
            
        elif key == ord('x'):  # Print detailed statistics
            self.print_detailed_stats()
            
        elif key == ord('w'):  # Decrease processing resolution
            old_w, old_h = self.processing_width, self.processing_height
            self.processing_width = max(320, self.processing_width - 160)
            self.processing_height = max(240, self.processing_height - 120)
            print(f"📐 Processing resolution: {old_w}x{old_h} → {self.processing_width}x{self.processing_height}")
            
        elif key == ord('e'):  # Increase processing resolution
            old_w, old_h = self.processing_width, self.processing_height
            self.processing_width = min(1920, self.processing_width + 160)
            self.processing_height = min(1080, self.processing_height + 120)
            print(f"📐 Processing resolution: {old_w}x{old_h} → {self.processing_width}x{self.processing_height}")
            
        elif key == ord('a'):  # Toggle dynamic adjustment
            self.dynamic_adjustment_enabled = not self.dynamic_adjustment_enabled
            status = "ENABLED" if self.dynamic_adjustment_enabled else "DISABLED"
            print(f"🎯 Dynamic adjustment: {status}")
            
        elif key == ord('z'):  # Reset dynamic scaling
            old_scale = self.current_processing_scale
            self.current_processing_scale = 1.0
            self.performance_history = []
            self.consecutive_poor_detections = 0
            self.consecutive_good_detections = 0
            print(f"🔄 Dynamic scaling reset: {old_scale:.2f} → 1.00")
            
        elif key == ord('c'):  # Force increase processing scale
            old_scale = self.current_processing_scale
            self.current_processing_scale = min(self.max_processing_scale, 
                                            self.current_processing_scale + 0.2)
            print(f"🔼 Manual scale increase: {old_scale:.2f} → {self.current_processing_scale:.2f}")
            
        elif key == ord('v'):  # Force decrease processing scale
            old_scale = self.current_processing_scale
            self.current_processing_scale = max(self.min_processing_scale, 
                                            self.current_processing_scale - 0.2)
            print(f"🔽 Manual scale decrease: {old_scale:.2f} → {self.current_processing_scale:.2f}")
            
        elif key == ord('n'):  # Toggle between fixed and dynamic processing
            if self.processing_scale == 1.0:  # Currently using fixed resolution
                self.processing_scale = 0.0  # Switch to dynamic scaling
                print("🎯 Switched to DYNAMIC processing scale")
            else:  # Currently using dynamic scaling
                self.processing_scale = 1.0  # Switch to fixed resolution
                print("📐 Switched to FIXED processing resolution")
                
        elif key == ord('m'):  # Cycle through processing presets
            self.cycle_processing_preset()
            
        elif key == ord('t'):  # Toggle face tracking (if implemented)
            self.toggle_face_tracking()
            
        elif key == ord('l'):  # Toggle logging to file
            self.toggle_logging()
            
        elif key == ord('k'):  # Take snapshot with metadata
            self.take_annotated_snapshot(display_frame)
            
        # Display resize methods (1-8, 0)
        elif key == ord('1'):
            self.set_display_method("fit_to_screen")
        elif key == ord('2'):
            self.set_display_size(1280, 720, "fixed_size")
        elif key == ord('3'):
            self.set_display_scale(0.5)
        elif key == ord('4'):
            self.set_display_scale(0.75)
        elif key == ord('5'):
            self.set_display_scale(1.0)
        elif key == ord('6'): 
            self.set_display_scale(1.5)
        elif key == ord('7'):
            self.set_display_size(1280, 720, "crop")
        elif key == ord('8'):
            self.set_display_size(1280, 720, "letterbox")
        elif key == ord('0'):
            self.set_display_method("fit_to_screen")
            self.set_max_display_size(3840, 2160)
            print("📺 Displaying original size")
            
        # Number pad controls for fine-grained adjustments
        elif key == ord('.'):  # Fine increase processing interval
            old_interval = self.processing_interval
            self.processing_interval = min(self.processing_interval + 5, 60)
            print(f"⏱️  Processing interval: 1/{old_interval} → 1/{self.processing_interval}")
            
        elif key == ord(','):  # Fine decrease processing interval
            old_interval = self.processing_interval
            self.processing_interval = max(self.processing_interval - 5, 1)
            print(f"⏱️  Processing interval: 1/{old_interval} → 1/{self.processing_interval}")
        # MASK DETECTION CONTROL
        elif key == ord('u'):  # Toggle mask detection
            self.face_system.mask_detector_enabled = not self.face_system.mask_detector_enabled
            status = "ENABLED" if self.face_system.mask_detector_enabled else "DISABLED"
            print(f"😷 Mask detection: {status}")            
                
    def draw_tracking_info(self, frame: np.ndarray, results: List[Dict]):
        """Draw tracking status on frame"""
        for result in results:
            x1, y1, x2, y2 = result['bbox']
            identity = result['identity']
            track_state = result.get('track_state', 'NEW')
            track_id = result.get('track_id', 'N/A')
            
            # Color coding based on track state
            if track_state == 'COOLDOWN':
                color = (0, 255, 255)  # Yellow - trusted identity
                label = f"{identity} ✓"
            elif track_state == 'TRACKING':
                color = (0, 255, 0)    # Green - building confidence
                conf_count = next((t['confidence_count'] for t in self.face_tracker.active_tracks.values() 
                                if t.get('current_bbox') == result['bbox']), 0)
                label = f"{identity} ({conf_count}/{self.face_tracker.confidence_frames})"
            else:  # NEW
                color = (0, 0, 255)    # Red - new/unconfirmed
                label = f"{identity} ?" if identity else "Unknown"
            
            # Draw bounding box with track state
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with track info
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                        (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw track ID for debugging
            if self.debug_mode:
                cv2.putText(frame, f"Track: {track_id}", (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)            
                        
            
    def run(self, source: str = "0"):
        """Main loop with enhanced key controls"""
        try:
            self.initialize_stream(source)
            self.start_frame_capture()
            
            print("🎮 Starting with ENHANCED KEY CONTROLS")
            self.print_control_reference()
            
            last_results = []
            last_performance = {}
            
            while self.running:
                original_frame = self.get_current_frame()
                if original_frame is None:
                    time.sleep(0.01)
                    continue
                
                self.calculate_fps()
                self.update_dynamic_system()
                
                # Store original frame size
                original_h, original_w = original_frame.shape[:2]
                
                # Resize for processing using dynamic scale
                processing_frame = self.enhanced_resize_for_processing(original_frame)
                processed_h, processed_w = processing_frame.shape[:2]
                
                # Resize for display
                display_frame = self.resize_frame_for_display(original_frame)
                
                should_process = self.should_process_frame()
                results = last_results
                
                if should_process:
                    # Process on dynamically resized frame
                    raw_results = self.face_system.process_frame_with_mask(processing_frame)
                    processing_results = self.face_tracker.update(raw_results, self.frame_count)
                    
                    # Scale bounding boxes back to original frame
                    scaled_results = []
                    for result in processing_results:
                        scaled_bbox = self.scale_bbox_to_original(
                            result['bbox'], 
                            (original_h, original_w), 
                            (processed_h, processed_w)
                        )
                        scaled_result = result.copy()
                        scaled_result['bbox'] = scaled_bbox
                        scaled_results.append(scaled_result)
                    
                    last_results = scaled_results
                    self.processing_count += 1
                    
                    # Dynamic adjustment
                    if self.dynamic_adjustment_enabled and self.frame_count % self.adaptive_check_interval == 0:
                        performance = self.analyze_detection_performance(scaled_results, processing_frame.shape)
                        self.performance_history.append(performance)
                        last_performance = performance
                        self.apply_dynamic_adjustment(performance)
                    
                    # Log performance data if enabled
                    if getattr(self, 'logging_enabled', False):
                        self.log_performance_data()
                else:
                    scaled_results = last_results

                # ALWAYS apply display scaling to results (whether new or cached)
                display_h, display_w = display_frame.shape[:2]
                display_results = []
                for result in scaled_results:
                    display_bbox = self.scale_bbox_to_display(
                        result['bbox'],
                        (original_h, original_w),
                        (display_h, display_w)
                    )
                    display_result = result.copy()
                    display_result['bbox'] = display_bbox
                    display_results.append(display_result)

                results = display_results

                if self.save_debug_frames and should_process:
                    self.save_debug_frame(display_frame, results)
                    
                    # Dynamic adjustment
                    if self.dynamic_adjustment_enabled and self.frame_count % self.adaptive_check_interval == 0:
                        performance = self.analyze_detection_performance(results, processing_frame.shape)
                        self.performance_history.append(performance)
                        last_performance = performance
                        self.apply_dynamic_adjustment(performance)
                    
                    # Log performance data if enabled
                    if getattr(self, 'logging_enabled', False):
                        self.log_performance_data()
                    
                    if self.save_debug_frames:
                        self.save_debug_frame(display_frame, results)
                
                # Enhanced drawing
                self.draw_enhanced_results(display_frame, results, last_performance)
                cv2.imshow('Dynamic Face Recognition System', display_frame)
                
                # Handle ALL key controls in one place
                key = cv2.waitKey(1) & 0xFF
                self.handle_key_controls(key, display_frame)
                            
        except Exception as e:
            print(f"❌ Error in main loop: {e}")
        finally:
            self.stop()
                                
    def stop(self):
        """Cleanup resources"""
        self.running = False
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        # Print final statistics
        print("\n📊 FINAL STATISTICS:")
        self.print_detailed_stats()
        print("🛑 System stopped gracefully")
            
    def print_control_reference(self):
        """Print comprehensive control reference with mask controls"""
        print("\n" + "="*60)
        print("🎮 COMPREHENSIVE KEYBOARD CONTROLS")
        print("="*60)
        print("🎯 CORE CONTROLS:")
        print("  'q' - Quit application")
        print("  's' - Save current frame") 
        print("  'r' - Reset processing counters")
        print("  'x' - Print detailed statistics")
        
        print("\n⏱️  PROCESSING CONTROLS:")
        print("  '+' - Increase processing interval (process less)")
        print("  '-' - Decrease processing interval (process more)")
        print("  'w' - Decrease processing resolution")
        print("  'e' - Increase processing resolution")
        
        print("\n🎯 DYNAMIC ADJUSTMENT CONTROLS:")
        print("  'a' - Toggle dynamic adjustment")
        print("  'z' - Reset dynamic scaling to 1.0")
        print("  'c' - Manually increase processing scale")
        print("  'v' - Manually decrease processing scale")
        print("  'n' - Toggle fixed/dynamic processing")
        print("  'm' - Cycle processing presets")
        
        print("\n😷 MASK DETECTION CONTROLS:")  # NEW SECTION
        print("  'u' - Toggle mask detection")
        
        print("\n🖼️  DISPLAY CONTROLS:")
        print("  '1' - Fit to screen")
        print("  '2' - Fixed size (1280x720)")
        print("  '3' - Scale 0.5x")
        print("  '4' - Scale 0.75x")
        print("  '5' - Scale 1.0x")
        print("  '6' - Scale 1.5x")
        print("  '7' - Crop maintain aspect")
        print("  '8' - Letterbox maintain aspect")
        print("  '0' - Original size")
        print("  'i' - Toggle resize info")
        
        print("\n🐛 DEBUG CONTROLS:")
        print("  'd' - Toggle debug mode")
        print("  'p' - Toggle performance stats")
        print("  'b' - Toggle detection debug")
        print("  'f' - Toggle save debug frames")
        
        print("\n📊 ADVANCED CONTROLS:")
        print("  't' - Toggle face tracking")
        print("  'l' - Toggle performance logging")
        print("  'k' - Take annotated snapshot")
        
        print("="*60)
        print()
    
class SimpleFaceTracker:
    def __init__(self, confidence_frames=20, cooldown_seconds=20, min_iou=0.3):
        self.confidence_frames = confidence_frames
        self.cooldown_frames = cooldown_seconds * 30  # Convert to frames
        self.min_iou = min_iou
        self.active_tracks = {}
        self.next_track_id = 0
        
    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection area
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        intersection = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def _find_best_match(self, bbox, current_tracks):
        """Find best matching track for a detection"""
        best_match_id = None
        best_iou = self.min_iou
        
        for track_id, track in current_tracks.items():
            iou = self._calculate_iou(bbox, track['current_bbox'])
            if iou > best_iou:
                best_iou = iou
                best_match_id = track_id
        
        return best_match_id
    
    def _create_track(self, result, frame_count):
        """Enhanced track creation with mask status"""
        return {
            'identity': result['identity'],
            'recognition_confidence': result['recognition_confidence'],
            'detection_confidence': result['detection_confidence'],
            'current_bbox': result['bbox'],
            'confidence_count': 1 if result['identity'] else 0,
            'first_detected_frame': frame_count,
            'last_updated_frame': frame_count,
            'cooldown_counter': 0,
            'track_state': 'TRACKING',
            # NEW: Mask status tracking
            'is_wearing_mask': result.get('is_wearing_mask', False),
            'mask_confidence': result.get('mask_confidence', 0.0)
        }
    
    def _update_track(self, track, new_result, frame_count):
        """Enhanced track update with mask status"""
        updated_track = track.copy()
        updated_track['current_bbox'] = new_result['bbox']
        updated_track['last_updated_frame'] = frame_count
        updated_track['is_wearing_mask'] = new_result.get('is_wearing_mask', False)
        updated_track['mask_confidence'] = new_result.get('mask_confidence', 0.0)
        
        # Only update identity if no mask and good recognition
        if (not new_result.get('is_wearing_mask', False) and 
            new_result['identity'] and 
            new_result['recognition_confidence'] > 0.6):
            
            if new_result['identity'] == track['identity']:
                # Same identity - increase confidence
                updated_track['confidence_count'] += 1
                updated_track['recognition_confidence'] = new_result['recognition_confidence']
            else:
                # Different identity - reset if more confident
                if new_result['recognition_confidence'] > track['recognition_confidence']:
                    updated_track['identity'] = new_result['identity']
                    updated_track['recognition_confidence'] = new_result['recognition_confidence']
                    updated_track['confidence_count'] = 1
        
        # Check if we should enter cooldown mode
        if (updated_track['confidence_count'] >= self.confidence_frames and 
            updated_track['track_state'] == 'TRACKING'):
            updated_track['track_state'] = 'COOLDOWN'
            updated_track['cooldown_counter'] = self.cooldown_frames
        
        return updated_track
    
    def _update_cooldowns(self, updated_tracks, frame_count):
        """Update cooldown counters and handle state transitions"""
        for track_id, track in list(updated_tracks.items()):
            if track['track_state'] == 'COOLDOWN':
                track['cooldown_counter'] -= 1
                
                # Reset to tracking when cooldown ends
                if track['cooldown_counter'] <= 0:
                    track['track_state'] = 'TRACKING'
                    track['confidence_count'] = 0  # Reset for new recognition cycle
    
    def _get_final_results(self, original_results):
        """Generate final results with tracking overrides"""
        final_results = []
        
        for result in original_results:
            # Find if this result matches any track
            matched_track_id = self._find_best_match(result['bbox'], self.active_tracks)
            
            if matched_track_id and self.active_tracks[matched_track_id]['track_state'] == 'COOLDOWN':
                # Use track identity during cooldown
                track = self.active_tracks[matched_track_id]
                final_result = result.copy()
                final_result['identity'] = track['identity']
                final_result['recognition_confidence'] = track['recognition_confidence']
                final_result['track_id'] = matched_track_id
                final_result['track_state'] = 'COOLDOWN'
            else:
                # Use original recognition
                final_result = result.copy()
                if matched_track_id:
                    final_result['track_id'] = matched_track_id
                    final_result['track_state'] = self.active_tracks[matched_track_id]['track_state']
                else:
                    final_result['track_state'] = 'NEW'
            
            final_results.append(final_result)
        
        return final_results
    
    def update(self, recognition_results, frame_count):
        """Main update method"""
        if not recognition_results:
            # No detections - update cooldowns on existing tracks
            self._update_cooldowns(self.active_tracks, frame_count)
            return []
        
        # Update existing tracks and create new ones
        updated_tracks = {}
        
        for result in recognition_results:
            matched_track_id = self._find_best_match(result['bbox'], self.active_tracks)
            
            if matched_track_id is not None:
                # Update existing track
                track = self._update_track(self.active_tracks[matched_track_id], result, frame_count)
                updated_tracks[matched_track_id] = track
            else:
                # Create new track
                track_id = self.next_track_id
                self.next_track_id += 1
                new_track = self._create_track(result, frame_count)
                updated_tracks[track_id] = new_track
        
        # Update cooldowns and handle state transitions
        self._update_cooldowns(updated_tracks, frame_count)
        
        # Remove tracks that haven't been updated (missed detections)
        current_tracks = {}
        for track_id, track in updated_tracks.items():
            # Keep tracks for a few frames even if not detected
            if frame_count - track['last_updated_frame'] <= 5:  # 5 frame tolerance
                current_tracks[track_id] = track
        
        self.active_tracks = current_tracks
        
        return self._get_final_results(recognition_results)

def validate_onnx_model(self):
    """Validate the ONNX model with a test inference"""
    if self.mask_session is None:
        print("❌ No ONNX model loaded for validation")
        return False
    
    try:
        # Create a dummy test image
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        print("🧪 Validating ONNX model with test inference...")
        is_wearing_mask, confidence, debug_info = self.detect_mask_onnx_advanced(test_image)
        
        print(f"✅ ONNX Model Validation Successful:")
        print(f"   - Input shape: {debug_info['input_shape']}")
        print(f"   - Output shape: {debug_info['output_shape']}")
        print(f"   - Preprocessing time: {debug_info['preprocessing_time']:.1f}ms")
        print(f"   - Inference time: {debug_info['inference_time']:.1f}ms")
        print(f"   - Test result: mask={is_wearing_mask}, confidence={confidence:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ ONNX Model Validation Failed: {e}")
        return False

def print_onnx_model_info(self):
    """Print detailed information about the ONNX mask detection model"""
    if self.mask_session is None:
        print("❌ No ONNX mask detection model loaded")
        return
    
    print("\n🔍 ONNX Mask Detection Model Information:")
    print("=" * 50)
    
    # Input information
    inputs = self.mask_session.get_inputs()
    for i, input in enumerate(inputs):
        print(f"Input {i}:")
        print(f"  Name: {input.name}")
        print(f"  Shape: {input.shape}")
        print(f"  Type: {input.type}")
    
    # Output information  
    outputs = self.mask_session.get_outputs()
    for i, output in enumerate(outputs):
        print(f"Output {i}:")
        print(f"  Name: {output.name}")
        print(f"  Shape: {output.shape}")
        print(f"  Type: {output.type}")
    
    # Available providers
    providers = self.mask_session.get_providers()
    print(f"Available providers: {providers}")
    print(f"Current provider: {self.mask_session.get_provider_options()}")
    print("=" * 50)
                                     
# Enhanced configuration with ONNX mask detection
CONFIG = {
    'detection_model_path': r'D:\SCMA\3-APD\fromAraya\Computer-Vision-CV\3.1_FaceRecog\yolov11n-face.pt',
    'embeddings_db_path': r'D:\SCMA\3-APD\fromAraya\Computer-Vision-CV\3.1_FaceRecog\\person_folder1.4.json',
    'detection_confidence': 0.6,
    'detection_iou': 0.5,
    'roi_padding': 10,
    'embedding_model': 'Facenet',
    'recognition_threshold': 0.5,
    
    # ONNX MASK DETECTION CONFIG
    'mask_model_path': r'D:\SCMA\3-APD\fromAraya\Computer-Vision-CV\3.1_FaceRecog\run_py\mask_detector2.onnx',
    'mask_confidence_threshold': 0.4,
    'enable_mask_detection': True,
}

TRACKING_CONFIG = {
    'confidence_frames': 20,
    'cooldown_seconds': 20,
    'min_iou': 0.3
}

def validate_config(config: Dict) -> bool:
    """Validate configuration parameters"""
    required_keys = ['detection_model_path', 'embeddings_db_path', 'detection_confidence']
    
    for key in required_keys:
        if key not in config:
            print(f"❌ Missing required config key: {key}")
            return False
    
    if config['detection_confidence'] < 0 or config['detection_confidence'] > 1:
        print("❌ Detection confidence must be between 0 and 1")
        return False
        
    if config['recognition_threshold'] < 0 or config['recognition_threshold'] > 1:
        print("❌ Recognition threshold must be between 0 and 1")
        return False
    
    return True

def main():
    # Initialize system
    face_system = FaceRecognitionSystem(CONFIG)
    
    # Create processor with optimization
    processor = RealTimeProcessor(
        face_system=face_system,
        processing_interval=5,
        buffer_size=5
    )
    
        # Choose your input source
    sources = {
        '1': '0',                          # Default camera
        '2': 'rtsp://admin:Admin888@192.168.0.2:554/Streaming/Channels/101',  # RTSP
        '3': 'http://192.168.1.101:8080/video',                   # IP camera
        '4': 'video.mp4'                   # Video file
    }
    
    print("Available sources:")
    for key, source in sources.items():
        print(f"  {key}: {source}")
    
    choice = input("Select source (1-4) or enter custom RTSP URL: ").strip()
    
    if choice in sources:
        source = sources[choice]
    else:
        source = choice  # Custom input
    
    # Configure display
    processor.set_display_size(1024, 768, "fixed_size")
    
    try:
        processor.run(source)  # Use default camera
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        processor.stop()    

if __name__ == "__main__":
    main()