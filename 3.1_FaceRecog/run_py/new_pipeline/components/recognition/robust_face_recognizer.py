# components/recognition/robust_face_recognizer.py
from .face_recognizer import FaceRecognizer
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import time
from collections import deque
import cv2

class RobustFaceRecognizer(FaceRecognizer):
    """
    Enhanced face recognizer with multi-scale processing, quality assessment,
    and temporal fusion for improved robustness
    """
    
    def __init__(self, name: str, config: Dict):
        super().__init__(name, config)
        
        # Robustness features
        self.enable_multi_scale = config.get('enable_multi_scale', True)
        self.enable_quality_assessment = config.get('enable_quality_aware', True)
        self.enable_temporal_fusion = config.get('enable_temporal_fusion', True)
        
        # Multi-scale processing
        self.scale_factors = [0.5, 0.75, 1.0, 1.25, 1.5]
        self.rotation_angles = [-10, -5, 0, 5, 10]
        
        # Temporal fusion
        self.temporal_buffer = {}
        self.buffer_size = config.get('temporal_buffer_size', 10)
        
        # Quality assessment
        self.min_face_quality = config.get('min_face_quality', 0.3)
        
        print("🎯 RobustFaceRecognizer initialized with enhanced features")
    
    def _extract_embedding(self, face_roi: np.ndarray) -> Optional[np.ndarray]:
        """Enhanced embedding extraction with multi-scale processing"""
        if self.enable_multi_scale:
            return self._extract_multi_scale_embedding(face_roi)
        else:
            return super()._extract_embedding(face_roi)
    
    def _extract_multi_scale_embedding(self, face_roi: np.ndarray) -> Optional[np.ndarray]:
        """Extract embeddings from multiple scales and rotations"""
        embeddings = []
        h, w = face_roi.shape[:2]
        
        for scale in self.scale_factors:
            # Skip if face becomes too small
            new_w, new_h = int(w * scale), int(h * scale)
            if new_w < 20 or new_h < 20:
                continue
                
            # Scale the face ROI
            scaled_face = cv2.resize(face_roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Extract embedding from scaled version
            embedding = self._extract_single_embedding(scaled_face)
            if embedding is not None:
                embeddings.append(embedding)
            
            # Add rotated versions for robustness
            for angle in self.rotation_angles:
                rotated_face = self._rotate_face(scaled_face, angle)
                rot_embedding = self._extract_single_embedding(rotated_face)
                if rot_embedding is not None:
                    embeddings.append(rot_embedding)
        
        # Fuse multiple embeddings
        if embeddings:
            return self._fuse_embeddings(embeddings)
        else:
            return None
    
    def _extract_single_embedding(self, face_roi: np.ndarray) -> Optional[np.ndarray]:
        """Extract single embedding (base method without multi-scale)"""
        return super()._extract_embedding(face_roi)
    
    def _rotate_face(self, face: np.ndarray, angle: float) -> np.ndarray:
        """Rotate face by small angle for robustness"""
        h, w = face.shape[:2]
        center = (w // 2, h // 2)
        
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(face, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC)
        
        return rotated
    
    def _fuse_embeddings(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """Fuse multiple embeddings into robust representation"""
        # Simple average fusion
        return np.mean(embeddings, axis=0)
    
    def _assess_face_quality(self, face_roi: np.ndarray) -> float:
        """Assess face quality for recognition confidence"""
        try:
            # Simple quality assessment based on sharpness and contrast
            if len(face_roi.shape) == 3:
                gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_roi
            
            # Sharpness (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness = min(1.0, laplacian_var / 500.0)
            
            # Contrast (standard deviation)
            contrast = np.std(gray) / 255.0
            
            # Combined quality score
            quality = 0.6 * sharpness + 0.4 * contrast
            
            return quality
            
        except Exception:
            return 0.5  # Default quality
    
    def _recognize_face(self, embedding: np.ndarray) -> Tuple[Optional[str], float, Dict]:
        """Enhanced recognition with temporal fusion and quality adaptation"""
        # Base recognition
        identity, confidence, similarity_scores = super()._recognize_face(embedding)
        
        # Generate track ID for temporal fusion (simple hash based on embedding)
        track_id = hash(tuple(embedding[:10])) % 1000000
        
        # Update temporal buffer
        if self.enable_temporal_fusion and identity:
            self._update_temporal_buffer(track_id, identity, confidence)
            
            # Get temporal consensus
            temporal_identity, temporal_confidence = self._get_temporal_consensus(track_id)
            
            if temporal_identity and temporal_confidence > confidence:
                identity = temporal_identity
                confidence = temporal_confidence
        
        # Add quality information
        similarity_scores['temporal_fusion'] = self.enable_temporal_fusion
        similarity_scores['multi_scale'] = self.enable_multi_scale
        
        return identity, confidence, similarity_scores
    
    def _update_temporal_buffer(self, track_id: int, identity: str, confidence: float):
        """Update temporal buffer with recent recognition results"""
        if track_id not in self.temporal_buffer:
            self.temporal_buffer[track_id] = deque(maxlen=self.buffer_size)
        
        self.temporal_buffer[track_id].append({
            'identity': identity,
            'confidence': confidence,
            'timestamp': time.time()
        })
    
    def _get_temporal_consensus(self, track_id: int) -> Tuple[Optional[str], float]:
        """Get consensus identity from temporal buffer"""
        if track_id not in self.temporal_buffer or not self.temporal_buffer[track_id]:
            return None, 0.0
        
        buffer = self.temporal_buffer[track_id]
        
        # Count occurrences of each identity
        identity_counts = {}
        identity_confidences = {}
        
        for recognition in buffer:
            identity = recognition['identity']
            confidence = recognition['confidence']
            
            if identity not in identity_counts:
                identity_counts[identity] = 0
                identity_confidences[identity] = []
            
            identity_counts[identity] += 1
            identity_confidences[identity].append(confidence)
        
        # Find identity with highest frequency and confidence
        best_identity = None
        best_score = 0.0
        
        for identity, count in identity_counts.items():
            avg_confidence = np.mean(identity_confidences[identity])
            frequency = count / len(buffer)
            
            # Combined score: frequency * confidence
            combined_score = frequency * avg_confidence
            
            if combined_score > best_score and combined_score > self.similarity_threshold:
                best_score = combined_score
                best_identity = identity
        
        return best_identity, best_score
    
    def get_robust_stats(self) -> Dict[str, Any]:
        """Get robust recognition statistics"""
        stats = self.get_recognition_stats()
        stats.update({
            'temporal_tracks': len(self.temporal_buffer),
            'multi_scale_enabled': self.enable_multi_scale,
            'temporal_fusion_enabled': self.enable_temporal_fusion,
            'quality_assessment_enabled': self.enable_quality_assessment,
            'buffer_size': self.buffer_size
        })
        
        # Temporal buffer statistics
        if self.temporal_buffer:
            buffer_sizes = [len(buffer) for buffer in self.temporal_buffer.values()]
            stats['temporal_stats'] = {
                'active_tracks': len(self.temporal_buffer),
                'avg_buffer_size': np.mean(buffer_sizes),
                'max_buffer_size': max(buffer_sizes)
            }
        
        return stats