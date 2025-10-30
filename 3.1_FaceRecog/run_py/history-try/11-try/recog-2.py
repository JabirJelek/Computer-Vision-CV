import cv2
import json
import numpy as np
import threading
from pathlib import Path
from ultralytics import YOLO
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
from threading import Thread, Lock
from queue import Queue
from collections import deque
import time
from typing import Dict, List, Tuple, Optional
import datetime
import csv
import onnxruntime as ort
import requests
from urllib.parse import quote
from scipy.spatial.distance import mahalanobis
from sklearn.metrics.pairwise import paired_distances

import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional
import time

import cv2
import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional

class ConfigManager:
    """Unified configuration manager for hierarchical configuration management"""
    
    def __init__(self, base_config: Dict):
        self.base = base_config.copy()  # Deep copy to prevent mutations
        self.derived = self._build_derived_configs()
        self.validation_rules = self._build_validation_rules()
        self.config_history = deque(maxlen=10)  # Track config changes
        
        # Validate base config
        self._validate_config(self.base)
        
        print("✅ ConfigManager initialized with unified configuration system")
    
    def _build_derived_configs(self) -> Dict[str, Dict]:
        """Build all derived configurations from base"""
        derived_configs = {}
        
        # Robust Face Recognition Config
        derived_configs['robust_face_recognition'] = {
            **self.base,
            # Robustness enhancements
            'enable_multi_scale': True,
            'enable_temporal_fusion': True, 
            'enable_quality_aware': True,
            'enable_quality_adaptive_similarity': True,
            'min_face_quality': 0.3,
            'temporal_buffer_size': 10,
            
            # Multi-scale processing
            'scale_factors': [0.5, 0.75, 1.0, 1.25, 1.5],
            'rotation_angles': [-10, -5, 0, 5, 10],
            
            # Quality assessment weights
            'quality_weights': {
                'sharpness': 0.3,
                'brightness': 0.2, 
                'contrast': 0.15,
                'size': 0.2,
                'position': 0.1,
                'blur': 0.05
            },
            
            # Enhanced similarity configuration
            'similarity_method': 'quality_adaptive',
            'similarity_weights': {
                'cosine': 0.25,
                'angular': 0.40,
                'pearson': 0.15,
                'dot_product': 0.15,
                'euclidean': 0.10,
                'manhattan': 0.15,
                'jaccard': 0.20
            },
        }
        
        # Context-Aware Processing Config
        derived_configs['context_aware_processing'] = {
            **derived_configs['robust_face_recognition'],
            
            # Context-aware scaling parameters
            'min_processing_scale': 0.3,
            'max_processing_scale': 2.5,
            'scale_adjustment_step': 0.1,
            'context_weight': 0.4,
            'performance_weight': 0.6,
            
            # Enable context features
            'enable_context_aware_scaling': True,
        }
        
        # High Performance Config
        derived_configs['high_performance'] = {
            **self.base,
            'processing_interval': 10,
            'current_processing_scale': 0.6,
            'processing_width': 640,
            'processing_height': 480,
            'enable_multi_scale': False,
            'enable_temporal_fusion': False,
            'similarity_method': 'balanced',
        }
        
        # High Accuracy Config
        derived_configs['high_accuracy'] = {
            **derived_configs['robust_face_recognition'],
            'processing_interval': 2,
            'current_processing_scale': 1.5,
            'processing_width': 1920,
            'processing_height': 1080,
            'detection_confidence': 0.7,
            'recognition_threshold': 0.7,
        }
        
        # Balanced Config
        derived_configs['balanced'] = {
            **derived_configs['robust_face_recognition'],
            'processing_interval': 5,
            'current_processing_scale': 1.0,
            'processing_width': 1280,
            'processing_height': 720,
        }
        
        return derived_configs
    
    def _build_validation_rules(self) -> Dict[str, any]:
        """Build configuration validation rules"""
        return {
            'detection_confidence': {'min': 0.0, 'max': 1.0, 'type': float},
            'recognition_threshold': {'min': 0.0, 'max': 1.0, 'type': float},
            'mask_detection_threshold': {'min': 0.0, 'max': 1.0, 'type': float},
            'processing_interval': {'min': 1, 'max': 60, 'type': int},
            'min_processing_scale': {'min': 0.1, 'max': 1.0, 'type': float},
            'max_processing_scale': {'min': 1.0, 'max': 5.0, 'type': float},
            'min_face_quality': {'min': 0.0, 'max': 1.0, 'type': float},
            'temporal_buffer_size': {'min': 1, 'max': 50, 'type': int},
        }
    
    def _validate_config(self, config: Dict) -> bool:
        """Validate configuration against rules"""
        errors = []
        
        for key, rules in self.validation_rules.items():
            if key in config:
                value = config[key]
                # Type check
                if not isinstance(value, rules['type']):
                    errors.append(f"{key}: expected {rules['type']}, got {type(value)}")
                # Range check
                elif 'min' in rules and value < rules['min']:
                    errors.append(f"{key}: value {value} below minimum {rules['min']}")
                elif 'max' in rules and value > rules['max']:
                    errors.append(f"{key}: value {value} above maximum {rules['max']}")
        
        if errors:
            print("❌ Configuration validation errors:")
            for error in errors:
                print(f"   - {error}")
            return False
        
        return True
    
    def get_component_config(self, component_name: str) -> Dict:
        """Get configuration for specific component"""
        if component_name in self.derived:
            config = self.derived[component_name]
            # Log config usage
            self.config_history.append({
                'component': component_name,
                'timestamp': time.time(),
                'config_keys': list(config.keys())[:5]  # First 5 keys for logging
            })
            return config
        else:
            print(f"⚠️  Config for '{component_name}' not found, using base config")
            return self.base
    
    def get_available_configs(self) -> List[str]:
        """Get list of available configuration profiles"""
        return ['base'] + list(self.derived.keys())
    
    def create_custom_config(self, profile_name: str, base_profile: str, overrides: Dict) -> bool:
        """Create a custom configuration profile"""
        if profile_name in self.derived:
            print(f"❌ Config profile '{profile_name}' already exists")
            return False
        
        # Get base profile
        if base_profile == 'base':
            base_config = self.base
        elif base_profile in self.derived:
            base_config = self.derived[base_profile]
        else:
            print(f"❌ Base profile '{base_profile}' not found")
            return False
        
        # Apply overrides
        custom_config = {**base_config, **overrides}
        
        # Validate
        if not self._validate_config(custom_config):
            return False
        
        # Store
        self.derived[profile_name] = custom_config
        print(f"✅ Created custom config profile: '{profile_name}'")
        return True
    
    def update_config_value(self, profile_name: str, key: str, value: any) -> bool:
        """Update a specific configuration value"""
        if profile_name == 'base':
            config = self.base
        elif profile_name in self.derived:
            config = self.derived[profile_name]
        else:
            print(f"❌ Config profile '{profile_name}' not found")
            return False
        
        # Validate the update
        if key in self.validation_rules:
            rules = self.validation_rules[key]
            if not isinstance(value, rules['type']):
                print(f"❌ Invalid type for {key}: expected {rules['type']}")
                return False
            if 'min' in rules and value < rules['min']:
                print(f"❌ Value for {key} below minimum {rules['min']}")
                return False
            if 'max' in rules and value > rules['max']:
                print(f"❌ Value for {key} above maximum {rules['max']}")
                return False
        
        # Apply update
        old_value = config.get(key)
        config[key] = value
        print(f"🔄 Updated {profile_name}.{key}: {old_value} → {value}")
        return True
    
    def get_config_info(self, profile_name: str = None) -> Dict:
        """Get information about configuration profiles"""
        if profile_name:
            if profile_name == 'base':
                config = self.base
            elif profile_name in self.derived:
                config = self.derived[profile_name]
            else:
                return {'error': f"Profile '{profile_name}' not found"}
            
            return {
                'profile_name': profile_name,
                'key_count': len(config),
                'sample_keys': list(config.keys())[:10],
                'validation_status': self._validate_config(config)
            }
        else:
            # Return info for all profiles
            info = {
                'base': {
                    'key_count': len(self.base),
                    'validation_status': self._validate_config(self.base)
                }
            }
            for name, config in self.derived.items():
                info[name] = {
                    'key_count': len(config),
                    'validation_status': self._validate_config(config)
                }
            return info
    
    def export_config(self, profile_name: str, filepath: str = None) -> bool:
        """Export configuration to JSON file"""
        if profile_name == 'base':
            config = self.base
        elif profile_name in self.derived:
            config = self.derived[profile_name]
        else:
            print(f"❌ Config profile '{profile_name}' not found")
            return False
        
        try:
            if filepath is None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = f"config_{profile_name}_{timestamp}.json"
            
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"✅ Exported config '{profile_name}' to {filepath}")
            return True
        except Exception as e:
            print(f"❌ Failed to export config: {e}")
            return False
    
    def import_config(self, filepath: str, profile_name: str = None) -> bool:
        """Import configuration from JSON file"""
        try:
            with open(filepath, 'r') as f:
                imported_config = json.load(f)
            
            # Determine profile name
            if profile_name is None:
                profile_name = Path(filepath).stem
            
            # Validate imported config
            if not self._validate_config(imported_config):
                return False
            
            # Store as custom profile
            self.derived[profile_name] = imported_config
            print(f"✅ Imported config as '{profile_name}' from {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to import config: {e}")
            return False

class DurationAwareAlertManager:
    def __init__(self, config: Dict):
        self.config = config
        self.server_url = config.get('alert_server_url')
        self.cooldown_seconds = config.get('alert_cooldown_seconds', 30)
        self.enabled = config.get('enable_voice_alerts', True)
        self.last_alert_time = 0
        self.alert_lock = threading.Lock()
        
        # 🆕 Retry configuration
        self.max_retries = config.get('alert_max_retries', 3)
        self.retry_delay = config.get('alert_retry_delay', 2)  # seconds
        self.retry_timeout = config.get('alert_retry_timeout', 5)  # seconds per attempt
        
        # 🆕 Duration tracking parameters
        self.min_violation_frames = config.get('min_violation_frames', 20)
        self.min_violation_seconds = config.get('min_violation_seconds', 2.0)
        self.max_gap_frames = config.get('max_gap_frames', 5)
        
        # 🆕 Violation duration tracking per identity
        self.violation_timers = {}
        self.alerted_identities = set()
        
        print(f"🔊 Duration-aware alerts: {self.min_violation_frames} frames / {self.min_violation_seconds}s threshold")
        print(f"🔄 Retry configuration: {self.max_retries} attempts, {self.retry_delay}s delay")
        
    def update_violation_duration(self, violations: List[Dict], current_frame_count: int):
        """Update violation duration tracking for each identity"""
        current_time = time.time()
        current_violators = set()
        
        for violation in violations:
            identity = violation.get('identity', 'Unknown')
            current_violators.add(identity)
            
            # Start or update timer for this identity
            if identity not in self.violation_timers:
                self.violation_timers[identity] = {
                    'start_time': current_time,
                    'start_frame': current_frame_count,
                    'frame_count': 1,
                    'continuous': True,
                    'last_frame': current_frame_count
                }
            else:
                # Update existing timer
                timer = self.violation_timers[identity]
                
                # Check for continuity
                frame_gap = current_frame_count - timer['last_frame']
                if frame_gap <= self.max_gap_frames + 1:  # Allow small gaps
                    timer['frame_count'] += 1
                    timer['continuous'] = True
                else:
                    # Large gap detected - reset but keep identity
                    timer['start_time'] = current_time
                    timer['start_frame'] = current_frame_count
                    timer['frame_count'] = 1
                    timer['continuous'] = True
                    # Remove from alerted set since it's a new violation period
                    if identity in self.alerted_identities:
                        self.alerted_identities.remove(identity)
                
                timer['last_frame'] = current_frame_count
        
        # Remove identities that are no longer violating
        expired_identities = []
        for identity, timer in self.violation_timers.items():
            if identity not in current_violators:
                # Check if we should keep tracking for small gaps
                frame_gap = current_frame_count - timer['last_frame']
                if frame_gap > self.max_gap_frames:
                    expired_identities.append(identity)
                # else: keep in timers for gap tolerance
        
        for identity in expired_identities:
            self._reset_violation_timer(identity)

    def _reset_violation_timer(self, identity: str):
        """Reset violation timer for an identity"""
        if identity in self.violation_timers:
            del self.violation_timers[identity]
        if identity in self.alerted_identities:
            self.alerted_identities.remove(identity)

    def should_trigger_alert(self, identity: str) -> bool:
        """Check if violation duration meets threshold for alert"""
        if identity not in self.violation_timers:
            return False
            
        timer = self.violation_timers[identity]
        
        # Check frame count threshold
        frame_condition = timer['frame_count'] >= self.min_violation_frames
        
        # Check time duration threshold  
        time_condition = (time.time() - timer['start_time']) >= self.min_violation_seconds
        
        # Check if violation is continuous
        continuous_condition = timer.get('continuous', True)
        
        # Check if we haven't already alerted for this identity in current violation
        not_already_alerted = identity not in self.alerted_identities
        
        return frame_condition and time_condition and continuous_condition and not_already_alerted

    def get_violation_duration_info(self, identity: str) -> Dict[str, float]:
        """Get duration information for an identity"""
        if identity not in self.violation_timers:
            return {'frames': 0, 'seconds': 0.0}
        
        timer = self.violation_timers[identity]
        return {
            'frames': timer['frame_count'],
            'seconds': time.time() - timer['start_time']
        }

    def trigger_duration_alert(self, violations: List[Dict], current_frame_count: int) -> bool:
        """Main method to check and trigger duration-based alerts - RETURN STATUS"""
        if not self.enabled or not self.server_url:
            return False
        
        # Update violation duration tracking
        self.update_violation_duration(violations, current_frame_count)
        
        # Find identities that meet duration threshold
        alert_identities = []
        for violation in violations:
            identity = violation.get('identity', 'Unknown')
            if self.should_trigger_alert(identity):
                alert_identities.append(identity)
        
        if not alert_identities:
            return False
        
        # Generate and send alert
        success = self._send_violation_alert(alert_identities, violations)
        
        # Mark these identities as alerted
        if success:
            for identity in alert_identities:
                self.alerted_identities.add(identity)
                duration_info = self.get_violation_duration_info(identity)
                print(f"🚨 Duration alert triggered for {identity} "
                    f"({duration_info['frames']} frames, {duration_info['seconds']:.1f}s)")
        
        return success  # 🆕 Return whether alert was actually sent

    def _send_violation_alert(self, alert_identities: List[str], all_violations: List[Dict]) -> bool:
        """Send voice alert for violations that meet duration threshold"""
        # Filter violations to only those that triggered alerts
        alert_violations = [v for v in all_violations if v.get('identity') in alert_identities]
        
        if not alert_violations:
            return False
            
        # Count recognized vs unknown violators
        recognized = [v for v in alert_violations if v['identity'] and v['identity'] != 'Unknown']
        unknown = [v for v in alert_violations if not v['identity'] or v['identity'] == 'Unknown']
        
        if recognized:
            # Alert for recognized people by name
            names = [v['identity'] for v in recognized]
            if len(names) == 1:
                message = f"{names[0]} tidak memakai masker"
            else:
                name_list = " dan ".join(names)
                message = f"{name_list} tidak memakai masker"
        else:
            # Alert for unknown people
            count = len(unknown)
            if count == 1:
                message = "Satu orang tidak dikenal tidak memakai masker"
            else:
                message = f"{count} orang tidak dikenal tidak memakai masker"
        
        # # Add duration information for the first identity
        # if alert_identities:
        #     first_identity = alert_identities[0]
        #     duration_info = self.get_violation_duration_info(first_identity)
        #     message += f" selama {duration_info['seconds']:.1f} detik"
        
        # # Add urgency for multiple violations
        # total_violations = len(alert_violations)
        # if total_violations > 2:
        #     message += ". Situasi darurat!"
        
        # Send the alert
        return self.send_voice_alert(message)

    def send_voice_alert(self, message: str, identity: str = None, mask_status: str = None) -> bool:
        """Send voice alert to server in non-blocking thread (original method preserved)"""
        if not self.enabled or not self.server_url:
            return False
            
        # Rate limiting
        current_time = time.time()
        with self.alert_lock:
            if current_time - self.last_alert_time < self.cooldown_seconds:
                return False
            self.last_alert_time = current_time
        
        # Run in background thread to avoid blocking main processing
        thread = threading.Thread(
            target=self._send_alert_thread,
            args=(message, identity, mask_status),
            daemon=True
        )
        thread.start()
        return True
            
    def _log_audio_alert(self, message: str, status: str):
        """Log audio alerts to CSV file with retry information"""
        try:
            if not hasattr(self, 'audio_logging_enabled') or not self.audio_logging_enabled:
                return
                
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_data = {
                'timestamp': timestamp,
                'alert_message': message,
                'status': status,
                'server_url': self.server_url,
                'retry_config': f"{self.max_retries} attempts, {self.retry_delay}s delay"  # 🆕 Add retry info
            }
            
            # Append to the main log file if available
            if hasattr(self, 'log_file') and self.log_file:
                with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        log_data['timestamp'],
                        'AUDIO_ALERT',  # Special identifier for audio alerts
                        log_data['alert_message'],
                        log_data['status'],
                        log_data['server_url'],
                        log_data['retry_config']  # 🆕 Include retry info
                    ])
                    
        except Exception as e:
            print(f"❌ Failed to log audio alert: {e}") 

    def _send_alert_thread(self, message: str, identity: str, mask_status: str):
        """Background thread for sending alerts with RETRY LOGIC"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        for attempt in range(self.max_retries):
            try:
                # URL encode the message
                encoded_message = quote(message)
                alert_url = f"{self.server_url}?pesan={encoded_message}"
                
                # 🆕 Log attempt with retry info
                if attempt == 0:
                    print(f"🎵 [{timestamp}] Sending voice alert: {message}")
                else:
                    print(f"🔄 [{timestamp}] Retry {attempt}/{self.max_retries}: {message}")
                
                # Send HTTP request with timeout
                response = requests.get(alert_url, timeout=self.retry_timeout)
                
                if response.status_code == 200:
                    print(f"🔊 [{timestamp}] Voice alert sent successfully")
                    # 🆕 Log to CSV if audio logging is enabled
                    if hasattr(self, 'audio_logging_enabled') and self.audio_logging_enabled:
                        self._log_audio_alert(message, f"SUCCESS (attempt {attempt + 1})")
                    return  # Success - exit retry loop
                else:
                    error_msg = f"HTTP {response.status_code}"
                    print(f"❌ [{timestamp}] Attempt {attempt + 1} failed: {error_msg}")
                    
            except requests.exceptions.Timeout:
                error_msg = "Request timeout"
                print(f"⏰ [{timestamp}] Attempt {attempt + 1} failed: {error_msg}")
            except requests.exceptions.ConnectionError:
                error_msg = "Connection error"
                print(f"🔌 [{timestamp}] Attempt {attempt + 1} failed: {error_msg}")
            except requests.exceptions.RequestException as e:
                error_msg = f"Request error: {e}"
                print(f"❌ [{timestamp}] Attempt {attempt + 1} failed: {error_msg}")
            except Exception as e:
                error_msg = f"Unexpected error: {e}"
                print(f"🚨 [{timestamp}] Attempt {attempt + 1} failed: {error_msg}")
            
            # 🆕 If not the last attempt, wait before retrying
            if attempt < self.max_retries - 1:
                print(f"⏳ [{timestamp}] Retrying in {self.retry_delay} seconds...")
                time.sleep(self.retry_delay)
        
        # 🆕 If we get here, all attempts failed
        final_error = f"All {self.max_retries} attempts failed"
        print(f"🚨 [{timestamp}] {final_error} for alert: {message}")
        if hasattr(self, 'audio_logging_enabled') and self.audio_logging_enabled:
            self._log_audio_alert(message, f"FAILED: {final_error}")                
    def toggle_alerts(self):
        """Toggle voice alerts on/off"""
        self.enabled = not self.enabled
        status = "ENABLED" if self.enabled else "DISABLED"
        print(f"🔊 Voice alerts: {status}")
        return self.enabled

    def get_duration_stats(self) -> Dict:
        """Get current duration tracking statistics with real-time updates"""
        stats = {
            'total_tracked_identities': len(self.violation_timers),
            'alerted_identities': list(self.alerted_identities),
            'violation_timers': {}
        }
        
        current_time = time.time()
        for identity, timer in self.violation_timers.items():
            stats['violation_timers'][identity] = {
                'frame_count': timer['frame_count'],
                'duration_seconds': current_time - timer['start_time'],  # 🆕 Real-time duration
                'continuous': timer.get('continuous', True),
                'last_frame': timer['last_frame']
            }
        
        return stats

class SceneContextAnalyzer:
    """Advanced scene context analysis for intelligent scaling decisions"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.previous_frame = None
        self.motion_history = deque(maxlen=30)
        self.context_history = deque(maxlen=50)
        
        # Context thresholds
        self.thresholds = {
            'high_face_density': 0.05,  # faces per 1000 pixels
            'low_face_density': 0.005,
            'high_complexity': 0.6,
            'low_complexity': 0.2,
            'dark_lighting': 0.3,
            'bright_lighting': 0.7,
            'high_motion': 0.4,
            'low_motion': 0.1
        }
        
        print("🎯 Scene Context Analyzer initialized")

    def analyze_scene_context(self, frame: np.ndarray, detection_results: List[Dict]) -> Dict[str, float]:
        """Comprehensive scene context analysis"""
        if frame is None or frame.size == 0:
            return self._get_default_context()
        
        try:
            # Convert to grayscale for analysis
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            
            h, w = gray.shape
            
            context = {
                'face_density': self._calculate_face_density(detection_results, w, h),
                'scene_complexity': self._calculate_scene_complexity(gray),
                'lighting_conditions': self._assess_lighting_conditions(gray),
                'motion_level': self._estimate_motion_level(gray),
                'texture_density': self._calculate_texture_density(gray),
                'edge_concentration': self._calculate_edge_concentration(gray),
                'color_variance': self._calculate_color_variance(frame),
                'focus_quality': self._assess_focus_quality(gray)
            }
            
            # Store context for trend analysis
            self.context_history.append(context)
            
            return context
            
        except Exception as e:
            print(f"❌ Scene context analysis error: {e}")
            return self._get_default_context()

    def _calculate_face_density(self, results: List[Dict], frame_width: int, frame_height: int) -> float:
        """Calculate face density in the scene"""
        if not results:
            return 0.0
        
        total_face_area = 0
        for result in results:
            x1, y1, x2, y2 = result['bbox']
            face_area = (x2 - x1) * (y2 - y1)
            total_face_area += face_area
        
        frame_area = frame_width * frame_height
        density = total_face_area / frame_area if frame_area > 0 else 0.0
        
        return min(1.0, density * 10)  # Normalize to 0-1 range

    def _calculate_scene_complexity(self, gray_frame: np.ndarray) -> float:
        """Calculate scene complexity using edge density and variance"""
        try:
            # Edge detection
            edges = cv2.Canny(gray_frame, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Texture analysis using variance
            laplacian_var = cv2.Laplacian(gray_frame, cv2.CV_64F).var()
            texture_complexity = min(1.0, laplacian_var / 1000.0)
            
            # Combined complexity score
            complexity = 0.6 * edge_density + 0.4 * texture_complexity
            return min(1.0, complexity)
            
        except:
            return 0.5

    def _assess_lighting_conditions(self, gray_frame: np.ndarray) -> float:
        """Assess lighting conditions (0=dark, 1=bright)"""
        try:
            brightness = np.mean(gray_frame) / 255.0
            
            # Calculate contrast
            contrast = np.std(gray_frame) / 128.0  # Normalized
            
            # Combined lighting score (prioritize adequate lighting)
            if 0.3 <= brightness <= 0.7:
                lighting_score = 0.8 + (0.2 * contrast)  # Good lighting
            else:
                lighting_score = brightness * contrast  # Poor lighting
            
            return min(1.0, lighting_score)
            
        except:
            return 0.5

    def _estimate_motion_level(self, current_gray: np.ndarray) -> float:
        """Estimate motion level between consecutive frames"""
        try:
            if self.previous_frame is None:
                self.previous_frame = current_gray
                return 0.0
            
            # Calculate frame difference
            frame_diff = cv2.absdiff(self.previous_frame, current_gray)
            motion_level = np.mean(frame_diff) / 255.0
            
            # Update previous frame
            self.previous_frame = current_gray
            
            # Store in history for smoothing
            self.motion_history.append(motion_level)
            
            # Use moving average for stability
            if len(self.motion_history) > 0:
                smoothed_motion = np.mean(list(self.motion_history))
                return min(1.0, smoothed_motion * 3)  # Amplify for better sensitivity
            
            return motion_level
            
        except:
            return 0.0

    def _calculate_texture_density(self, gray_frame: np.ndarray) -> float:
        """Calculate texture density using local binary patterns (simplified)"""
        try:
            # Use variance of Gaussian blur differences as texture indicator
            blur1 = cv2.GaussianBlur(gray_frame, (5, 5), 0)
            blur2 = cv2.GaussianBlur(gray_frame, (9, 9), 0)
            texture_map = cv2.absdiff(blur1, blur2)
            texture_density = np.mean(texture_map) / 255.0
            return min(1.0, texture_density * 2)
        except:
            return 0.5

    def _calculate_edge_concentration(self, gray_frame: np.ndarray) -> float:
        """Calculate edge concentration in different regions"""
        try:
            edges = cv2.Canny(gray_frame, 50, 150)
            h, w = edges.shape
            
            # Divide frame into 3x3 grid and analyze edge distribution
            grid_h, grid_w = h // 3, w // 3
            edge_concentrations = []
            
            for i in range(3):
                for j in range(3):
                    roi = edges[i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w]
                    concentration = np.sum(roi > 0) / roi.size
                    edge_concentrations.append(concentration)
            
            # Use standard deviation of concentrations as measure of focus
            concentration_variance = np.std(edge_concentrations)
            return min(1.0, concentration_variance * 5)
        except:
            return 0.5

    def _calculate_color_variance(self, color_frame: np.ndarray) -> float:
        """Calculate color variance in the scene"""
        try:
            if len(color_frame.shape) != 3:
                return 0.5
                
            hsv = cv2.cvtColor(color_frame, cv2.COLOR_BGR2HSV)
            saturation = hsv[:, :, 1]
            color_variance = np.std(saturation) / 255.0
            return min(1.0, color_variance * 2)
        except:
            return 0.5

    def _assess_focus_quality(self, gray_frame: np.ndarray) -> float:
        """Assess image focus quality using frequency analysis"""
        try:
            # Use FFT to assess focus (blurry images have less high frequency content)
            fft = np.fft.fft2(gray_frame)
            fft_shift = np.fft.fftshift(fft)
            magnitude = np.log(np.abs(fft_shift) + 1)
            
            # High frequency content (edges)
            h, w = magnitude.shape
            center_h, center_w = h // 2, w // 2
            high_freq_region = magnitude[center_h-20:center_h+20, center_w-20:center_w+20]
            high_freq_content = np.mean(high_freq_region)
            
            focus_quality = min(1.0, high_freq_content / 8.0)  # Normalize
            return focus_quality
        except:
            return 0.5

    def get_context_recommendations(self, context: Dict[str, float]) -> Dict[str, any]:
        """Get scaling recommendations based on scene context"""
        recommendations = {
            'suggested_scale_adjustment': 1.0,
            'reasoning': [],
            'priority_factors': {}
        }
        
        # Face density recommendations
        if context['face_density'] > self.thresholds['high_face_density']:
            recommendations['suggested_scale_adjustment'] *= 1.3
            recommendations['reasoning'].append("High face density - increase resolution")
            recommendations['priority_factors']['face_density'] = 'high'
        elif context['face_density'] < self.thresholds['low_face_density']:
            recommendations['suggested_scale_adjustment'] *= 0.8
            recommendations['reasoning'].append("Low face density - can reduce resolution")
            recommendations['priority_factors']['face_density'] = 'low'
        else:
            recommendations['priority_factors']['face_density'] = 'medium'

        # Scene complexity recommendations
        if context['scene_complexity'] > self.thresholds['high_complexity']:
            recommendations['suggested_scale_adjustment'] *= 0.7
            recommendations['reasoning'].append("Complex scene - reduce resolution for performance")
            recommendations['priority_factors']['complexity'] = 'high'
        elif context['scene_complexity'] < self.thresholds['low_complexity']:
            recommendations['suggested_scale_adjustment'] *= 1.2
            recommendations['reasoning'].append("Simple scene - can increase resolution")
            recommendations['priority_factors']['complexity'] = 'low'
        else:
            recommendations['priority_factors']['complexity'] = 'medium'

        # Lighting condition recommendations
        if context['lighting_conditions'] < self.thresholds['dark_lighting']:
            recommendations['suggested_scale_adjustment'] *= 1.4
            recommendations['reasoning'].append("Poor lighting - increase resolution for detail")
            recommendations['priority_factors']['lighting'] = 'poor'
        elif context['lighting_conditions'] > self.thresholds['bright_lighting']:
            recommendations['suggested_scale_adjustment'] *= 1.1
            recommendations['reasoning'].append("Bright lighting - slight increase for overexposure compensation")
            recommendations['priority_factors']['lighting'] = 'bright'
        else:
            recommendations['priority_factors']['lighting'] = 'good'

        # Motion level recommendations
        if context['motion_level'] > self.thresholds['high_motion']:
            recommendations['suggested_scale_adjustment'] *= 0.6
            recommendations['reasoning'].append("High motion - reduce resolution for stability")
            recommendations['priority_factors']['motion'] = 'high'
        elif context['motion_level'] < self.thresholds['low_motion']:
            recommendations['suggested_scale_adjustment'] *= 1.1
            recommendations['reasoning'].append("Low motion - can increase resolution")
            recommendations['priority_factors']['motion'] = 'low'
        else:
            recommendations['priority_factors']['motion'] = 'medium'

        # Focus quality recommendations
        if context['focus_quality'] < 0.3:
            recommendations['suggested_scale_adjustment'] *= 0.8
            recommendations['reasoning'].append("Poor focus - reducing resolution won't help")
            recommendations['priority_factors']['focus'] = 'poor'
        elif context['focus_quality'] > 0.7:
            recommendations['suggested_scale_adjustment'] *= 1.2
            recommendations['reasoning'].append("Good focus - increase resolution for detail")
            recommendations['priority_factors']['focus'] = 'good'
        else:
            recommendations['priority_factors']['focus'] = 'medium'

        return recommendations

    def _get_default_context(self) -> Dict[str, float]:
        """Return default context when analysis fails"""
        return {
            'face_density': 0.0,
            'scene_complexity': 0.5,
            'lighting_conditions': 0.5,
            'motion_level': 0.0,
            'texture_density': 0.5,
            'edge_concentration': 0.5,
            'color_variance': 0.5,
            'focus_quality': 0.5
        }

    def get_context_statistics(self) -> Dict[str, any]:
        """Get statistics about historical context analysis"""
        if not self.context_history:
            return {}
        
        contexts = list(self.context_history)
        stats = {}
        
        for key in contexts[0].keys():
            values = [ctx[key] for ctx in contexts]
            stats[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'trend': 'increasing' if values[-1] > values[0] else 'decreasing'
            }
        
        return stats

class ContextAwareDynamicScaling:
    """Dynamic scaling enhanced with scene context awareness"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.scene_analyzer = SceneContextAnalyzer(config)
        
        # Scaling parameters
        self.scaling_params = {
            'min_scale': config.get('min_processing_scale', 0.3),
            'max_scale': config.get('max_processing_scale', 2.5),
            'base_step': config.get('scale_adjustment_step', 0.1),
            'context_weight': config.get('context_weight', 0.4),  # How much context influences scaling
            'performance_weight': config.get('performance_weight', 0.6)
        }
        
        # Historical data
        self.scaling_decisions = deque(maxlen=100)
        self.performance_history = deque(maxlen=50)
        self.context_recommendations_history = deque(maxlen=50)
        
        self.current_scale = 1.0
        self.adjustment_cooldown = 0
        
        print("🎯 Context-Aware Dynamic Scaling initialized")

    def compute_optimal_scale(self, frame: np.ndarray, detection_results: List[Dict], 
                            performance_metrics: Dict) -> float:
        """Compute optimal scale using both performance and context"""
        
        # Analyze scene context
        scene_context = self.scene_analyzer.analyze_scene_context(frame, detection_results)
        context_recommendations = self.scene_analyzer.get_context_recommendations(scene_context)
        
        # Store for analysis
        self.context_recommendations_history.append({
            'context': scene_context,
            'recommendations': context_recommendations,
            'timestamp': time.time()
        })
        
        # Calculate performance-based scale
        performance_scale = self._calculate_performance_scale(performance_metrics)
        
        # Calculate context-based scale
        context_scale = self._calculate_context_scale(context_recommendations)
        
        # Combine with weights
        optimal_scale = (
            performance_scale * self.scaling_params['performance_weight'] +
            context_scale * self.scaling_params['context_weight']
        )
        
        # Apply predictive adjustments
        optimal_scale = self._apply_predictive_adjustments(optimal_scale, scene_context)
        
        # Apply bounds
        optimal_scale = max(self.scaling_params['min_scale'], 
                           min(self.scaling_params['max_scale'], optimal_scale))
        
        # Record decision
        decision = {
            'timestamp': time.time(),
            'performance_scale': performance_scale,
            'context_scale': context_scale,
            'optimal_scale': optimal_scale,
            'previous_scale': self.current_scale,
            'context': scene_context,
            'performance': performance_metrics,
            'recommendations': context_recommendations
        }
        self.scaling_decisions.append(decision)
        self.performance_history.append(performance_metrics)
        
        return optimal_scale

    def _calculate_performance_scale(self, performance: Dict) -> float:
        """Calculate scale based on performance metrics"""
        base_scale = 1.0
        
        # Face detection performance
        if performance['detection_count'] == 0:
            # No detections - try higher resolution
            base_scale *= 1.5
        elif performance['avg_face_size'] < 40:
            # Small faces - increase resolution
            size_ratio = 40 / max(performance['avg_face_size'], 1)
            base_scale *= min(2.0, 1.0 + (size_ratio - 1) * 0.1)
        
        # Detection quality
        if performance['detection_quality'] < 0.4:
            base_scale *= 1.3  # Poor quality - try higher resolution
        elif performance['detection_quality'] > 0.8:
            base_scale *= 0.8  # Excellent quality - can reduce resolution
        
        return base_scale

    def _calculate_context_scale(self, recommendations: Dict) -> float:
        """Calculate scale based on scene context recommendations"""
        return recommendations.get('suggested_scale_adjustment', 1.0)

    def _apply_predictive_adjustments(self, current_scale: float, context: Dict) -> float:
        """Apply predictive adjustments based on context patterns"""
        if len(self.scaling_decisions) < 5:
            return current_scale
        
        # Analyze historical success with similar contexts
        similar_contexts = []
        for decision in self.scaling_decisions:
            context_similarity = self._calculate_context_similarity(context, decision['context'])
            if context_similarity > 0.7:  # Similar context
                performance_improvement = self._calculate_performance_improvement(decision)
                similar_contexts.append((decision['optimal_scale'], performance_improvement))
        
        if similar_contexts:
            # Find the scale that gave best performance in similar contexts
            best_scale, best_improvement = max(similar_contexts, key=lambda x: x[1])
            
            if best_improvement > 0.1:  # Significant improvement
                # Blend with historical best (30% weight)
                adjusted_scale = current_scale * 0.7 + best_scale * 0.3
                return adjusted_scale
        
        return current_scale

    def _calculate_context_similarity(self, ctx1: Dict, ctx2: Dict) -> float:
        """Calculate similarity between two context profiles"""
        similarity = 0.0
        weights = {
            'face_density': 0.3,
            'scene_complexity': 0.2,
            'lighting_conditions': 0.2,
            'motion_level': 0.15,
            'focus_quality': 0.15
        }
        
        for key, weight in weights.items():
            if key in ctx1 and key in ctx2:
                diff = abs(ctx1[key] - ctx2[key])
                similarity += weight * (1.0 - diff)
        
        return similarity

    def _calculate_performance_improvement(self, decision: Dict) -> float:
        """Calculate performance improvement from a scaling decision"""
        if 'performance' not in decision:
            return 0.0
        
        perf = decision['performance']
        # Simple improvement metric based on detection quality and count
        improvement = perf.get('detection_quality', 0) * 0.7 + \
                     min(1.0, perf.get('detection_count', 0) / 10) * 0.3
        return improvement

    def apply_scale_adjustment(self, new_scale: float) -> bool:
        """Apply scale adjustment with cooldown and validation"""
        if self.adjustment_cooldown > 0:
            self.adjustment_cooldown -= 1
            return False
        
        scale_change = abs(new_scale - self.current_scale)
        
        # Only apply significant changes
        if scale_change >= self.scaling_params['base_step'] * 0.5:
            old_scale = self.current_scale
            self.current_scale = new_scale
            self.adjustment_cooldown = 5  # Cooldown period
            
            # Log the adjustment
            print(f"🎯 Context-aware scaling: {old_scale:.2f} → {new_scale:.2f}")
            
            # Print context reasoning if available
            if self.context_recommendations_history:
                latest_rec = self.context_recommendations_history[-1]['recommendations']
                for reason in latest_rec.get('reasoning', [])[:2]:  # Show top 2 reasons
                    print(f"   📋 {reason}")
            
            return True
        
        return False

    def get_scaling_statistics(self) -> Dict:
        """Get scaling statistics and insights"""
        stats = {
            'current_scale': self.current_scale,
            'total_decisions': len(self.scaling_decisions),
            'average_scale': 0.0,
            'context_influence': self.scaling_params['context_weight'],
            'recent_context_stats': self.scene_analyzer.get_context_statistics()
        }
        
        if self.scaling_decisions:
            scales = [d['optimal_scale'] for d in self.scaling_decisions]
            stats['average_scale'] = np.mean(scales)
            stats['scale_variance'] = np.var(scales)
        
        # Most common context recommendations
        if self.context_recommendations_history:
            recent_recs = list(self.context_recommendations_history)[-10:]  # Last 10
            common_reasons = {}
            for rec in recent_recs:
                for reason in rec['recommendations'].get('reasoning', []):
                    common_reasons[reason] = common_reasons.get(reason, 0) + 1
            
            stats['common_recommendations'] = common_reasons
        
        return stats

class EnhancedSimilarityEngine:
    def __init__(self, config: Dict):
        self.config = config
        self.similarity_methods = {
            'cosine': self.cosine_similarity,
            'euclidean': self.euclidean_similarity,
            'manhattan': self.manhattan_similarity,
            'mahalanobis': self.mahalanobis_similarity,
            'pearson': self.pearson_correlation,
            'jaccard': self.jaccard_similarity,
            'wasserstein': self.wasserstein_distance,
            'angular': self.angular_similarity,
            'dot_product': self.dot_product_similarity,
            'canberra': self.canberra_similarity
        }
        
        # Statistical data for Mahalanobis (learned from embeddings)
        self.embedding_covariance = None
        self.embedding_mean = None
        self.covariance_inv = None
              
        
    def compute_similarity_matrix(self, embedding: np.ndarray, centroids: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Compute multiple similarity scores and return weighted combination"""
        similarity_scores = {}
        
        for identity, centroid in centroids.items():
            scores = {}
            
            # Compute all similarity measures
            for method_name, method_func in self.similarity_methods.items():
                if method_name == 'mahalanobis' and self.covariance_inv is None:
                    continue  # Skip Mahalanobis until trained
                try:
                    score = method_func(embedding, centroid)
                    scores[method_name] = score
                except Exception as e:
                    if self.config.get('verbose', False):
                        print(f"Similarity method {method_name} failed: {e}")
                    scores[method_name] = 0.0
            
            # Weighted combination (configurable weights)
            weights = self.config.get('similarity_weights', {
                'cosine': 0.25,
                'angular': 0.20,
                'dot_product': 0.15,
                'euclidean': 0.10,
                'manhattan': 0.08,
                'pearson': 0.12,
                'jaccard': 0.05,
                'canberra': 0.05
            })
            
            final_score = 0.0
            total_weight = 0.0
            
            for method, score in scores.items():
                if method in weights:
                    # Normalize scores to 0-1 range if needed
                    normalized_score = self._normalize_score(method, score)
                    final_score += weights[method] * normalized_score
                    total_weight += weights[method]
            
            if total_weight > 0:
                similarity_scores[identity] = final_score / total_weight
            else:
                similarity_scores[identity] = 0.0
        
        return similarity_scores
    
    def _normalize_score(self, method: str, score: float) -> float:
        """Normalize different similarity measures to 0-1 range - FIXED VERSION"""
        if method in ['cosine', 'pearson', 'angular', 'dot_product']:
            # Map from [-1, 1] to [0, 1]
            normalized = (score + 1) / 2.0
            return max(0.0, min(1.0, normalized))
        elif method in ['jaccard']:
            # Already in [0, 1] range
            return max(0.0, min(1.0, score))
        elif method in ['euclidean', 'manhattan', 'canberra', 'wasserstein', 'mahalanobis']:
            # Distance metrics - convert to similarity with exponential decay
            if score == 0:
                return 1.0
            # Use exponential decay for better normalization
            similarity = np.exp(-score / 2.0)  # Better than 1/(1+score)
            return max(0.0, min(1.0, similarity))
        else:
            return max(0.0, min(1.0, score))
    
    # 🎯 CORE SIMILARITY METHODS - FIXED VERSIONS
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Enhanced cosine similarity with epsilon protection"""
        vec1 = vec1.flatten().astype(np.float64)
        vec2 = vec2.flatten().astype(np.float64)
        
        # Add small epsilon to avoid division by zero
        eps = 1e-8
        norm1 = np.linalg.norm(vec1) + eps
        norm2 = np.linalg.norm(vec2) + eps
        
        cosine_sim = np.dot(vec1, vec2) / (norm1 * norm2)
        return float(cosine_sim)  # Returns [-1, 1]
    
    def euclidean_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Euclidean distance - will be converted to similarity in normalization"""
        distance = np.linalg.norm(vec1 - vec2)
        return float(distance)  # Returns distance (0 to ∞)
    
    def manhattan_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Manhattan distance (L1 norm)"""
        distance = np.sum(np.abs(vec1 - vec2))
        return float(distance)  # Returns distance (0 to ∞)
    
    def angular_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Angular similarity - more robust for high dimensions"""
        cosine_sim = self.cosine_similarity(vec1, vec2)
        # Convert cosine to angular distance in radians, then to similarity
        angular_distance = np.arccos(np.clip(cosine_sim, -1.0, 1.0))
        angular_similarity = 1.0 - (angular_distance / np.pi)
        return float(angular_similarity)  # Returns [0, 1]
    
    def dot_product_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Normalized dot product similarity"""
        dot_result = np.dot(vec1.flatten(), vec2.flatten())
        # Normalize by vector magnitudes to get cosine similarity
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 > 0 and norm2 > 0:
            return float(dot_result / (norm1 * norm2))  # Same as cosine
        return 0.0
    
    # 📊 STATISTICAL SIMILARITY METHODS - FIXED VERSIONS
    
    def pearson_correlation(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Pearson correlation coefficient with improved handling"""
        vec1_flat = vec1.flatten().astype(np.float64)
        vec2_flat = vec2.flatten().astype(np.float64)
        
        # Handle constant vectors
        vec1_std, vec2_std = np.std(vec1_flat), np.std(vec2_flat)
        if vec1_std == 0 and vec2_std == 0:
            # Both constant - check if they're the same
            return 1.0 if np.allclose(vec1_flat, vec2_flat) else 0.0
        elif vec1_std == 0 or vec2_std == 0:
            # One constant, one variable - no correlation
            return 0.0
            
        correlation = np.corrcoef(vec1_flat, vec2_flat)[0, 1]
        return float(correlation) if not np.isnan(correlation) else 0.0
    
    def mahalanobis_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Mahalanobis distance - accounts for feature correlations"""
        if self.covariance_inv is None:
            return 0.0
            
        try:
            diff = vec1.flatten() - vec2.flatten()
            # Handle dimensionality mismatch
            if len(diff) != self.covariance_inv.shape[0]:
                return 0.0
                
            distance = np.sqrt(np.dot(np.dot(diff, self.covariance_inv), diff))
            return float(distance)  # Returns distance (0 to ∞)
        except:
            return 0.0
    
    def canberra_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Canberra distance - weighted version of Manhattan"""
        vec1_flat = vec1.flatten()
        vec2_flat = vec2.flatten()
        
        numerator = np.abs(vec1_flat - vec2_flat)
        denominator = np.abs(vec1_flat) + np.abs(vec2_flat)
        
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            canberra = np.sum(np.where(denominator != 0, numerator / denominator, 0))
        
        return float(canberra)  # Returns distance (0 to ∞)
    
    # 🎪 SET-BASED SIMILARITY METHODS - FIXED VERSIONS
    
    def jaccard_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Improved Jaccard similarity for continuous vectors"""
        # Use percentile-based thresholding instead of fixed 0
        all_values = np.concatenate([vec1.flatten(), vec2.flatten()])
        threshold = np.percentile(all_values, 50)  # Median as threshold
        
        vec1_binary = (vec1 > threshold).astype(int)
        vec2_binary = (vec2 > threshold).astype(int)
        
        intersection = np.sum(vec1_binary & vec2_binary)
        union = np.sum(vec1_binary | vec2_binary)
        
        if union == 0:
            return 0.0
            
        return float(intersection / union)  # Returns [0, 1]
    
    def wasserstein_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Earth Mover's Distance approximation"""
        # Sort vectors and compute EMD
        vec1_sorted = np.sort(vec1.flatten())
        vec2_sorted = np.sort(vec2.flatten())
        
        # Ensure same length by interpolation if needed
        min_len = min(len(vec1_sorted), len(vec2_sorted))
        if min_len == 0:
            return 0.0
            
        wasserstein = np.mean(np.abs(vec1_sorted[:min_len] - vec2_sorted[:min_len]))
        return float(wasserstein)  # Returns distance (0 to ∞)
    
    # 🧠 MACHINE LEARNING ENHANCEMENTS
    
    def learn_statistics(self, embeddings: List[np.ndarray]):
        """Learn covariance matrix for Mahalanobis distance"""
        if not embeddings:
            return
            
        try:
            # Stack all embeddings
            embedding_matrix = np.vstack([emb.flatten() for emb in embeddings])
            
            # Compute mean and covariance
            self.embedding_mean = np.mean(embedding_matrix, axis=0)
            self.embedding_covariance = np.cov(embedding_matrix, rowvar=False)
            
            # Add small epsilon for numerical stability
            epsilon = 1e-6
            self.embedding_covariance += np.eye(self.embedding_covariance.shape[0]) * epsilon
            
            # Precompute inverse covariance
            self.covariance_inv = np.linalg.pinv(self.embedding_covariance)
            
            print(f"✅ Learned Mahalanobis statistics on {len(embeddings)} embeddings")
            print(f"   - Covariance shape: {self.embedding_covariance.shape}")
            
        except Exception as e:
            print(f"❌ Failed to learn Mahalanobis statistics: {e}")
    
    def adaptive_similarity(self, vec1: np.ndarray, vec2: np.ndarray, 
                          method_weights: Dict[str, float] = None) -> float:
        """Adaptive similarity that learns which methods work best"""
        if method_weights is None:
            method_weights = {
                'cosine': 0.30,
                'angular': 0.25,
                'pearson': 0.20,
                'dot_product': 0.15,
                'manhattan': 0.10
            }
        
        total_score = 0.0
        total_weight = 0.0
        
        for method, weight in method_weights.items():
            if method in self.similarity_methods:
                try:
                    raw_score = self.similarity_methods[method](vec1, vec2)
                    normalized_score = self._normalize_score(method, raw_score)
                    total_score += weight * normalized_score
                    total_weight += weight
                except:
                    continue
        
        return total_score / total_weight if total_weight > 0 else 0.0

class AdaptiveWeightSimilarityEngine(EnhancedSimilarityEngine):
    """Online weight optimization based on method performance"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        # Performance tracking
        self.method_performance = defaultdict(lambda: {
            'true_positives': 0, 'false_positives': 0,
            'true_negatives': 0, 'false_negatives': 0,
            'total_calls': 0, 'recent_scores': deque(maxlen=100)
        })
        
        # Weight learning
        self.current_weights = config.get('similarity_weights', {
            'cosine': 0.25, 'angular': 0.20, 'pearson': 0.15,
            'dot_product': 0.15, 'euclidean': 0.10, 'manhattan': 0.10, 'jaccard': 0.05
        })
        
        self.learning_rate = config.get('weight_learning_rate', 0.01)
        self.min_samples_for_learning = config.get('min_learning_samples', 50)
        self.performance_history = deque(maxlen=1000)
        
        # Embedding statistics
        self.embedding_statistics = {
            'mean_norm': 0.0,
            'std_norm': 1.0,
            'sparsity': 0.0,
            'dimensionality': 0
        }
        
        print("🎯 Adaptive Weight Similarity Engine initialized")

    def update_performance_tracking(self, result: Dict, ground_truth: Optional[bool] = None):
        """Update method performance based on recognition results"""
        if ground_truth is None or 'detailed_scores' not in result:
            return
            
        true_identity = result.get('true_identity')
        predicted_identity = result.get('identity')
        detailed_scores = result['detailed_scores']
        
        for method_name, score in detailed_scores.items():
            if method_name not in self.similarity_methods:
                continue
                
            perf = self.method_performance[method_name]
            perf['total_calls'] += 1
            perf['recent_scores'].append(score)
            
            # Determine if this method would have made correct prediction
            threshold = result.get('adaptive_threshold', 0.5)
            method_prediction = score >= threshold
            
            # Compare with ground truth
            is_correct = (predicted_identity == true_identity) if true_identity else None
            
            if is_correct is not None:
                if method_prediction and is_correct:
                    perf['true_positives'] += 1
                elif method_prediction and not is_correct:
                    perf['false_positives'] += 1
                elif not method_prediction and is_correct:
                    perf['false_negatives'] += 1
                else:  # not method_prediction and not is_correct
                    perf['true_negatives'] += 1
                        
    # Add to AdaptiveWeightSimilarityEngine class
    def compute_quality_adaptive_similarity(self, embedding: np.ndarray, 
                                        centroids: Dict[str, np.ndarray],
                                        quality_scores: Dict[str, float]) -> Dict[str, float]:
        """Compute similarity with quality-based method selection"""
        # Use quality scores to select appropriate methods
        quality = quality_scores.get('overall', 0.5)
        
        if quality > 0.7:
            # High quality - use precise methods
            method_weights = {
                'cosine': 0.35, 'angular': 0.25, 'pearson': 0.20,
                'dot_product': 0.20, 'euclidean': 0.0, 'manhattan': 0.0
            }
        elif quality > 0.4:
            # Medium quality - balanced approach
            method_weights = {
                'cosine': 0.25, 'angular': 0.25, 'pearson': 0.15,
                'dot_product': 0.15, 'euclidean': 0.10, 'manhattan': 0.10
            }
        else:
            # Low quality - robust methods
            method_weights = {
                'cosine': 0.15, 'angular': 0.20, 'pearson': 0.10,
                'dot_product': 0.10, 'euclidean': 0.15, 'manhattan': 0.20,
                'jaccard': 0.10
            }
        
        return self.adaptive_weighted_similarity(embedding, centroids, method_weights)                    

    def calculate_method_performance(self, method_name: str) -> Dict[str, float]:
        """Calculate comprehensive performance metrics for a method"""
        perf = self.method_performance[method_name]
        
        tp = perf['true_positives']
        fp = perf['false_positives']
        tn = perf['true_negatives']
        fn = perf['false_negatives']
        total = tp + fp + tn + fn
        
        if total == 0:
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'confidence': 0.0}
        
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Confidence based on sample size and consistency
        sample_confidence = min(1.0, total / self.min_samples_for_learning)
        score_std = np.std(list(perf['recent_scores'])) if perf['recent_scores'] else 1.0
        consistency_confidence = 1.0 / (1.0 + score_std)  # Higher when scores are consistent
        
        confidence = sample_confidence * consistency_confidence
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confidence': confidence,
            'total_samples': total
        }

    def update_weights_online(self):
        """Update weights based on recent performance"""
        if len(self.performance_history) < self.min_samples_for_learning:
            return
            
        total_performance = 0.0
        method_scores = {}
        
        # Calculate performance scores for each method
        for method_name in self.similarity_methods.keys():
            perf_metrics = self.calculate_method_performance(method_name)
            
            # Combined performance score (weighted combination of metrics)
            performance_score = (
                0.4 * perf_metrics['f1'] +
                0.3 * perf_metrics['accuracy'] +
                0.2 * perf_metrics['precision'] +
                0.1 * perf_metrics['recall']
            ) * perf_metrics['confidence']
            
            method_scores[method_name] = performance_score
            total_performance += performance_score
        
        # Normalize and update weights
        if total_performance > 0:
            new_weights = {}
            for method_name, score in method_scores.items():
                current_weight = self.current_weights.get(method_name, 0.1)
                target_weight = score / total_performance
                
                # Smooth update with learning rate
                updated_weight = (1 - self.learning_rate) * current_weight + self.learning_rate * target_weight
                new_weights[method_name] = updated_weight
            
            # Renormalize
            total_weight = sum(new_weights.values())
            if total_weight > 0:
                self.current_weights = {k: v/total_weight for k, v in new_weights.items()}
                
            if self.config.get('verbose', False) and len(self.performance_history) % 100 == 0:
                print("🔄 Updated similarity weights:", {k: f"{v:.3f}" for k, v in self.current_weights.items()})

    def adaptive_weighted_similarity(self, embedding: np.ndarray, centroids: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Compute similarity using adaptively learned weights"""
        similarity_scores = {}
        
        for identity, centroid in centroids.items():
            total_score = 0.0
            total_weight = 0.0
            
            for method_name, method_func in self.similarity_methods.items():
                if method_name not in self.current_weights:
                    continue
                    
                try:
                    raw_score = method_func(embedding, centroid)
                    normalized_score = self._normalize_score(method_name, raw_score)
                    weight = self.current_weights[method_name]
                    
                    total_score += weight * normalized_score
                    total_weight += weight
                except Exception as e:
                    if self.config.get('verbose', False):
                        print(f"⚠️ Method {method_name} failed: {e}")
                    continue
            
            if total_weight > 0:
                similarity_scores[identity] = total_score / total_weight
            else:
                similarity_scores[identity] = 0.0
        
        return similarity_scores

    def get_weight_statistics(self) -> Dict:
        """Get current weight statistics and performance"""
        stats = {
            'current_weights': self.current_weights,
            'total_performance_samples': len(self.performance_history),
            'method_performance': {}
        }
        
        for method_name in self.similarity_methods.keys():
            stats['method_performance'][method_name] = self.calculate_method_performance(method_name)
            
        return stats

class MultiScaleFaceProcessor:
    def __init__(self, config: Dict):
        self.config = config
        self.scale_factors = [0.5, 0.75, 1.0, 1.25, 1.5]  # Multi-scale processing
        self.rotation_angles = [-10, -5, 0, 5, 10]  # Small rotations for robustness
        
    def _extract_single_embedding(self, face_roi: np.ndarray) -> Optional[np.ndarray]:
        """Extract embedding from a single face ROI"""
        # This would typically call the main system's embedding extraction
        # For now, return a dummy embedding or implement actual extraction
        try:
            # Convert to RGB and normalize
            if len(face_roi.shape) == 3:
                face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            else:
                face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2RGB)
            
            face_rgb = face_rgb.astype(np.float32) / 255.0
            
            # Use DeepFace for embedding extraction
            embedding_obj = DeepFace.represent(
                face_rgb,
                model_name=self.config.get('embedding_model', 'Facenet'),
                enforce_detection=False,
                detector_backend='skip',
                align=True
            )
            
            if embedding_obj and len(embedding_obj) > 0:
                return np.array(embedding_obj[0]['embedding'])
                
        except Exception as e:
            if self.config.get('verbose', False):
                print(f"Multi-scale embedding extraction error: {e}")
                
        return None        
        
    def extract_multi_scale_embeddings(self, face_roi: np.ndarray) -> List[np.ndarray]:
        """Extract embeddings from multiple scales and slight rotations"""
        embeddings = []
        h, w = face_roi.shape[:2]
        
        for scale in self.scale_factors:
            # Scale the face ROI
            new_w, new_h = int(w * scale), int(h * scale)
            if new_w < 20 or new_h < 20:  # Minimum size
                continue
                
            scaled_face = cv2.resize(face_roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Extract embedding from scaled version
            embedding = self._extract_single_embedding(scaled_face)
            if embedding is not None:
                embeddings.append(embedding)
                
            # Add slightly rotated versions
            for angle in self.rotation_angles:
                rotated_face = self._rotate_face(scaled_face, angle)
                rot_embedding = self._extract_single_embedding(rotated_face)
                if rot_embedding is not None:
                    embeddings.append(rot_embedding)
        
        return embeddings
    
    def _rotate_face(self, face: np.ndarray, angle: float) -> np.ndarray:
        """Rotate face by small angle for robustness"""
        h, w = face.shape[:2]
        center = (w // 2, h // 2)
        
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(face, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC)
        
        return rotated
    
    def fuse_embeddings(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """Fuse multiple embeddings into robust representation"""
        if not embeddings:
            return None
            
        # Simple average fusion
        fused = np.mean(embeddings, axis=0)
        
        # Optional: weighted fusion based on quality scores
        # weights = [self._calculate_embedding_quality(emb) for emb in embeddings]
        # fused = np.average(embeddings, axis=0, weights=weights)
        
        return fused
    
class TemporalFusion:
    def __init__(self, config: Dict):
        self.config = config
        self.temporal_buffer = {}  # track_id -> deque of recent recognitions
        self.buffer_size = config.get('temporal_buffer_size', 10)
        self.confidence_threshold = config.get('temporal_confidence_threshold', 0.6)
        
    def update_temporal_buffer(self, track_id: int, identity: str, confidence: float):
        """Update temporal buffer with recent recognition results"""
        if track_id not in self.temporal_buffer:
            self.temporal_buffer[track_id] = deque(maxlen=self.buffer_size)
        
        self.temporal_buffer[track_id].append({
            'identity': identity,
            'confidence': confidence,
            'timestamp': time.time()
        })
    
    def get_temporal_consensus(self, track_id: int) -> Tuple[Optional[str], float]:
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
            
            if combined_score > best_score and combined_score > self.confidence_threshold:
                best_score = combined_score
                best_identity = identity
        
        return best_identity, best_score
    
class FaceQualityAssessor:
    def __init__(self, config: Dict):
        self.config = config
        
    def assess_face_quality(self, face_roi: np.ndarray, bbox: List[int] = None) -> Dict[str, float]:
        """Comprehensive face quality assessment"""
        quality_scores = {}
        
        # 1. Brightness and contrast
        quality_scores['brightness'] = self._assess_brightness(face_roi)
        quality_scores['contrast'] = self._assess_contrast(face_roi)
        
        # 2. Sharpness (Laplacian variance)
        quality_scores['sharpness'] = self._assess_sharpness(face_roi)
        
        # 3. Face size and position
        if bbox:
            quality_scores['size'] = self._assess_face_size(bbox, face_roi.shape)
            quality_scores['position'] = self._assess_face_position(bbox, face_roi.shape)
        
        # 4. Blur detection
        quality_scores['blur'] = self._assess_blur(face_roi)
        
        # 5. Overall quality score (weighted combination)
        weights = {
            'brightness': 0.15,
            'contrast': 0.15,
            'sharpness': 0.25,
            'size': 0.20,
            'position': 0.15,
            'blur': 0.10
        }
        
        overall_quality = 0.0
        for metric, score in quality_scores.items():
            if metric in weights:
                overall_quality += weights[metric] * score
        
        quality_scores['overall'] = overall_quality
        
        return quality_scores
    
    def _assess_sharpness(self, face: np.ndarray) -> float:
        """Assess image sharpness using Laplacian variance"""
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Normalize to 0-1 range (typical good values: 100-1000)
        sharpness = min(1.0, laplacian_var / 500.0)
        return sharpness
    
    def _assess_brightness(self, face: np.ndarray) -> float:
        """Assess brightness level"""
        hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
        brightness = np.mean(hsv[:,:,2]) / 255.0
        
        # Ideal brightness is around 0.5, penalize extremes
        brightness_score = 1.0 - abs(brightness - 0.5) * 2.0
        return max(0.0, brightness_score)
    
    def _assess_contrast(self, face: np.ndarray) -> float:
        """Assess contrast using standard deviation"""
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        contrast = np.std(gray) / 255.0
        return min(1.0, contrast * 2.0)  # Normalize
    
    def _assess_blur(self, face: np.ndarray) -> float:
        """Assess blur using FFT"""
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Compute FFT and shift
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        
        # Calculate magnitude spectrum
        magnitude = 20 * np.log(np.abs(fft_shift) + 1)
        
        # High-frequency content indicates sharpness
        center_h, center_w = h // 2, w // 2
        high_freq = magnitude[center_h-10:center_h+10, center_w-10:center_w+10]
        high_freq_mean = np.mean(high_freq)
        
        # Normalize to 0-1
        blur_score = min(1.0, high_freq_mean / 30.0)
        return blur_score
    
    def _assess_face_size(self, bbox: List[int], frame_shape: Tuple[int, int]) -> float:
        """Assess if face is large enough for good recognition"""
        x1, y1, x2, y2 = bbox
        face_area = (x2 - x1) * (y2 - y1)
        frame_area = frame_shape[0] * frame_shape[1]
        
        ratio = face_area / frame_area
        
        # Ideal ratio is around 0.1-0.3 of frame area
        if ratio < 0.05:  # Too small
            return ratio / 0.05
        elif ratio > 0.3:  # Too large (might be too close)
            return 0.3 / ratio
        else:
            return 1.0
    
    def _assess_face_position(self, bbox: List[int], frame_shape: Tuple[int, int]) -> float:
        """Assess face position in frame"""
        x1, y1, x2, y2 = bbox
        frame_h, frame_w = frame_shape
        
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Distance from center (normalized)
        dist_x = abs(center_x - frame_w / 2) / (frame_w / 2)
        dist_y = abs(center_y - frame_h / 2) / (frame_h / 2)
        
        # Combined distance
        distance = np.sqrt(dist_x**2 + dist_y**2)
        
        # Score: 1.0 at center, decreasing towards edges
        position_score = max(0.0, 1.0 - distance)
        return position_score
    
class AdaptiveThresholdManager:
    def __init__(self, config: Dict):
        self.config = config
        self.base_threshold = config.get('recognition_threshold', 0.5)
        self.quality_weights = config.get('quality_weights', {
            'sharpness': 0.3,
            'brightness': 0.2,
            'size': 0.25,
            'position': 0.15,
            'blur': 0.1
        })
        
    def compute_adaptive_threshold(self, quality_scores: Dict[str, float]) -> float:
        """Compute adaptive threshold based on face quality"""
        if not quality_scores:
            return self.base_threshold
        
        # Calculate quality-weighted threshold adjustment
        quality_adjustment = 0.0
        total_weight = 0.0
        
        for metric, weight in self.quality_weights.items():
            if metric in quality_scores:
                # Lower quality -> higher threshold (more strict)
                adjustment = (1.0 - quality_scores[metric]) * weight
                quality_adjustment += adjustment
                total_weight += weight
        
        if total_weight > 0:
            quality_adjustment /= total_weight
        
        # Adaptive threshold: base + adjustment
        adaptive_threshold = self.base_threshold + quality_adjustment * (1.0 - self.base_threshold)
        
        # Clamp to reasonable range
        adaptive_threshold = max(self.base_threshold * 0.5, min(0.95, adaptive_threshold))
        
        return adaptive_threshold
    
    def should_process_face(self, quality_scores: Dict[str, float]) -> bool:
        """Decide whether to process face based on quality"""
        if not quality_scores:
            return True
        
        overall_quality = quality_scores.get('overall', 0.5)
        min_quality = self.config.get('min_face_quality', 0.3)
        
        return overall_quality >= min_quality              
    
class AlertManager:
    def __init__(self, config: Dict):
        self.config = config
        self.server_url = config.get('alert_server_url')
        self.cooldown_seconds = config.get('alert_cooldown_seconds', 30)
        self.enabled = config.get('enable_voice_alerts', True)
        self.last_alert_time = 0
        self.alert_lock = threading.Lock()
        
    def send_voice_alert(self, message: str, identity: str = None, mask_status: str = None):
        """Send voice alert to server in non-blocking thread"""
        if not self.enabled or not self.server_url:
            return False
            
        # Rate limiting
        current_time = time.time()
        with self.alert_lock:
            if current_time - self.last_alert_time < self.cooldown_seconds:
                return False
            self.last_alert_time = current_time
        
        # Run in background thread to avoid blocking main processing
        thread = threading.Thread(
            target=self._send_alert_thread,
            args=(message, identity, mask_status),
            daemon=True
        )
        thread.start()
        return True
    
    def _send_alert_thread(self, message: str, identity: str, mask_status: str):
        """Background thread for sending alerts"""
        try:
            # URL encode the message
            encoded_message = quote(message)
            alert_url = f"{self.server_url}?pesan={encoded_message}"
            
            # Send HTTP request with timeout
            response = requests.get(alert_url, timeout=5)
            
            if response.status_code == 200:
                print(f"🔊 Voice alert sent: {message}")
            else:
                print(f"❌ Alert server returned status: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to send voice alert: {e}")
        except Exception as e:
            print(f"❌ Unexpected error in alert thread: {e}")
    
    def toggle_alerts(self):
        """Toggle voice alerts on/off"""
        self.enabled = not self.enabled
        status = "ENABLED" if self.enabled else "DISABLED"
        print(f"🔊 Voice alerts: {status}")
        return self.enabled

class FaceRecognitionSystem:
    def __init__(self, config: Dict):
        self.config = config
        self.detection_model = None
        self.mask_detector = None
        self.embeddings_db = {}
        self.identity_centroids = {}
        
        # ENHANCED: Use circular buffers to prevent memory leaks
        self.debug_stats = {
            'total_frames_processed': 0,
            'total_faces_detected': 0,
            'total_faces_recognized': 0,
            'total_masks_detected': 0,
            'detection_times': deque(maxlen=50),  # Fixed size - was unbounded list
            'mask_detection_times': deque(maxlen=50),
            'embedding_times': deque(maxlen=50),
            'recognition_times': deque(maxlen=50),
            'last_processing_time': 0
        }
        
        self._load_models()
        self._load_mask_detector()  
        self._load_embeddings_database()
        
    def reset_statistics_periodically(self):
        """Reset statistics periodically to prevent memory issues"""
        if self.debug_stats['total_frames_processed'] % 1000 == 0:  # Reset every 1000 frames
            self.debug_stats['total_frames_processed'] = 0
            self.debug_stats['total_faces_detected'] = 0
            self.debug_stats['total_faces_recognized'] = 0
            self.debug_stats['total_masks_detected'] = 0        

    def _load_mask_detector(self):
        """Load ONNX mask detection model"""
        try:
            mask_model_path = Path(self.config.get('mask_model_path', ''))
            if not mask_model_path.exists():
                print("⚠️  Mask model not found, continuing without mask detection")
                return
                
            # Initialize ONNX Runtime session
            self.mask_detector = ort.InferenceSession(str(mask_model_path))
            
            # Validate model input/output
            input_name = self.mask_detector.get_inputs()[0].name
            output_name = self.mask_detector.get_outputs()[0].name
            
            print(f"✅ Mask detection model loaded")
            print(f"   - Input: {input_name}, Output: {output_name}")
            print(f"   - Input shape: {self.mask_detector.get_inputs()[0].shape}")
            
        except Exception as e:
            print(f"❌ Failed to load mask detection model: {e}")
            # Don't raise exception, continue without mask detection

    def detect_mask(self, face_roi: np.ndarray) -> Tuple[str, float]:
        """Enhanced mask detection with better ROI handling and thresholding"""
        if self.mask_detector is None:
            return "no_mask", 0.0
            
        start_time = time.time()
        
        try:
            # More thorough ROI validation
            if (face_roi.size == 0 or face_roi.shape[0] < 40 or face_roi.shape[1] < 40 or
                np.std(face_roi) < 15):  # Check for low variance (blurry/featureless)
                return "unknown", 0.0
                
            # Preserve aspect ratio with smart padding
            input_size = (224, 224)
            h, w = face_roi.shape[:2]
            
            # Calculate scaling factor that preserves aspect ratio
            scale = min(input_size[0] / h, input_size[1] / w)
            new_h, new_w = int(h * scale), int(w * scale)
            
            # Resize with aspect ratio preservation
            resized_face = cv2.resize(face_roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Create padded image
            padded_face = np.zeros((input_size[0], input_size[1], 3), dtype=np.float32)
            y_offset = (input_size[0] - new_h) // 2
            x_offset = (input_size[1] - new_w) // 2
            padded_face[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_face
            
            # Convert to RGB and normalize
            rgb_face = cv2.cvtColor(padded_face.astype(np.uint8), cv2.COLOR_BGR2RGB)
            normalized_face = rgb_face.astype(np.float32) / 255.0
            
            # Add batch dimension
            input_data = np.expand_dims(normalized_face, axis=0)
            
            # Run inference
            input_name = self.mask_detector.get_inputs()[0].name
            output_name = self.mask_detector.get_outputs()[0].name
            
            outputs = self.mask_detector.run([output_name], {input_name: input_data})
            predictions = outputs[0][0]  # First batch, first prediction
            
            # 🎯 ENHANCED FRAMEWORK LOGIC WITH CONFIDENCE THRESHOLDING
            mask_prob = float(predictions[0])
            without_mask_prob = float(predictions[1])
            
            # Get the configuration threshold (default to 0.8 if not set)
            mask_threshold = self.config.get('mask_detection_threshold', 0.8)
            
            # Use threshold to reduce false positives
            if mask_prob > without_mask_prob and mask_prob >= mask_threshold:
                mask_status = "mask"
                confidence = mask_prob
            else: #without_mask_prob > mask_prob and without_mask_prob >= mask_threshold:
                mask_status = "no_mask" 
                confidence = without_mask_prob
            # else:
            #     # Confidence too low for either class
            #     mask_status = "unknown"
            #     confidence = max(mask_prob, without_mask_prob)
            
            # Update stats
            mask_time = (time.time() - start_time) * 1000
            self.debug_stats['mask_detection_times'].append(mask_time)
            
            return mask_status, confidence
            
        except Exception as e:
            print(f"Mask detection error: {e}")
            return "unknown", 0.0
                
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
            
            # Remove the manual slicing - deque handles it automatically
            detection_time = (time.time() - start_time) * 1000
            self.debug_stats['detection_times'].append(detection_time)
                        
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
        
        return best_identity, best_similarity

    def process_frame(self, frame: np.ndarray) -> List[Dict]:
        """Enhanced pipeline: detect → mask detection → extract → recognize"""
        start_time = time.time()
        results = []
        
        # Detect faces
        detections = self.detect_faces(frame)
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            
            padding = self.config.get('roi_padding', 20)  # Increased padding for better mask detection
            h, w = frame.shape[:2]
            x1_pad = max(0, x1 - padding)
            y1_pad = max(0, y1 - padding)
            x2_pad = min(w, x2 + padding)
            y2_pad = min(h, y2 + padding)
            
            face_roi = frame[y1_pad:y2_pad, x1_pad:x2_pad]
            
            # Better ROI validation
            if (face_roi.size == 0 or face_roi.shape[0] < 40 or face_roi.shape[1] < 40 or
                np.std(face_roi) < 10):  # Check for low contrast
                continue
                
            # NEW: Enhanced Mask detection
            mask_status, mask_confidence = self.detect_mask(face_roi)
            
            # Continue with embedding extraction only if we have a good face ROI
            embedding = self.extract_embedding(face_roi)
            if embedding is None:
                continue
                
            identity, recognition_confidence = self.recognize_face(embedding)
            
            # Update mask statistics
            if mask_status == "mask":
                self.debug_stats['total_masks_detected'] += 1
            
            results.append({
                'bbox': detection['bbox'],
                'detection_confidence': detection['confidence'],
                'mask_status': mask_status,  
                'mask_confidence': mask_confidence,  
                'identity': identity,
                'recognition_confidence': recognition_confidence,
                'embedding': embedding.tolist()
            })
        
        # Update overall stats
        self.debug_stats['total_frames_processed'] += 1
        self.debug_stats['total_faces_detected'] += len(detections)
        self.debug_stats['total_faces_recognized'] += len([r for r in results if r['identity']])
        self.debug_stats['last_processing_time'] = (time.time() - start_time) * 1000
            
        return results
        
    def get_debug_stats(self) -> Dict:
        """Enhanced performance statistics"""
        stats = self.debug_stats.copy()
        
        # Calculate averages and percentiles
        stats['avg_detection_time'] = np.mean(stats['detection_times']) if stats['detection_times'] else 0
        stats['p95_detection_time'] = np.percentile(stats['detection_times'], 95) if stats['detection_times'] else 0
        stats['max_detection_time'] = np.max(stats['detection_times']) if stats['detection_times'] else 0
        
        stats['avg_embedding_time'] = np.mean(stats['embedding_times']) if stats['embedding_times'] else 0
        stats['p95_embedding_time'] = np.percentile(stats['embedding_times'], 95) if stats['embedding_times'] else 0
        
        stats['avg_recognition_time'] = np.mean(stats['recognition_times']) if stats['recognition_times'] else 0
        
        # Calculate mask detection time
        stats['avg_detection_time'] = np.mean(stats['detection_times']) if stats['detection_times'] else 0
        stats['avg_mask_detection_time'] = np.mean(stats['mask_detection_times']) if stats['mask_detection_times'] else 0  # NEW     
        
        if stats['total_faces_detected'] > 0:
            stats['mask_detection_rate'] = (stats['total_masks_detected'] / stats['total_faces_detected']) * 100
        else:
            stats['mask_detection_rate'] = 0
                     
        # Calculate rates and efficiencies
        if stats['total_faces_detected'] > 0:
            stats['recognition_rate'] = (stats['total_faces_recognized'] / stats['total_faces_detected']) * 100
        else:
            stats['recognition_rate'] = 0
            
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

# 🔧 INTEGRATION WITH EXISTING FACE RECOGNITION SYSTEM

def get_quality_factors_method_selection(self, quality_scores: Dict[str, float]) -> List[str]:
    """Select methods based on specific quality factors"""
    sharpness = quality_scores.get('sharpness', 0.5)
    brightness = quality_scores.get('brightness', 0.5)
    contrast = quality_scores.get('contrast', 0.5)
    
    selected_methods = []
    
    # Sharpness-based selection
    if sharpness < 0.3:
        # Low sharpness - use rotation/scale invariant methods
        selected_methods.extend(['angular', 'jaccard', 'canberra'])
    else:
        # High sharpness - use precise methods
        selected_methods.extend(['cosine', 'pearson', 'dot_product'])
    
    # Brightness-based selection
    if brightness < 0.3 or brightness > 0.8:
        # Extreme brightness - use illumination robust methods
        selected_methods.extend(['jaccard', 'angular', 'manhattan'])
    else:
        # Normal brightness - use standard methods
        selected_methods.extend(['cosine', 'euclidean'])
    
    # Contrast-based selection
    if contrast < 0.3:
        # Low contrast - use binary/robust methods
        selected_methods.extend(['jaccard', 'canberra'])
    
    # Remove duplicates and return
    return list(dict.fromkeys(selected_methods))

class RobustFaceRecognitionSystem(FaceRecognitionSystem):
    def __init__(self, config: Dict):
        super().__init__(config)
        
        # Enhanced components
        self.multi_scale_processor = MultiScaleFaceProcessor(config)
        self.temporal_fusion = TemporalFusion(config)
        self.quality_assessor = FaceQualityAssessor(config)
        self.threshold_manager = AdaptiveThresholdManager(config)
        
        # Quality-adaptive engine
        self.similarity_engine = AdaptiveWeightSimilarityEngine(config)
        
        # Enhanced configuration
        self.robust_config = {
            'enable_multi_scale': config.get('enable_multi_scale', True),
            'enable_temporal_fusion': config.get('enable_temporal_fusion', True),
            'enable_quality_aware': config.get('enable_quality_aware', True),
            'min_face_quality': config.get('min_face_quality', 0.3),
            'temporal_buffer_size': config.get('temporal_buffer_size', 10),
        }
        
        # Initialize for statistics
        self.last_results = []
        
        print("🎯 Robust Face Recognition with BALANCED similarity engine")
              
    def process_frame_robust(self, frame: np.ndarray) -> List[Dict]:
        """Enhanced robust processing with quality-adaptive similarity"""
        start_time = time.time()
        results = []
        
        # Detect faces
        detections = self.detect_faces(frame)
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            
            # Extract face ROI with padding
            padding = self.config.get('roi_padding', 15)
            h, w = frame.shape[:2]
            x1_pad = max(0, x1 - padding)
            y1_pad = max(0, y1 - padding)
            x2_pad = min(w, x2 + padding)
            y2_pad = min(h, y2 + padding)
            
            face_roi = frame[y1_pad:y2_pad, x1_pad:x2_pad]
            
            # Skip if ROI is invalid
            if face_roi.size == 0 or face_roi.shape[0] < 20 or face_roi.shape[1] < 20:
                continue
                
            # Quality assessment
            quality_scores = self.quality_assessor.assess_face_quality(face_roi, detection['bbox'])
            
            # Skip very low quality faces entirely
            if not self.threshold_manager.should_process_face(quality_scores):
                if self.config.get('verbose', False):
                    print(f"⏭️ Skipping low quality face (score: {quality_scores.get('overall', 0):.2f})")
                continue
                
            # Mask detection
            mask_status, mask_confidence = self.detect_mask(face_roi)
            
            # Enhanced embedding extraction
            if self.robust_config['enable_multi_scale'] and quality_scores.get('overall', 0) > 0.3:
                embeddings = self.multi_scale_processor.extract_multi_scale_embeddings(face_roi)
                if embeddings:
                    embedding = self.multi_scale_processor.fuse_embeddings(embeddings)
                else:
                    embedding = self.extract_embedding(face_roi)
            else:
                embedding = self.extract_embedding(face_roi)
            
            if embedding is None:
                continue
            
            # QUALITY-ADAPTIVE RECOGNITION
            if self.robust_config['enable_quality_adaptive_similarity']:
                identity, recognition_confidence, detailed_scores = self.recognize_face_quality_adaptive(
                    embedding, quality_scores
                )
            else:
                # Fallback to original method
                identity, recognition_confidence = self.recognize_face(embedding)
                detailed_scores = {}
            
            # Generate track ID for temporal fusion
            track_id = self._generate_track_id(detection['bbox'])
            
            # Update temporal buffer
            if identity and self.robust_config['enable_temporal_fusion']:
                self.temporal_fusion.update_temporal_buffer(track_id, identity, recognition_confidence)
                
                # Get temporal consensus
                temporal_identity, temporal_confidence = self.temporal_fusion.get_temporal_consensus(track_id)
                
                if temporal_identity and temporal_confidence > recognition_confidence:
                    identity = temporal_identity
                    recognition_confidence = temporal_confidence
            
            # Adaptive threshold for display
            adaptive_threshold = self.threshold_manager.compute_adaptive_threshold(quality_scores)
            
            results.append({
                'bbox': detection['bbox'],
                'detection_confidence': detection['confidence'],
                'mask_status': mask_status,
                'mask_confidence': mask_confidence,
                'identity': identity,
                'recognition_confidence': recognition_confidence,
                'embedding': embedding.tolist(),
                'quality_scores': quality_scores,
                'adaptive_threshold': adaptive_threshold,
                'track_id': track_id,
                'detailed_scores': detailed_scores,
                'similarity_profile': self.similarity_engine.current_profile,
                'quality_adaptive': self.robust_config['enable_quality_adaptive_similarity']
            })
        
        # Update stats and store last results
        self._update_robust_stats(results, start_time)
        self.last_results = results  # Store for statistics
        return results

    def _update_robust_stats(self, results: List[Dict], start_time: float):
        """Update enhanced statistics"""
        self.debug_stats['total_frames_processed'] += 1
        self.debug_stats['total_faces_detected'] += len(results)
        self.debug_stats['total_faces_recognized'] += len([r for r in results if r['identity']])
        self.debug_stats['last_processing_time'] = (time.time() - start_time) * 1000
        
        # Quality statistics
        if results:
            avg_quality = np.mean([r.get('quality_scores', {}).get('overall', 0) for r in results])
            self.debug_stats.setdefault('avg_face_quality', deque(maxlen=50)).append(avg_quality)
       
    def recognize_face_quality_adaptive(self, embedding: np.ndarray, 
                                    quality_scores: Dict[str, float]) -> Tuple[Optional[str], float, Dict]:
        """Enhanced recognition with quality-adaptive similarity methods"""
        
        if not self.identity_centroids:
            return None, 0.0, {}
        
        # Use quality-adaptive similarity engine
        similarity_scores = self.similarity_engine.adaptive_weighted_similarity(
            embedding, self.identity_centroids
        )

        # Compute adaptive threshold based on quality
        adaptive_threshold = self.threshold_manager.compute_adaptive_threshold(quality_scores)
        
        # Find best match
        best_identity = None
        best_score = 0.0
        detailed_scores = {}
        
        for identity, score in similarity_scores.items():
            detailed_scores[identity] = score
            if score > best_score and score >= adaptive_threshold:
                best_score = score
                best_identity = identity
        
        # Debug output
        if self.config.get('verbose', False) and best_identity:
            profile = self.similarity_engine.current_profile
            print(f"✅ Recognized: {best_identity} (score: {best_score:.3f}, profile: {profile})")
        
        return best_identity, best_score, detailed_scores    
 
    def _generate_track_id(self, bbox: List[int]) -> int:
        """Generate simple track ID from bounding box position"""
        x1, y1, x2, y2 = bbox
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        return hash(f"{center_x}_{center_y}") % 1000000
         
    def get_balanced_stats(self) -> Dict:
        """Get balanced engine statistics"""
        stats = self.similarity_engine.get_engine_stats()
        
        # Add recognition statistics
        if hasattr(self, 'last_results'):
            total_faces = len(self.last_results)
            recognized_faces = len([r for r in self.last_results if r['identity']])
            stats['recognition_rate'] = recognized_faces / total_faces if total_faces > 0 else 0
        
        return stats         

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
    def __init__(self, face_system, processing_interval: int = 5, buffer_size: int = 3):
        self.face_system = face_system
        self.cap = None
        self.fps = 0
        self.frame_count = 0
        self.processing_count = 0
        self.start_time = time.time()
        self.config = face_system.config
        
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
        self.processing_lock = Lock()
        
        # RTSP configuration
        self.rtsp_url = None
        self.reconnect_delay = 5
        self.max_reconnect_attempts = 5
        
        # Enhanced display resizing
        self.resizer = DisplayResizer()
        self.show_resize_info = False
        self.original_frame_size = None
        
        # Processing resolution
        self.processing_width = 1000
        self.processing_height = 500
        self.processing_scale = 1.0
        
        # 🆕 CONTEXT-AWARE SCALING SYSTEM
        self.context_aware_scaler = ContextAwareDynamicScaling(self.config)
        self.enable_context_awareness = True
        self.context_debug_mode = False

        # Performance tracking
        self.performance_history = deque(maxlen=100)
        self.consecutive_poor_detections = 0
        self.consecutive_good_detections = 0
        self.adjustment_cooldown = 0

        # Resolution adjustment parameters
        self.min_processing_scale = 0.3
        self.max_processing_scale = 2.5
        self.current_processing_scale = 1.0
        self.scale_adjustment_step = 0.1
        
        # Performance thresholds
        self.target_detection_rate = 0.7
        self.target_face_size = 80
        self.min_face_size = 30
        
        # Detection quality tracking
        self.consecutive_poor_detections = 0
        self.consecutive_good_detections = 0
        self.adjustment_cooldown = 0
        
        print("🎯 Context-aware dynamic scaling ENABLED")
        self.face_system = face_system
        self.cap = None
        self.fps = 0
        self.frame_count = 0
        self.processing_count = 0
        self.start_time = time.time()
        self.config = face_system.config
        
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
        self.processing_lock = Lock()
        
        # RTSP configuration
        self.rtsp_url = None
        self.reconnect_delay = 5
        self.max_reconnect_attempts = 5
        
        # Enhanced display resizing 
        self.resizer = DisplayResizer()
        self.show_resize_info = False
        self.original_frame_size = None
        
        # Processing resolution - resize input stream for processing
        self.processing_width = 1280  # Default processing width
        self.processing_height = 720  # Default processing height
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
        self.max_history_size = 50

        # Add proper initialization for tracking attributes
        self.performance_history = []
        self.consecutive_poor_detections = 0
        self.consecutive_good_detections = 0
        self.adjustment_cooldown = 0
        self.log_file = None

        # Stream health monitoring
        self.consecutive_good_frames = 0
        self.last_reconnect_time = 0
        
        # Resolution adjustment parameters
        self.min_processing_scale = 0.5   # Minimum scale (50% of original)
        self.max_processing_scale = 3.0   # Maximum scale (150% of original)
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
        self.current_preset_index = 0
                
        print("🎮 Enhanced keyboard controls LOADED")   

        # Enhanced logging system
        self.logging_enabled = False
        self.audio_logging_enabled = False
        self.log_file = None
        self.log_start_time = None
        self.log_interval = 5 
        self.log_counter = 0

        # Logging columns configuration
        self.log_columns = [
            'timestamp', 
            'identity', 
            'mask_status'
        ]
        
        print("📊 Enhanced logging system READY - Face names + Mask Status")        
     
        
        # Initialize face tracker with proper parameters
        self.face_tracker = SimpleFaceTracker(
            confidence_frames=3,  # Reduced from 15 for more responsive tracking
            cooldown_seconds=5,   # Reduced from 11 for faster updates
            min_iou=0.3
        )
        
        # Enhanced Image Logging System
        self.image_logging_enabled = False
        self.image_log_folder = None
        self.image_log_interval = 3  # Save every 3rd frame with mask violations
        self.max_images_per_session = 500  # Prevent disk space issues
        self.saved_image_count = 0
        self.last_image_save_time = 0
        self.min_save_interval = 2.0  # Minimum seconds between saves
        
        # 🆕 VOICE ALERT SYSTEM
        self.alert_manager = DurationAwareAlertManager(self.config)
        self.sent_alerts = set()  # Track alerted identities to avoid duplicates
        
        print("🖼️  Enhanced image logging system READY") 
        print("🔊 Voice alert system READY")
            
    def perform_memory_cleanup(self):
        """Periodic memory cleanup"""
        if self.frame_count % 500 == 0:  # Every 500 frames
            # Clear various histories and buffers
            if len(self.performance_history) > 100:
                self.performance_history.clear()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Reset face system statistics periodically
            if hasattr(self, 'face_system'):
                self.face_system.reset_statistics_periodically()
            
            #print(f"🧹 Memory cleanup performed at frame {self.frame_count}")        
             
    def monitor_system_health(self):
        """Monitor system health and adjust accordingly"""
        if self.frame_count % 100 == 0:
            # Check if processing is falling behind
            processing_lag = time.time() - self.last_processed_time
            if processing_lag > 1.0:  # 1 second lag
                print(f"⚠️ System lagging: {processing_lag:.2f}s - reducing load")
                self.processing_interval = min(self.processing_interval + 2, 30)
                
            # Check memory usage
            try:
                import psutil
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                if memory_mb > 1050:  # 1050 MB threshold
                    print(f"⚠️ High memory usage: {memory_mb:.1f}MB - performing cleanup")
                    self.perform_memory_cleanup()
            except ImportError:
                pass             
             
    def toggle_context_awareness(self):
        """Toggle context-aware scaling"""
        self.enable_context_awareness = not self.enable_context_awareness
        status = "ENABLED" if self.enable_context_awareness else "DISABLED"
        print(f"🎯 Context-aware scaling: {status}")

    def print_context_statistics(self):
        """Print detailed context analysis statistics"""
        if not hasattr(self, 'context_aware_scaler'):
            print("❌ Context-aware scaling not available")
            return
        
        stats = self.context_aware_scaler.get_scaling_statistics()
        
        print("\n" + "="*60)
        print("📊 CONTEXT-AWARE SCALING STATISTICS")
        print("="*60)
        print(f"Current Scale: {stats['current_scale']:.2f}")
        print(f"Total Decisions: {stats['total_decisions']}")
        print(f"Context Influence: {stats['context_influence']:.0%}")
        
        if 'recent_context_stats' in stats and stats['recent_context_stats']:
            print(f"\n📈 Recent Context Analysis:")
            for metric, values in stats['recent_context_stats'].items():
                print(f"   {metric:20}: {values['mean']:.3f} ± {values['std']:.3f}")
        
        if 'common_recommendations' in stats:
            print(f"\n🎯 Common Recommendations:")
            for reason, count in stats['common_recommendations'].items():
                print(f"   {count:2}x {reason}")
        
        print("="*60)

    def update_dynamic_system(self):
        """Update dynamic adjustment system state"""
        if self.adjustment_cooldown > 0:
            self.adjustment_cooldown -= 1
        
        # Trim performance history
        if len(self.performance_history) > self.max_history_size:
            self.performance_history.pop(0)

    def enhanced_resize_for_processing(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame for processing using dynamic scale with error handling"""
        try:
            if self.current_processing_scale == 1.0:
                return frame
                
            h, w = frame.shape[:2]
            new_w = max(64, int(w * self.current_processing_scale))  # Minimum 64px
            new_h = max(64, int(h * self.current_processing_scale))  # Minimum 64px
            
            return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        except Exception as e:
            print(f"❌ Error in enhanced_resize_for_processing: {e}")
            return frame  # Fallback to original frame

    def scale_bbox_to_original(self, bbox: List[int], original_shape: Tuple[int, int], 
                            processed_shape: Tuple[int, int]) -> List[int]:
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

    def _prepare_display_results(self, results: List[Dict], original_frame: np.ndarray, 
                            display_frame: np.ndarray) -> List[Dict]:
        """Scale results to display coordinates"""
        if not results:
            return []
        
        original_h, original_w = original_frame.shape[:2]
        display_h, display_w = display_frame.shape[:2]
        
        display_results = []
        for result in results:
            display_bbox = self.scale_bbox_to_display(
                result['bbox'],
                (original_h, original_w),
                (display_h, display_w)
            )
            display_result = result.copy()
            display_result['bbox'] = display_bbox
            display_results.append(display_result)
        
        return display_results

    def scale_bbox_to_display(self, bbox: List[int], original_shape: Tuple[int, int], 
                            display_shape: Tuple[int, int]) -> List[int]:
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
        
    def check_and_send_alerts(self, results: List[Dict]) -> Dict:
        """Enhanced alert checking with detailed tracking information"""
        if not self.alert_manager.enabled:
            return {}
            
        # Detect current violations
        current_violations = []
        for result in results:
            if (result.get('mask_status') == 'no_mask' and 
                result.get('mask_confidence', 0) > 0.3):
                
                current_violations.append({
                    'identity': result.get('identity', 'Unknown'),
                    'mask_confidence': result.get('mask_confidence', 0),
                    'bbox': result['bbox']
                })
        
        # Use duration-aware alerting and get detailed results
        alert_triggered = self.alert_manager.trigger_duration_alert(current_violations, self.frame_count)
        
        # 🆕 Return detailed alert activity information
        alert_results = {
            'triggered_alerts': current_violations if alert_triggered else [],
            'cooldown_skipped': self._get_cooldown_identities(current_violations),
            'duration_tracking': self.alert_manager.get_duration_stats().get('violation_timers', {}),
            'alert_sent': alert_triggered
        }
        
        return alert_results
        
    def _get_cooldown_identities(self, violations: List[Dict]) -> List[str]:
        """Get identities that are in cooldown and skipped for alerts"""
        cooldown_identities = []
        current_time = time.time()
        
        for violation in violations:
            identity = violation.get('identity', 'Unknown')
            # Check if this identity was recently alerted
            if identity in self.alert_manager.alerted_identities:
                cooldown_identities.append(identity)
        
        return cooldown_identities    
         
    def _print_duration_debug(self, violations: List[Dict]):
        """Print duration tracking debug information"""
        for violation in violations[:2]:  # Only first 2 to avoid spam
            identity = violation.get('identity', 'Unknown')
            duration_info = self.alert_manager.get_violation_duration_info(identity)
            if duration_info['frames'] > 0:
                print(f"🔄 Tracking: {identity} - {duration_info['frames']} frames, {duration_info['seconds']:.1f}s")            
            
    def _send_violation_alert(self, violations: List[Dict]):
        """Send voice alert for mask violations"""
        # Count recognized vs unknown violators
        recognized = [v for v in violations if v['identity'] and v['identity'] != 'Unknown']
        unknown = [v for v in violations if not v['identity'] or v['identity'] == 'Unknown']
        
        if recognized:
            # Alert for recognized people by name
            names = [v['identity'] for v in recognized]
            if len(names) == 1:
                message = f"Perhatian, {names[0]} tidak memakai masker"
            else:
                name_list = " dan ".join(names)
                message = f"Perhatian, {name_list} tidak memakai masker"
        else:
            # Alert for unknown people
            count = len(unknown)
            if count == 1:
                message = "Perhatian, satu orang tidak dikenal tidak memakai masker"
            else:
                message = f"Perhatian, {count} orang tidak dikenal tidak memakai masker"
        
        # Add urgency for multiple violations
        total_violations = len(violations)
        if total_violations > 2:
            message += ". Situasi darurat!"
        
        # Send the alert
        success = self.alert_manager.send_voice_alert(message)
        
        if success:
            print(f"🚨 Alert triggered: {message}")
            print(f"   - Recognized violators: {len(recognized)}")
            print(f"   - Unknown violators: {len(unknown)}")        
                   
    # ADD to RealTimeProcessor class
    def get_frame_for_processing(self) -> Optional[np.ndarray]:
        """Thread-safe frame acquisition with timeout and validation"""
        try:
            # Quick check without lock first
            if self.frame_queue.empty():
                return None
                
            with self.processing_lock:  # PREVENTS parallel processing
                frame = self.frame_queue.get(block=True, timeout=0.05)  # Short timeout
                
                # Validate frame integrity
                if (frame is None or frame.size == 0 or 
                    frame.shape[0] < 10 or frame.shape[1] < 10):
                    return None
                    
                self.consecutive_good_frames += 1
                return frame
                
        except Exception as e:
            if self.config.get('verbose', False):
                print(f"Frame acquisition skipped: {e}")
            return None        
            
    def setup_image_logging(self, csv_filename: str = None):
        """Setup image logging folder structure matching CSV filename"""
        try:
            if csv_filename is None:
                # Generate matching filename with CSV
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_filename = f"face_recognition_detailed_{timestamp}.csv"
            
            # Extract base name without extension
            base_name = Path(csv_filename).stem
            self.image_log_folder = Path(f"{base_name}_images")
            
            # Create directory
            self.image_log_folder.mkdir(exist_ok=True)
            
            print(f"🖼️  Image logging ENABLED: {self.image_log_folder}")
            print(f"   - Format: jpeg annotated frames")
            print(f"   - Trigger: Mask violations only")
            print(f"   - Max images: {self.max_images_per_session}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to setup image logging: {e}")
            self.image_logging_enabled = False
            self.image_log_folder = None
            return False        
        
    def has_mask_violations(self, results: List[Dict]) -> bool:
        """Check if frame contains mask violations - INCLUDES UNKNOWN PEOPLE"""
        if not results:
            return False
        
        for result in results:
            mask_status = result.get('mask_status')
            mask_conf = result.get('mask_confidence', 0)
            
            # Log ANY person without mask, regardless of recognition status
            # Lowered confidence threshold from 0.5 to 0.3 to be less conservative
            if mask_status == 'no_mask' and mask_conf > 0.3:
                return True
        
        return False       
            
    def save_annotated_frame(self, frame: np.ndarray, results: List[Dict], original_frame: np.ndarray = None):
        """Save annotated frame with bounding boxes, labels, and enhanced metadata overlay"""
        if not self.image_logging_enabled or not self.image_log_folder:
            return False
        
        # Check limits
        if self.saved_image_count >= self.max_images_per_session:
            print("🖼️  Image log limit reached, disabling image logging")
            self.image_logging_enabled = False
            return False
        
        # Rate limiting
        current_time = time.time()
        if current_time - self.last_image_save_time < self.min_save_interval:
            return False
        
        try:
            # Create a copy to draw on (use original frame if available for better quality)
            if original_frame is not None:
                save_frame = original_frame.copy()
            else:
                save_frame = frame.copy()
            
            # Draw bounding boxes and labels on the saved frame
            for result in results:
                x1, y1, x2, y2 = result['bbox']
                identity = result['identity']
                rec_conf = result.get('recognition_confidence', 0)
                det_conf = result.get('detection_confidence', 0)
                mask_status = result.get('mask_status', 'unknown')  
                mask_conf = result.get('mask_confidence', 0.0)
                
                # Color coding based on mask status and recognition (same as display)
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
                cv2.rectangle(save_frame, (x1, y1), (x2, y2), color, 3)
                
                # Prepare label with comprehensive information
                if identity:
                    base_label = f"{identity} (Rec:{rec_conf:.2f})"
                else:
                    base_label = f"Unknown (Det:{det_conf:.2f})"
                
                # Add mask status to label
                mask_label = f" | Mask: {mask_status}({mask_conf:.2f})"
                full_label = base_label + mask_label
                
                # Draw label background
                label_size = cv2.getTextSize(full_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(save_frame, (x1, y1 - label_size[1] - 15), 
                            (x1 + label_size[0], y1), color, -1)
                
                # Draw label text
                cv2.putText(save_frame, full_label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                
                # Draw mask status indicator
                status_text = f"Mask: {mask_status.upper()}"
                status_color = (0, 255, 0) if mask_status == "mask" else (0, 0, 255)
                cv2.putText(save_frame, status_text, (x1, y2 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            
            # Enhanced metadata overlay
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            filename_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            
            # Add comprehensive header
            header_text = f"MASK VIOLATION - {timestamp}"
            cv2.putText(save_frame, header_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            
            # Add system info
            fps_text = f"FPS: {self.fps:.1f} | Scale: {self.current_processing_scale:.2f}"
            cv2.putText(save_frame, fps_text, (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add violation summary
            violations = [r for r in results if r.get('mask_status') == 'no_mask']
            violation_text = f"Violations: {len(violations)} | Total Faces: {len(results)}"
            cv2.putText(save_frame, violation_text, (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Add resolution info
            h, w = save_frame.shape[:2]
            res_text = f"Resolution: {w}x{h}"
            cv2.putText(save_frame, res_text, (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw violation counter on top right
            counter_text = f"#{self.saved_image_count + 1:04d}"
            counter_size = cv2.getTextSize(counter_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
            cv2.putText(save_frame, counter_text, 
                    (w - counter_size[0] - 20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            
            # Generate filename with violation count
            filename = f"violation_{filename_time}_{self.saved_image_count + 1:04d}.jpeg"
            filepath = self.image_log_folder / filename
            
            # Save as high-quality JPEG
            success = cv2.imwrite(str(filepath), save_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            if success:
                self.saved_image_count += 1
                self.last_image_save_time = current_time
                
                # Print detailed save information
                print(f"🖼️  Saved violation image #{self.saved_image_count}: {filename}")
                print(f"   - Faces: {len(results)}, Violations: {len(violations)}")
                print(f"   - Resolution: {w}x{h}")
                print(f"   - Path: {filepath}")
                
                return True
            else:
                print(f"❌ Failed to save image: {filepath}")
                return False
                
        except Exception as e:
            print(f"❌ Error saving annotated frame: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    def analyze_detection_performance(self, results: List[Dict], original_frame_shape: Tuple[int, int]) -> Dict:
        """Comprehensive analysis of detection performance for dynamic adjustment"""
        performance = {
            'detection_count': len(results),
            'face_sizes': [],
            'detection_confidences': [],
            'recognition_rates': [],
            'avg_face_size': 0,
            'detection_quality': 0,
            'needs_adjustment': False,
            'adjustment_direction': 0
        }
        
        if not results:
            performance['detection_quality'] = 0
            performance['needs_adjustment'] = True
            performance['adjustment_direction'] = 1
            return performance
        
        # Analyze each detection using ORIGINAL frame coordinates
        orig_h, orig_w = original_frame_shape
        frame_diagonal = np.sqrt(orig_h**2 + orig_w**2)
        
        for result in results:
            x1, y1, x2, y2 = result['bbox']
            face_width = x2 - x1
            face_height = y2 - y1
            face_size = min(face_width, face_height)
            
            # Normalize face size relative to frame size
            normalized_face_size = face_size / frame_diagonal * 1000  # Scale factor
            
            performance['face_sizes'].append(normalized_face_size)
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
        """Resize frame for processing using dynamic scale with error handling"""
        try:
            if self.current_processing_scale == 1.0:
                return frame
                
            h, w = frame.shape[:2]
            new_w = max(64, int(w * self.current_processing_scale))  # Minimum 64px
            new_h = max(64, int(h * self.current_processing_scale))  # Minimum 64px
            
            return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        except Exception as e:
            print(f"❌ Error in enhanced_resize_for_processing: {e}")
            return frame  # Fallback to original frame

    def update_dynamic_system(self):
        """Update dynamic adjustment system state"""
        if self.adjustment_cooldown > 0:
            self.adjustment_cooldown -= 1
        
        # Trim performance history
        if len(self.performance_history) > self.max_history_size:
            self.performance_history.pop(0)           

    # REPLACE the _capture_frames method in RealTimeProcessor
    def _capture_frames(self):
        """Enhanced stable frame capture with memory protection"""
        reconnect_attempts = 0
        max_queue_size = 2  # Conservative limit
        
        while self.running:
            try:
                ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    print("⚠️  Frame capture failed, attempting to reconnect...")
                    reconnect_attempts += 1
                    
                    if reconnect_attempts >= self.max_reconnect_attempts:
                        print("❌ Max reconnection attempts reached")
                        break
                    
                    time.sleep(self.reconnect_delay)
                    self._reconnect_stream()
                    continue
                
                reconnect_attempts = 0
                
                # ENHANCED: Frame validation
                if (frame.size == 0 or frame.shape[0] < 50 or frame.shape[1] < 50 or
                    np.mean(frame) < 10 or np.mean(frame) > 250):  # Basic corruption check
                    continue
                
                with self.frame_lock:
                    self.latest_frame = frame
                
                # ENHANCED: Conservative queue management
                if self.frame_queue.qsize() >= max_queue_size:
                    try:
                        self.frame_queue.get_nowait()  # Discard oldest frame
                    except:
                        pass
                        
                try:
                    self.frame_queue.put(frame, block=False, timeout=0.01)
                except:
                    pass  # Skip frame if queue is full
                            
            except Exception as e:
                print(f"🚨 Capture thread error: {e}")
                time.sleep(0.1)  # Shorter sleep for faster recovery
                                
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

    def toggle_logging(self, filename: str = None):
        """Toggle both CSV and image logging with coordinated setup"""
        if not self.logging_enabled:
            # Enable both CSV and image logging
            self.setup_logging(filename)
            self.setup_image_logging(self.log_file)
            self.logging_enabled = True
            self.image_logging_enabled = True
            self.log_counter = 0
            self.saved_image_count = 0
            
            # 🆕 Enable audio alert logging
            self.audio_logging_enabled = True
            print("🟢 Enhanced logging STARTED")
            print("   - CSV: timestamp, identity, mask_status")
            print("   - Images: jpeg frames for mask violations")
            print("   - Audio: voice alert logging with retry")
            print(f"   - Retry: {self.alert_manager.max_retries} attempts")
            print(f"   - Image folder: {self.image_log_folder}")
            
        else:
            # Disable both
            if self.log_file:
                duration = datetime.datetime.now() - self.log_start_time
                print(f"🔴 Logging STOPPED: {self.log_file}")
                print(f"   - Duration: {duration}")
                print(f"   - CSV entries: {self.log_counter}")
                print(f"   - Violation images: {self.saved_image_count}")
            
            # 🆕 Disable audio alert logging
            self.audio_logging_enabled = False
            self.logging_enabled = False
            self.image_logging_enabled = False
            self.log_file = None
            self.image_log_folder = None
            self.log_start_time = None
                            
                                     
    def collect_log_data(self, results: List[Dict]) -> List[Dict]:
        """Collect individual face recognition and mask status data - FIXED"""
        log_entries = []
        
        # FIX: Check if results is a list of dictionaries
        if not isinstance(results, list):
            print(f"❌ Logging error: Expected list, got {type(results)}")
            return log_entries
        
        for result in results:
            # FIX: Proper dictionary access
            if isinstance(result, dict):
                identity = result.get('identity')
                mask_status = result.get('mask_status', 'unknown')
                
                # Only log recognized faces with valid identities
                if identity is not None and identity != "Unknown":
                    log_entries.append({
                        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                        'identity': str(identity),  # Ensure it's a string
                        'mask_status': str(mask_status)  # Ensure it's a string
                    })
            else:
                print(f"❌ Skipping non-dictionary result: {type(result)}")
        
        return log_entries
        
    def write_log_entries(self, log_entries: List[Dict]):
        """Write multiple log entries to CSV - FIXED"""
        if not self.logging_enabled or not self.log_file or not log_entries:
            return
        
        try:
            with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                for log_data in log_entries:
                    # Write ONLY the 3 required columns for each recognized face
                    row = [
                        log_data['timestamp'],
                        log_data['identity'],
                        log_data['mask_status']
                    ]
                    writer.writerow(row)
                    
            self.log_counter += len(log_entries)
            
            # Periodic status update
            if self.log_counter % 10 == 0:
                print(f"📊 Logged {self.log_counter} face entries")
                
        except Exception as e:
            print(f"❌ Log write error: {e}")
                                        
    def setup_logging(self, filename: str = None):
        """Setup CSV logging with face names, mask status, and audio alerts"""
        try:
            if filename is None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"face_recognition_detailed_{timestamp}.csv"
            
            self.log_file = filename
            self.log_start_time = datetime.datetime.now()
            
            # 🆕 Enhanced header with audio alert support
            enhanced_columns = [
                'timestamp', 
                'identity', 
                'mask_status',
                'alert_type',    # 🆕 New column
                'alert_status'   # 🆕 New column
            ]
            
            with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(enhanced_columns)
            
            print(f"📊 Enhanced logging ENABLED: {filename}")
            print(f"   - Columns: {enhanced_columns}")
            print(f"   - Logging: Recognized faces + mask status + audio alerts")
            
        except Exception as e:
            print(f"❌ Failed to setup logging: {e}")
            self.logging_enabled = False
            self.log_file = None
            
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
        """Draw comprehensive debug information on frame"""
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
            ]
        
        # Debug information
        debug_lines = []
        if self.debug_mode:
            debug_lines = [
                f"Detection: {stats['avg_detection_time']:.1f}ms",
                f"Embedding: {stats['avg_embedding_time']:.1f}ms",
                f"Recognition: {stats['avg_recognition_time']:.1f}ms",
                f"Total Faces: {stats['total_faces_detected']}",
                f"Recognized: {stats['total_faces_recognized']}",
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
        """Enhanced visualization with mask status"""
        if self.original_frame_size is None:
            self.original_frame_size = frame.shape[:2]
        
        # Draw bounding boxes and labels
        for result in results:
            x1, y1, x2, y2 = result['bbox']
            identity = result['identity']
            rec_conf = result['recognition_confidence']
            det_conf = result['detection_confidence']
            mask_status = result.get('mask_status', 'unknown')  
            mask_conf = result.get('mask_confidence', 0.0)  
            
            # Color coding based on mask status and recognition
            if identity:
                if mask_status == "mask":
                    color = (0, 255, 0)  # Yellow for recognized without mask
                else:
                    color = (0, 255, 255)    # Green for recognized with mask
            else:
                if mask_status == "mask":
                    color = (255, 255, 0)  # Cyan for unknown with mask
                else:
                    color = (0, 0, 255)    # Red for unknown without mask
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label with mask information
            if identity:
                base_label = f"{identity} ({rec_conf:.2f})"
            else:
                base_label = f"Unknown ({det_conf:.2f})"
            
            # Add mask status to label
            mask_label = f" | Mask: {mask_status}({mask_conf:.2f})"
            full_label = base_label + mask_label
            
            # Draw label background
            label_size = cv2.getTextSize(full_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                        (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(frame, full_label, (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            
        
        # Draw debug information
        self.draw_debug_info(frame, results)
        self.draw_detection_debug(frame, results)
        self.draw_resize_info(frame)

    def draw_mask_debug_info(self, frame: np.ndarray, results: List[Dict]):
        """Draw mask detection debug information"""
        if not self.debug_mode:
            return
            
        for i, result in enumerate(results):
            x1, y1, x2, y2 = result['bbox']
            mask_status = result.get('mask_status', 'unknown')
            mask_conf = result.get('mask_confidence', 0.0)
            
            # Draw mask status above bounding box
            status_text = f"Mask: {mask_status}({mask_conf:.2f})"
            
            # Calculate position (above the bounding box)
            text_y = y1 - 35
            
            # Draw background for text
            text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(frame, (x1, text_y - text_size[1] - 5), 
                        (x1 + text_size[0], text_y + 5), (0, 0, 0), -1)
            
            # Color code based on mask status
            if mask_status == "mask":
                color = (0, 255, 0)  # Green
            elif mask_status == "no_mask":
                color = (0, 0, 255)  # Red
            else:
                color = (255, 255, 0)  # Yellow
                
            # Draw mask status text
            cv2.putText(frame, status_text, (x1, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
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
        """Draw results with dynamic adjustment and mask debug information"""
        # Existing drawing logic
        self.draw_results(frame, results)
        
        # Add mask debug info
        self.draw_mask_debug_info(frame, results)
        
        # Add dynamic adjustment info if available
        if performance and self.show_performance_stats:
            self.draw_dynamic_adjustment_info(frame, performance)
            
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
            
        elif key == ord(';'):  # Change log interval
            old_interval = self.log_interval
            self.log_interval = max(1, (self.log_interval % 10) + 1)  # Cycle 1-10
            print(f"📊 Log interval: 1/{old_interval} → 1/{self.log_interval}")
            
        elif key == ord(':'):  # Print log status
            self.print_log_status()
            
        # Add to handle_key_controls method
        elif key == ord('Q'):  # Toggle quality-adaptive similarity
            self.face_system.robust_config['enable_quality_adaptive_similarity'] = \
                not self.face_system.robust_config['enable_quality_adaptive_similarity']
            status = "ENABLED" if self.face_system.robust_config['enable_quality_adaptive_similarity'] else "DISABLED"
            print(f"🎯 Quality-adaptive similarity: {status}")
            
        elif key == ord('C'):  # Toggle context awareness
            self.toggle_context_awareness()
            
        elif key == ord('X'):  # Print context statistics
            self.print_context_statistics()
            
        elif key == ord('Z'):  # Toggle context debug display
            self.context_debug_mode = not self.context_debug_mode
            status = "ON" if self.context_debug_mode else "OFF"
            print(f"🔍 Context debug display: {status}")                        
            
        elif key == ord('s'):
            # Save current frame
            timestamp = int(time.time())
            filename = f'captured_frame_{timestamp}.jpg'
            cv2.imwrite(filename, display_frame)
            print(f"💾 Frame saved: {filename}")
                
        elif key == ord('D'):  # Print duration tracking statistics
            stats = self.alert_manager.get_duration_stats()
            print("\n" + "="*50)
            print("📊 DURATION TRACKING STATISTICS")
            print("="*50)
            print(f"Tracked Identities: {stats['total_tracked_identities']}")
            print(f"Alerted Identities: {len(stats['alerted_identities'])}")
            
            for identity, timer_info in stats['violation_timers'].items():
                status = "✅ ALERTED" if identity in stats['alerted_identities'] else "🔄 TRACKING"
                print(f"  {identity}: {timer_info['frame_count']} frames, "
                    f"{timer_info['duration_seconds']:.1f}s - {status}")
            print("="*50)
            
        elif key == ord('R'):  # Reset all duration tracking
            self.alert_manager.violation_timers.clear()
            self.alert_manager.alerted_identities.clear()
            print("🔄 Duration tracking reset")
            
        elif key == ord('F'):  # Adjust frame threshold
            current = self.alert_manager.min_violation_frames
            new_threshold = max(5, (current + 5) % 60)  # Cycle 5-55 frames
            self.alert_manager.min_violation_frames = new_threshold
            print(f"🎯 Frame threshold: {current} → {new_threshold} frames")            
            
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
            
        
        elif key == ord('W'):  # Print balanced engine statistics
            if hasattr(self.face_system, 'get_balanced_stats'):
                stats = self.face_system.get_balanced_stats()
                print("\n" + "="*50)
                print("⚖️ BALANCED SIMILARITY ENGINE STATISTICS")
                print("="*50)
                print(f"Engine Type: {stats.get('engine_type', 'N/A')}")
                print(f"Active Methods: {stats.get('active_methods', [])}")
                print(f"Total Methods: {stats.get('total_methods', 0)}")
                print(f"Fixed Weights: {stats.get('fixed_weights', {})}")
                if 'recognition_rate' in stats:
                    print(f"Recognition Rate: {stats['recognition_rate']:.1%}")
                print("="*50)            
            
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
            
        elif key == ord('y'):  # Stability report
            self.print_stability_report()
            
        # 🆕 VOICE ALERT CONTROLS
        elif key == ord('V'):  # Toggle voice alerts
            self.alert_manager.toggle_alerts()           
            
        elif key == ord('9'):  # Test voice alert
            test_message = "Test suara dari sistem pengawasan masker"
            success = self.alert_manager.send_voice_alert(test_message)
            if success:
                print(f"🔊 Test alert sent: {test_message}")
            else:
                print("⏰ Test alert skipped - in cooldown period")
            
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
            
    def print_log_status(self):
        """Print current logging status - FIXED"""
        status = "🟢 ENABLED" if self.logging_enabled else "🔴 DISABLED"
        print(f"\n📊 LOGGING STATUS: {status}")
        
        if self.logging_enabled:
            duration = datetime.datetime.now() - self.log_start_time  # datetime.datetime
            print(f"   File: {self.log_file}")
            print(f"   Entries: {self.log_counter}")
            print(f"   Duration: {duration}")
            print(f"   Interval: Every {self.log_interval} processed frames")
            print(f"   Columns: timestamp, identity, mask_status")
        else:
            print("   Use 'l' to enable detailed face logging")

    def _prepare_display_results(self, results: List[Dict], original_frame: np.ndarray, display_frame: np.ndarray) -> List[Dict]:
        """Scale results to display coordinates"""
        if not results:
            return []
        
        original_h, original_w = original_frame.shape[:2]
        display_h, display_w = display_frame.shape[:2]
        
        display_results = []
        for result in results:
            display_bbox = self.scale_bbox_to_display(
                result['bbox'],
                (original_h, original_w),
                (display_h, display_w)
            )
            display_result = result.copy()
            display_result['bbox'] = display_bbox
            display_results.append(display_result)
        
        return display_results
            
    def debug_logging_flow(self, results: List[Dict]):
        """Enhanced debug why images aren't being logged"""
        if not self.logging_enabled:
            print("🔴 LOGGING DISABLED - Press 'l' to enable")
            return
        
        if not self.image_logging_enabled:
            print("🔴 IMAGE LOGGING DISABLED")
            return
        
        print(f"🔄 Processing Count: {self.processing_count}, Log Interval: {self.log_interval}, Image Interval: {self.image_log_interval}")
        print(f"🖼️  Saved so far: {self.saved_image_count}/{self.max_images_per_session}")
        
        # Check if we're due for logging
        if self.processing_count % self.log_interval != 0:
            print(f"⏰ Skipping - not due for logging (count % interval != 0)")
            return
        
        if not results:
            print("❌ No face detection results")
            return
        
        print(f"📊 Found {len(results)} face(s) in frame")
        
        has_violation = False
        for i, result in enumerate(results):
            identity = result.get('identity', 'Unknown')
            mask_status = result.get('mask_status', 'unknown')
            mask_conf = result.get('mask_confidence', 0)
            
            print(f"  Face {i+1}: {identity}, Mask: {mask_status}({mask_conf:.2f})")
            
            # Check violation conditions
            is_violation = (mask_status == 'no_mask' and mask_conf > 0.3)  # Lowered threshold
            is_recognized = (identity is not None and identity != "Unknown")
            
            if is_violation:
                has_violation = True
                if is_recognized:
                    print(f"  ✅ VIOLATION DETECTED: {identity} without mask!")
                else:
                    print(f"  ✅ VIOLATION DETECTED: Unknown person without mask!")
            else:
                if mask_status == 'mask':
                    print(f"  ✅ Mask worn properly")
                elif mask_status == 'unknown':
                    print(f"  ❓ Unknown mask status")
                else:
                    print(f"  ❓ Low mask confidence: {mask_conf:.2f}")
        
        # Check the actual violation method
        has_violations_method = self.has_mask_violations(results)
        print(f"🎯 has_mask_violations() returned: {has_violations_method}")
        
        # Check image logging conditions
        if has_violations_method:
            current_time = time.time()
            time_since_last = current_time - self.last_image_save_time
            due_for_image = (self.processing_count % self.image_log_interval == 0)
            within_limits = (self.saved_image_count < self.max_images_per_session)
            time_ok = (time_since_last >= self.min_save_interval)
            
            print(f"📸 Image logging conditions:")
            print(f"   - Due for image: {due_for_image} (count % {self.image_log_interval} == 0)")
            print(f"   - Within limits: {within_limits} ({self.saved_image_count}/{self.max_images_per_session})")
            print(f"   - Time OK: {time_ok} ({time_since_last:.1f}s >= {self.min_save_interval}s)")
            
            if due_for_image and within_limits and time_ok:
                print("🎯 ALL CONDITIONS MET - Image should be saved!")
            else:
                print("⏰ Some conditions not met for image saving")
                            
    def log_performance_data(self, results: List[Dict], display_frame: np.ndarray = None, original_frame: np.ndarray = None):
        """Enhanced logging with voice alerts and retry tracking"""
        if not self.logging_enabled:
            return
        
        # Only log every X processed frames to reduce I/O
        if self.processing_count % self.log_interval != 0:
            return
        
        print(f"\n=== LOGGING DEBUG Frame {self.processing_count} ===")
        self.debug_logging_flow(results)
        
        # 🆕 ENHANCED: CHECK FOR MASK VIOLATIONS AND SEND ALERTS WITH DETAILED LOGGING
        alert_results = self.check_and_send_alerts(results)
        
        # 🆕 LOG ALERT ACTIVITY
        if alert_results:
            print(f"🎯 ALERT ACTIVITY SUMMARY:")
            print(f"   - Triggered alerts: {len(alert_results['triggered_alerts'])}")
            print(f"   - Cooldown skipped: {len(alert_results['cooldown_skipped'])}")
            print(f"   - Duration tracking: {len(alert_results['duration_tracking'])} identities")
            
            # Show detailed alert information
            for alert in alert_results['triggered_alerts']:
                identity = alert.get('identity', 'Unknown')
                duration_info = self.alert_manager.get_violation_duration_info(identity)
                print(f"   🔊 ALERTING: {identity} - {duration_info['frames']} frames, {duration_info['seconds']:.1f}s")
                
            # Show identities in cooldown
            if alert_results['cooldown_skipped']:
                print(f"   ⏰ COOLDOWN: {len(alert_results['cooldown_skipped'])} identities waiting")
                for identity in list(alert_results['cooldown_skipped'])[:3]:  # Show first 3
                    print(f"      - {identity}")
                    
            # Show duration tracking status
            if alert_results['duration_tracking']:
                print(f"   📊 DURATION TRACKING:")
                for identity, timer in list(alert_results['duration_tracking'].items())[:3]:  # Show first 3
                    status = "ALERTED" if identity in self.alert_manager.alerted_identities else "TRACKING"
                    print(f"      - {identity}: {timer['frame_count']} frames, {timer['duration_seconds']:.1f}s - {status}")
        else:
            print(f"🎯 ALERT ACTIVITY: No alerts triggered this cycle")
        
        try:
            # CSV logging: Write entries for recognized faces
            log_entries = self.collect_log_data(results)
            
            # Image logging: Save frame if mask violations detected (ANY person)
            if (self.image_logging_enabled and 
                self.has_mask_violations(results)):
                
                print("🚨 ATTEMPTING TO SAVE VIOLATION IMAGE")
                
                # Check if we're due for image logging based on interval
                if (self.processing_count % self.image_log_interval == 0 and
                    self.saved_image_count < self.max_images_per_session):
                    
                    current_time = time.time()
                    # Check minimum time between saves
                    if current_time - self.last_image_save_time >= self.min_save_interval:
                        # Use original_frame if available, otherwise use display_frame
                        frame_to_save = original_frame if original_frame is not None else display_frame
                        if frame_to_save is not None:
                            success = self.save_annotated_frame(display_frame, results, frame_to_save)
                            if success:
                                print(f"✅ Image saved successfully! Total: {self.saved_image_count}")
                            else:
                                print("❌ Failed to save image")
                        else:
                            print("❌ No frame available for saving")
                    else:
                        print(f"⏰ Image save skipped - too soon since last save: {current_time - self.last_image_save_time:.1f}s")
                else:
                    print(f"⏰ Image save skipped - interval or limit: count={self.saved_image_count}, interval={self.image_log_interval}")
            else:
                print("ℹ️  No image saved - conditions not met")
            
            # CSV logging
            if log_entries:
                self.write_log_entries(log_entries)
                print(f"📝 CSV: Logged {len(log_entries)} face entries")
            else:
                print("📝 CSV: No recognized faces to log")
                
            print("=== END DEBUG ===\n")
                    
        except Exception as e:
            print(f"❌ Enhanced logging error: {e}")                                                               
    def run(self, source: str = "0"):
        """Main loop with memory management"""
        try:
            self.initialize_stream(source)
            self.start_frame_capture()
            
            print("🎮 Starting with MEMORY MANAGEMENT SYSTEM")
            self.print_control_reference()
            
            last_results = []
            last_performance = {}
            
            while self.running:
                # System health monitoring
                self.monitor_system_health()
                
                # Use stable frame acquisition
                original_frame = self.get_frame_for_processing()
                if original_frame is None:
                    time.sleep(0.03)  # Increased from 0.005 to 0.03 (33ms ~ 30fps)
                    continue
                
                # Skip frames if queue is backing up
                if self.frame_queue.qsize() > 2:
                    continue
                
                # Periodic memory cleanup
                if self.frame_count % 500 == 0:
                    self.perform_memory_cleanup()
                
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
                
                if should_process:
                    # Processing is already protected by processing_lock in get_frame_for_processing
                    raw_results = self.face_system.process_frame(processing_frame)
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

                    # Enhanced logging with image support
                    self.log_performance_data(scaled_results, display_frame, original_frame) # Pass display_frame for image logging
                    
                    last_results = scaled_results
                    self.processing_count += 1
                    
                    # Dynamic adjustment (protected by processing lock)
                    if self.dynamic_adjustment_enabled and self.frame_count % self.adaptive_check_interval == 0:
                        performance = self.analyze_detection_performance(scaled_results, (original_h, original_w))
                        self.performance_history.append(performance)
                        last_performance = performance
                        self.apply_dynamic_adjustment(performance)
                
                # Use cached results if not processing this frame
                display_results = self._prepare_display_results(last_results, original_frame, display_frame)
                
                # Enhanced drawing
                self.draw_enhanced_results(display_frame, display_results, last_performance)
                cv2.imshow('Dynamic Face Recognition System', display_frame)
                
                # Handle key controls
                key = cv2.waitKey(1) & 0xFF
                self.handle_key_controls(key, display_frame)
                            
        except Exception as e:
            print(f"❌ Error in main loop: {e}")
        finally:
            self.stop()
                                                                
    def stop(self):
        """Cleanup resources with image logging summary"""
        # Print final log summary
        if self.logging_enabled and self.log_file:
            duration = datetime.datetime.now() - self.log_start_time
            print(f"\n📊 ENHANCED LOGGING SUMMARY:")
            print(f"   CSV entries: {self.log_counter}")
            print(f"   Violation images: {self.saved_image_count}")
            print(f"   Duration: {duration}")
            print(f"   CSV file: {self.log_file}")
            print(f"   Image folder: {self.image_log_folder}")
        
        self.logging_enabled = False
        self.image_logging_enabled = False
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
        
    # ADD to RealTimeProcessor class
    def get_stability_metrics(self) -> Dict:
        """Monitor system stability metrics"""
        queue_health = "HEALTHY"
        if self.frame_queue.qsize() >= self.frame_queue.maxsize:
            queue_health = "FULL"
        elif self.frame_queue.qsize() == 0:
            queue_health = "EMPTY"
        
        return {
            'frame_queue_health': queue_health,
            'queue_size': self.frame_queue.qsize(),
            'consecutive_good_frames': self.consecutive_good_frames,
            'processing_lock_held': self.processing_lock.locked(),
            'memory_usage_mb': self.face_system.get_debug_stats().get('memory_mb', 0),
            'performance_history_size': len(self.performance_history)
        }

    def print_stability_report(self):
        """Print current stability status"""
        metrics = self.get_stability_metrics()
        print("\n" + "="*50)
        print("📊 STABILITY REPORT - Phase 1")
        print("="*50)
        print(f"Frame Queue: {metrics['queue_size']}/3 ({metrics['frame_queue_health']})")
        print(f"Consecutive Good Frames: {metrics['consecutive_good_frames']}")
        print(f"Processing Lock: {'LOCKED' if metrics['processing_lock_held'] else 'FREE'}")
        print(f"Memory Usage: {metrics['memory_usage_mb']:.1f} MB")
        print(f"Performance History: {metrics['performance_history_size']} entries")
        print("="*50)        
            
    def print_control_reference(self):
        """Print comprehensive control reference with alerts"""
        print("\n" + "="*60)
        print("🎮 ENHANCED KEYBOARD CONTROLS (WITH VOICE ALERTS)")
        print("="*60)
        print("🎯 CORE CONTROLS:")
        print("  'q' - Quit application")
        print("  's' - Save current frame")
        print("  'r' - Reset processing counters")
        print("  'x' - Print detailed statistics")
        
        print("\n⏱️  PROCESSING CONTROLS:")
        print("  '+' - Increase processing interval (process less)")
        print("  '-' - Decrease processing interval (process more)")
        print("  '.' - Large interval increase")
        print("  ',' - Large interval decrease")
        print("  'w' - Decrease processing resolution")
        print("  'e' - Increase processing resolution")
        
        print("\n🎯 DYNAMIC ADJUSTMENT CONTROLS:")
        print("  'a' - Toggle dynamic adjustment")
        print("  'z' - Reset dynamic scaling to 1.0")
        print("  'c' - Manually increase processing scale")
        print("  'v' - Manually decrease processing scale")
        print("  'V' - Print quality-adaptive statistics")    
        print("  'n' - Toggle fixed/dynamic processing")
        print("  'm' - Cycle processing presets")
        
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
        print("  'P' - Print quality-adaptive statistics")    
        print("  'b' - Toggle detection debug")
        print("  'f' - Toggle save debug frames")
        
        print("\n📊 ADVANCED CONTROLS:")
        print("  't' - Toggle face tracking")
        print("  'k' - Take annotated snapshot")
        
        print("\n🔊 VOICE ALERT CONTROLS:")
        print("  'v' - Toggle voice alerts")
        print("  '9' - Test voice alert")
        print(f"  Server: {self.config.get('alert_server_url', 'Not configured')}")
        
        print("\n📊 ENHANCED LOGGING SYSTEM:")
        print("  'l' - Toggle CSV + Image + Audio logging")
        print("  ';' - Change log interval (1-10 frames)")
        print("  ':' - Print current log status")
        print("  Features:")
        print("    - CSV: timestamp, identity, mask_status")
        print("    - Images: jpeg frames for mask violations")
        print("    - Organized folder structure")
        print("    - Voice alerts for mask violations")  
        
        
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
    
    def _cleanup_old_tracks(self, frame_count):
        """Remove tracks that haven't been updated recently"""
        expired_tracks = []
        for track_id, track in self.active_tracks.items():
            if frame_count - track['last_updated_frame'] > 100:  # 100 frames threshold
                expired_tracks.append(track_id)
        
        for track_id in expired_tracks:
            del self.active_tracks[track_id]    
    
    def _create_track(self, result, frame_count):
        """Create a new track from recognition result"""
        return {
            'identity': result['identity'],
            'recognition_confidence': result['recognition_confidence'],
            'detection_confidence': result['detection_confidence'],
            'current_bbox': result['bbox'],
            'confidence_count': 1 if result['identity'] else 0,
            'first_detected_frame': frame_count,
            'last_updated_frame': frame_count,
            'cooldown_counter': 0,
            'track_state': 'TRACKING'  # TRACKING, COOLDOWN, EXPIRED
        }
    
    def _update_track(self, track, new_result, frame_count):
        """Update existing track with new recognition"""
        updated_track = track.copy()
        updated_track['current_bbox'] = new_result['bbox']
        updated_track['last_updated_frame'] = frame_count
        
        # Only update identity if we have a good recognition
        if new_result['identity'] and new_result['recognition_confidence'] > 0.6:
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
    
class FairnessController:
    def __init__(self):
        self.recognition_counts = defaultdict(int)
        self.recent_recognitions = deque(maxlen=100)
        self.max_recognitions_per_person = 10  # Prevent domination
        
    def ensure_fair_attention(self, current_results: List[Dict]) -> List[Dict]:
        """Ensure no single person dominates system attention"""
        fair_results = []
        
        for result in current_results:
            identity = result.get('identity')
            if identity and identity != "Unknown":
                # Check if this person has been recognized too frequently
                recent_count = self._get_recent_recognition_count(identity)
                
                if recent_count < self.max_recognitions_per_person:
                    fair_results.append(result)
                    self.recognition_counts[identity] += 1
                    self.recent_recognitions.append(identity)
                else:
                    # Downgrade priority for over-represented persons
                    result['priority_score'] *= 0.5  # Reduce priority
                    fair_results.append(result)
            else:
                fair_results.append(result)
        
        return fair_results
    
    def _get_recent_recognition_count(self, identity: str) -> int:
        """Count how many times this identity was recently recognized"""
        return sum(1 for rec in self.recent_recognitions if rec == identity)                        
                                 
# Update your CONFIG dictionary:
CONFIG = {
    'detection_model_path': r'D:\SCMA\3-APD\fromAraya\Computer-Vision-CV\3.1_FaceRecog\yolov11n-face.pt',
    'mask_model_path': r'D:\SCMA\3-APD\fromAraya\Computer-Vision-CV\3.1_FaceRecog\run_py\mask_detector112.onnx',  
    'embeddings_db_path': r'D:\SCMA\3-APD\fromAraya\Computer-Vision-CV\3.1_FaceRecog\person_folder_3.json',
    'detection_confidence': 0.6,
    'detection_iou': 0.6,
    'mask_detection_threshold': 0.95,  
    'roi_padding': 20,  
    'embedding_model': 'Facenet',
    'recognition_threshold': 0.6,  
    'max_faces_per_frame': 10,
    'min_face_size': 40,  
    'enable_face_tracking': True,
    'tracking_max_age': 100,
    
    # 🆕 Duration threshold settings
    'min_violation_frames': 5,      # . seconds at 10 FPS
    'min_violation_seconds': 0.5,    # Minimum . seconds continuous violation
    'max_gap_frames': 5,             # Allow 5-frame gaps in tracking
    'verbose_duration_tracking': True,  # Print debug info
    
    # 🆕 Enhanced alert settings with retry configuration
    'alert_server_url': 'https://vps.scasda.my.id/actions/a_notifikasi_suara_speaker.php',
    'alert_cooldown_seconds': 5,
    'enable_voice_alerts': True,
    
    # 🆕 Retry configuration
    'alert_max_retries': 3,           # Number of retry attempts
    'alert_retry_delay': 2,           # Seconds between retries
    'alert_retry_timeout': 5,         # Timeout per attempt in seconds
}

# ENHANCED CONFIG WITH SIMILARITY SETTINGS
ROBUST_CONFIG = {
    **CONFIG,
    
    # Robustness enhancements
    'enable_multi_scale': True,
    'enable_temporal_fusion': True, 
    'enable_quality_aware': True,
    'enable_quality_adaptive_similarity': True,
    'min_face_quality': 0.3,
    'temporal_buffer_size': 10,
    
    # Multi-scale processing
    'scale_factors': [0.5, 0.75, 1.0, 1.25, 1.5],
    'rotation_angles': [-10, -5, 0, 5, 10],
    
    # Quality assessment weights
    'quality_weights': {
        'sharpness': 0.3,
        'brightness': 0.2, 
        'contrast': 0.15,
        'size': 0.2,
        'position': 0.1,
        'blur': 0.05
    },
    
    # Enhanced similarity configuration
    'similarity_method': 'quality_adaptive',
    'similarity_weights': {
        'cosine': 0.25,
        'angular': 0.40,
        'pearson': 0.15,
        'dot_product': 0.15,
        'euclidean': 0.10,
        'manhattan': 0.15,
        'jaccard': 0.20
    },
    
    # Quality-adaptive settings
    'quality_adaptive_verbose': True,
    'quality_profiles': {
        'high_quality': {'min_quality': 0.7},
        'medium_quality': {'min_quality': 0.4},
        'low_quality': {'min_quality': 0.2},
        'very_low_quality': {'min_quality': 0.0}
    }
}

# Use this config for context-aware processing
CONTEXT_AWARE_CONFIG = {
    **ROBUST_CONFIG,
    
    # Context-aware scaling parameters
    'min_processing_scale': 0.3,
    'max_processing_scale': 2.5,
    'scale_adjustment_step': 0.1,
    'context_weight': 0.4,
    'performance_weight': 0.6,
    
    # Enable context features
    'enable_context_aware_scaling': True,
}

# Enhanced recognition method
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

def main_priority_optimized():
    """Main function with priority-aware optimization"""
    # Create priority-aware system
    config_manager = ConfigManager(CONFIG)
    robust_config = config_manager.get_component_config('context_aware_processing')
    face_system = RobustFaceRecognitionSystem(robust_config)
    processor = RealTimeProcessor(face_system=face_system) # , processing_interval=10
    
    # Add fairness controller
    fairness_controller = FairnessController()
    
    print("🚀 Starting with DURATION-AWARE ALERT SYSTEM")
    print(f"   - Minimum: {CONTEXT_AWARE_CONFIG['min_violation_frames']} frames "
          f"({CONTEXT_AWARE_CONFIG['min_violation_seconds']}s)")
    print("   - Per-identity tracking with gap tolerance")
    
    def priority_aware_callback(results: List[Dict]):
        """Monitor system performance and fairness"""
        # Apply fairness controls
        fair_results = fairness_controller.ensure_fair_attention(results)
        
        # Log priority distribution
        priority_levels = {'high': 0, 'medium': 0, 'low': 0}
        for result in fair_results:
            level = result.get('processing_level', 'basic')
            priority_levels[level] += 1
        
        print(f"🎯 Priority Distribution: High={priority_levels['high']}, "
            f"Medium={priority_levels['medium']}, Low={priority_levels['low']}")
        
        # Monitor recognition fairness
        unique_identities = len(set(r['identity'] for r in fair_results if r['identity']))
        print(f"👥 Unique identities detected: {unique_identities}")
    
    processor.results_callback = priority_aware_callback
    
    # Choose input source
    source = select_source()
    
    # Configure display
    processor.set_display_size(1280, 720, "fixed_size")
    
    try:
        processor.run(source)
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        processor.stop()

def select_source():
    """Interactive source selection"""
    sources = {
        '1': '0',  # Default camera
        '2': 'rtsp://admin:Admin888@192.168.0.2:554/Streaming/Channels/101',
        '3': 'http://192.168.1.101:8080/video',
        '4': 'video.mp4'
    }
    
    print("Available sources:")
    for key, source in sources.items():
        print(f"  {key}: {source}")
    
    choice = input("Select source (1-4) or enter custom RTSP URL: ").strip()
    return sources.get(choice, choice)

if __name__ == "__main__":
    main_priority_optimized()