# core/config_manager.py
from typing import Dict, Any, List
import json
from pathlib import Path

class PipelineConfig:
    """Enhanced configuration management for pipeline"""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self._config = self._validate_and_default_config(config_dict)
        self._profiles = self._build_profiles()
    
    def _validate_and_default_config(self, config: Dict) -> Dict:
        """Validate configuration and set defaults"""
        required_keys = ['detection_model_path', 'embeddings_db_path']
        
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")
        
        # Set defaults for optional parameters
        defaults = {
            'detection_confidence': 0.6,
            'detection_iou': 0.6,
            'recognition_threshold': 0.7,
            'enable_face_tracking': True,
            'processing_interval': 5
        }
        
        # Merge with defaults
        merged_config = {**defaults, **config}
        
        # Validate paths exist
        self._validate_paths(merged_config)
        
        return merged_config
    
    def _validate_paths(self, config: Dict):
        """Validate that file paths exist"""
        path_keys = ['detection_model_path', 'embeddings_db_path', 'mask_model_path']
        
        for key in path_keys:
            if key in config and config[key]:
                path = Path(config[key])
                if not path.exists():
                    print(f"⚠️  Config: {key} path does not exist: {config[key]}")
    
    def _build_profiles(self) -> Dict[str, Dict]:
        """Build configuration profiles"""
        return {
            'performance': self._build_performance_profile(),
            'accuracy': self._build_accuracy_profile(), 
            'balanced': self._build_balanced_profile()
        }
    
    def _build_performance_profile(self) -> Dict:
        """Performance-optimized profile"""
        return {
            **self._config,
            'detection_confidence': 0.5,
            'processing_interval': 10,
            'enable_face_tracking': False
        }
    
    def _build_accuracy_profile(self) -> Dict:
        """Accuracy-optimized profile"""
        return {
            **self._config,
            'detection_confidence': 0.7,
            'recognition_threshold': 0.8,
            'processing_interval': 2
        }
    
    def _build_balanced_profile(self) -> Dict:
        """Balanced profile"""
        return self._config  # Use base config
    
    def get_profile(self, profile_name: str) -> 'PipelineConfig':
        """Get specific configuration profile"""
        if profile_name in self._profiles:
            return PipelineConfig(self._profiles[profile_name])
        else:
            print(f"⚠️  Profile '{profile_name}' not found, using balanced")
            return self
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self._config.get(key, default)
    
    def __getitem__(self, key: str) -> Any:
        """Dictionary-like access"""
        return self._config[key]
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists"""
        return key in self._config
    
    def items(self):
        """Iterate over config items"""
        return self._config.items()