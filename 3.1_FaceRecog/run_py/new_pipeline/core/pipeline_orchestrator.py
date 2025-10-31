# core/pipeline_orchestrator.py
from typing import Dict, List, Optional, Any
import time
import numpy as np
from collections import deque
import threading

from .base_processor import BaseProcessor
from interfaces.processor import IProcessor

class PipelineOrchestrator:
    """
    Enhanced pipeline coordinator with dynamic processor management,
    error handling, and comprehensive data flow coordination
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.processors: List[IProcessor] = []
        self.processor_map: Dict[str, IProcessor] = {}
        self.is_initialized = False
        self.is_running = False
        
        # Enhanced context management
        self.context = {
            'frame_count': 0,
            'start_time': time.time(),
            'performance_metrics': {},
            'pipeline_state': 'stopped',  # stopped, running, paused, error
            'last_error': None,
            'temporal_data': {},
            'component_states': {}
        }
        
        # Data validation and error handling
        self.error_history = deque(maxlen=50)
        self.processing_times = deque(maxlen=100)
        
        # Threading and state management
        self.pipeline_lock = threading.RLock()
        self.state_callbacks = []
        
        print("🎯 Enhanced PipelineOrchestrator initialized")
    
    def register_processor(self, processor: IProcessor) -> bool:
        """Dynamically register a processor with validation"""
        with self.pipeline_lock:
            try:
                if not isinstance(processor, BaseProcessor):
                    print(f"❌ Processor must inherit from BaseProcessor: {type(processor)}")
                    return False
                
                if processor.name in self.processor_map:
                    print(f"⚠️ Processor '{processor.name}' already registered, replacing")
                
                self.processors.append(processor)
                self.processor_map[processor.name] = processor
                
                print(f"✅ Registered processor: {processor.name}")
                return True
                
            except Exception as e:
                print(f"❌ Failed to register processor: {e}")
                return False
    
    def initialize_pipeline(self) -> bool:
        """Initialize all registered processors with dependency validation"""
        with self.pipeline_lock:
            try:
                if not self.processors:
                    print("❌ No processors registered")
                    return False
                
                print(f"🚀 Initializing pipeline with {len(self.processors)} processors...")
                
                # Initialize processors in registration order
                initialized_processors = []
                failed_processors = []
                
                for processor in self.processors:
                    try:
                        print(f"🔄 Initializing {processor.name}...")
                        if processor.initialize():
                            initialized_processors.append(processor)
                            self.context['component_states'][processor.name] = 'initialized'
                            print(f"✅ {processor.name} initialized successfully")
                        else:
                            failed_processors.append(processor.name)
                            self.context['component_states'][processor.name] = 'failed'
                            print(f"❌ {processor.name} initialization failed")
                    except Exception as e:
                        failed_processors.append(processor.name)
                        self.context['component_states'][processor.name] = 'error'
                        print(f"❌ {processor.name} initialization error: {e}")
                
                if failed_processors:
                    print(f"⚠️ {len(failed_processors)} processors failed: {failed_processors}")
                    # Continue with successful processors for graceful degradation
                
                self.is_initialized = len(initialized_processors) > 0
                self.context['pipeline_state'] = 'initialized' if self.is_initialized else 'error'
                
                if self.is_initialized:
                    print(f"✅ Pipeline initialized with {len(initialized_processors)}/{len(self.processors)} processors")
                    self._log_pipeline_status()
                else:
                    print("❌ Pipeline initialization failed - no processors available")
                
                return self.is_initialized
                
            except Exception as e:
                print(f"❌ Pipeline initialization error: {e}")
                self.context['pipeline_state'] = 'error'
                self.context['last_error'] = str(e)
                return False
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process single frame through entire pipeline with enhanced error handling
        and data validation
        """
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized")
        
        start_time = time.time()
        
        # Create pipeline data structure with validation
        pipeline_data = self._create_initial_data(frame)
        processing_errors = []
        
        try:
            # Execute processors in sequence with error isolation
            for i, processor in enumerate(self.processors):
                processor_start = time.time()
                
                try:
                    # Skip processors that failed initialization
                    if self.context['component_states'].get(processor.name) in ['failed', 'error']:
                        continue
                    
                    # Process with current data and context
                    pipeline_data = processor.process(pipeline_data, self.context)
                    
                    # Validate processor output
                    if not self._validate_processor_output(pipeline_data, processor.name):
                        raise ValueError(f"Invalid output from {processor.name}")
                    
                    # Update processor metrics
                    processor_time = time.time() - processor_start
                    self._update_processor_metrics(processor.name, processor_time, True)
                    
                except Exception as e:
                    # Isolate processor errors but continue pipeline
                    error_msg = f"Processor {processor.name} failed: {e}"
                    print(f"⚠️ {error_msg}")
                    processing_errors.append(error_msg)
                    self._update_processor_metrics(processor.name, time.time() - processor_start, False)
                    
                    # Mark processor as errored but continue
                    self.context['component_states'][processor.name] = 'error'
                    continue
            
            # Update overall pipeline metrics
            total_time = time.time() - start_time
            self.processing_times.append(total_time)
            self.context['frame_count'] += 1
            
            # Add processing metadata to results
            pipeline_data['pipeline_metadata'] = {
                'processing_time': total_time,
                'frame_id': self.context['frame_count'],
                'errors': processing_errors,
                'successful_processors': [p.name for p in self.processors 
                                        if self.context['component_states'].get(p.name) == 'initialized'],
                'timestamp': time.time()
            }
            
            return pipeline_data
            
        except Exception as e:
            error_msg = f"Pipeline processing failed: {e}"
            print(f"❌ {error_msg}")
            self.error_history.append(error_msg)
            self.context['last_error'] = error_msg
            
            # Return partial results with error information
            pipeline_data['pipeline_metadata'] = {
                'processing_time': time.time() - start_time,
                'frame_id': self.context['frame_count'],
                'errors': [error_msg] + processing_errors,
                'successful_processors': [],
                'timestamp': time.time(),
                'error': True
            }
            return pipeline_data
    
    def _create_initial_data(self, frame: np.ndarray) -> Dict:
        """Create initial pipeline data structure with validation"""
        if frame is None or frame.size == 0:
            raise ValueError("Invalid input frame")
        
        return {
            'input_frame': frame,
            'timestamp': time.time(),
            'frame_id': self.context['frame_count'],
            'detections': [],  # Standardized keys for data flow
            'recognitions': [],
            'analysis_results': [],
            'output_data': {},
            'metadata': {
                'frame_shape': frame.shape,
                'frame_dtype': str(frame.dtype)
            }
        }
    
    def _validate_processor_output(self, data: Dict, processor_name: str) -> bool:
        """Validate processor output maintains required structure"""
        if not isinstance(data, dict):
            print(f"❌ {processor_name}: Output must be a dictionary")
            return False
        
        # Check for required base keys
        required_keys = ['input_frame', 'timestamp', 'frame_id']
        for key in required_keys:
            if key not in data:
                print(f"❌ {processor_name}: Missing required key '{key}'")
                return False
        
        # Validate frame data
        if data['input_frame'] is None or data['input_frame'].size == 0:
            print(f"❌ {processor_name}: Invalid frame data")
            return False
        
        return True
    
    def _update_processor_metrics(self, processor_name: str, processing_time: float, success: bool):
        """Update metrics for individual processors"""
        if 'performance_metrics' not in self.context:
            self.context['performance_metrics'] = {}
        
        if processor_name not in self.context['performance_metrics']:
            self.context['performance_metrics'][processor_name] = {
                'total_calls': 0,
                'successful_calls': 0,
                'failed_calls': 0,
                'total_processing_time': 0.0,
                'avg_processing_time': 0.0,
                'last_processing_time': 0.0
            }
        
        metrics = self.context['performance_metrics'][processor_name]
        metrics['total_calls'] += 1
        metrics['last_processing_time'] = processing_time
        metrics['total_processing_time'] += processing_time
        metrics['avg_processing_time'] = metrics['total_processing_time'] / metrics['total_calls']
        
        if success:
            metrics['successful_calls'] += 1
        else:
            metrics['failed_calls'] += 1
    
    def start_pipeline(self):
        """Start pipeline execution"""
        with self.pipeline_lock:
            if not self.is_initialized:
                print("⚠️ Pipeline not initialized, attempting initialization...")
                if not self.initialize_pipeline():
                    raise RuntimeError("Cannot start pipeline - initialization failed")
            
            self.is_running = True
            self.context['pipeline_state'] = 'running'
            print("🎬 Pipeline started")
    
    def pause_pipeline(self):
        """Pause pipeline execution"""
        with self.pipeline_lock:
            self.is_running = False
            self.context['pipeline_state'] = 'paused'
            print("⏸️ Pipeline paused")
    
    def stop_pipeline(self):
        """Stop pipeline execution and cleanup"""
        with self.pipeline_lock:
            self.is_running = False
            self.context['pipeline_state'] = 'stopped'
            self.cleanup()
            print("🛑 Pipeline stopped")
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get comprehensive pipeline status"""
        with self.pipeline_lock:
            status = {
                'initialized': self.is_initialized,
                'running': self.is_running,
                'state': self.context['pipeline_state'],
                'processor_count': len(self.processors),
                'active_processors': len([p for p in self.processors 
                                        if self.context['component_states'].get(p.name) == 'initialized']),
                'frame_count': self.context['frame_count'],
                'last_error': self.context['last_error'],
                'performance_metrics': self.context['performance_metrics'],
                'component_states': self.context['component_states'],
                'avg_processing_time': np.mean(self.processing_times) if self.processing_times else 0,
                'recent_errors': list(self.error_history)[-5:]  # Last 5 errors
            }
            return status
    
    def _log_pipeline_status(self):
        """Log current pipeline configuration and status"""
        print("\n" + "="*60)
        print("📊 PIPELINE CONFIGURATION STATUS")
        print("="*60)
        print(f"Total Processors: {len(self.processors)}")
        print(f"Initialized: {self.is_initialized}")
        
        for processor in self.processors:
            state = self.context['component_states'].get(processor.name, 'unknown')
            status_icon = "✅" if state == 'initialized' else "❌"
            print(f"  {status_icon} {processor.name}: {state}")
        
        print("="*60)
    
    def add_state_callback(self, callback):
        """Add callback for pipeline state changes"""
        self.state_callbacks.append(callback)
    
    def _notify_state_change(self, old_state: str, new_state: str):
        """Notify callbacks of state changes"""
        for callback in self.state_callbacks:
            try:
                callback(old_state, new_state, self.context)
            except Exception as e:
                print(f"⚠️ State callback error: {e}")
    
    def cleanup(self):
        """Cleanup all processors and resources"""
        with self.pipeline_lock:
            print("🧹 Cleaning up pipeline...")
            
            for processor in self.processors:
                try:
                    processor.cleanup()
                    print(f"✅ Cleaned up {processor.name}")
                except Exception as e:
                    print(f"⚠️ Error cleaning up {processor.name}: {e}")
            
            self.processors.clear()
            self.processor_map.clear()
            self.is_initialized = False
            self.is_running = False
            self.context['pipeline_state'] = 'stopped'
            
            print("✅ Pipeline cleanup completed")

    def get_processor(self, name: str) -> Optional[IProcessor]:
        """Get processor by name"""
        return self.processor_map.get(name)
    
    def validate_data_flow(self) -> Dict[str, Any]:
        """Validate data flow between processors"""
        validation_results = {}
        
        for i, processor in enumerate(self.processors):
            processor_info = {
                'position': i,
                'name': processor.name,
                'state': self.context['component_states'].get(processor.name, 'unknown'),
                'dependencies': getattr(processor, 'dependencies', []),
                'provides': getattr(processor, 'provides', [])
            }
            validation_results[processor.name] = processor_info
        
        return validation_results
    
    def _initialize_recognition(self):
        """Initialize face recognition components"""
        try:
            # Choose between basic and robust recognizer based on config
            use_robust = self.config.get('enable_robust_recognition', True)
            
            if use_robust:
                from components.recognition.robust_face_recognizer import RobustFaceRecognizer
                robust_face_recognizer = RobustFaceRecognizer("FaceRecognizer", self.config)
            else:
                from components.recognition.face_recognizer import FaceRecognizer
                face_recognizer = FaceRecognizer("FaceRecognizer", self.config)
            
            if face_recognizer.initialize():
                self.register_processor(face_recognizer)
                print("✅ FaceRecognizer added to pipeline")
            else:
                raise RuntimeError("Failed to initialize FaceRecognizer")
                
        except Exception as e:
            print(f"❌ Failed to initialize recognition: {e}")
            # Pipeline can continue without recognition for graceful degradation    
            
    def _initialize_mask_detection(self):
        """Initialize mask detection component"""
        try:
            from components.detection.mask_detector import MaskDetector
            mask_detector = MaskDetector("MaskDetector", self.config)
            
            if mask_detector.initialize():
                self.register_processor(mask_detector)
                print("✅ MaskDetector added to pipeline")
            else:
                print("⚠️ MaskDetector initialization failed - continuing without mask detection")
                # Don't raise exception, continue without mask detection
                
        except Exception as e:
            print(f"❌ Failed to initialize mask detection: {e}")
            # Pipeline can continue without mask detection for graceful degradation
    
    def initialize_pipeline(self) -> bool:
        """Initialize all registered processors with dependency validation"""
        with self.pipeline_lock:
            try:
                if not self.processors:
                    print("❌ No processors registered")
                    return False
                
                print(f"🚀 Initializing pipeline with {len(self.processors)} processors...")
                
                # ✅ CORRECT: Initialize processors in registration order
                initialized_processors = []
                failed_processors = []
                
                for processor in self.processors:
                    try:
                        print(f"🔄 Initializing {processor.name}...")
                        if processor.initialize():
                            initialized_processors.append(processor)
                            self.context['component_states'][processor.name] = 'initialized'
                            print(f"✅ {processor.name} initialized successfully")
                        else:
                            failed_processors.append(processor.name)
                            self.context['component_states'][processor.name] = 'failed'
                            print(f"❌ {processor.name} initialization failed")
                    except Exception as e:
                        failed_processors.append(processor.name)
                        self.context['component_states'][processor.name] = 'error'
                        print(f"❌ {processor.name} initialization error: {e}")
                
                if failed_processors:
                    print(f"⚠️ {len(failed_processors)} processors failed: {failed_processors}")
                    # Continue with successful processors for graceful degradation
                
                self.is_initialized = len(initialized_processors) > 0
                self.context['pipeline_state'] = 'initialized' if self.is_initialized else 'error'
                
                if self.is_initialized:
                    print(f"✅ Pipeline initialized with {len(initialized_processors)}/{len(self.processors)} processors")
                    self._log_pipeline_status()
                else:
                    print("❌ Pipeline initialization failed - no processors available")
                
                return self.is_initialized
                
            except Exception as e:
                print(f"❌ Pipeline initialization error: {e}")
                self.context['pipeline_state'] = 'error'
                self.context['last_error'] = str(e)
                return False
