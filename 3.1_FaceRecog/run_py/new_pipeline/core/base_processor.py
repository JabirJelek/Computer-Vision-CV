# core/base_processor.py
from abc import abstractmethod
from typing import Dict, List, Any
import time
import numpy as np
from interfaces.processor import IProcessor

class BaseProcessor(IProcessor):
    """Complete base class for all pipeline processors"""
    
    def __init__(self, name: str, config: Dict):
        self.name = name
        self.config = config
        self.initialized = False
        self.metrics = {
            'total_process_calls': 0,
            'total_processing_time': 0.0,
            'avg_processing_time': 0.0,
            'last_processing_time': 0.0,
            'processing_times': []
        }
        
    def initialize(self) -> bool:
        """Common initialization with error handling"""
        try:
            print(f"🔄 Initializing {self.name}...")
            result = self._initialize_impl()
            self.initialized = result
            
            if result:
                print(f"✅ {self.name} initialized successfully")
            else:
                print(f"❌ {self.name} initialization failed")
                
            return result
            
        except Exception as e:
            print(f"❌ Initialization failed for {self.name}: {e}")
            self.initialized = False
            return False
    
    @abstractmethod
    def _initialize_impl(self) -> bool:
        """Processor-specific initialization - MUST be implemented by subclasses"""
        pass
    
    def process(self, data: Dict, context: Dict) -> Dict:
        """Template method with timing and error handling"""
        if not self.initialized:
            raise RuntimeError(f"Processor {self.name} not initialized")
        
        start_time = time.time()
        
        try:
            # Process the data
            result = self._process_impl(data, context)
            
            # Update metrics
            processing_time = time.time() - start_time
            self._update_metrics(processing_time)
            
            return result
            
        except Exception as e:
            print(f"❌ Processing failed in {self.name}: {e}")
            # Return original data on failure to allow pipeline continuation
            return data
    
    @abstractmethod
    def _process_impl(self, data: Dict, context: Dict) -> Dict:
        """Processor-specific implementation - MUST be implemented by subclasses"""
        pass
    
    def _update_metrics(self, processing_time: float):
        """Update common performance metrics"""
        self.metrics['total_process_calls'] += 1
        self.metrics['total_processing_time'] += processing_time
        self.metrics['last_processing_time'] = processing_time
        self.metrics['processing_times'].append(processing_time)
        
        # Keep only recent history
        if len(self.metrics['processing_times']) > 100:
            self.metrics['processing_times'].pop(0)
            
        # Update average
        self.metrics['avg_processing_time'] = (
            self.metrics['total_processing_time'] / self.metrics['total_process_calls']
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current processor metrics"""
        return self.metrics.copy()
    
    def cleanup(self):
        """Common cleanup - can be overridden by subclasses"""
        print(f"🧹 Cleaning up {self.name}")
        self.initialized = False
    
    def __del__(self):
        """Destructor to ensure cleanup - SAFER VERSION"""
        try:
            self.cleanup()
        except Exception as e:
            # Silently ignore errors during destruction
            pass