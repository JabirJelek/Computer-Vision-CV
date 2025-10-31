# interfaces/processor.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import numpy as np

class IProcessor(ABC):
    @abstractmethod
    def initialize(self, config: Dict) -> bool:
        pass
    
    @abstractmethod
    def process(self, frame: np.ndarray, context: Dict) -> Dict:
        pass
    
    @abstractmethod
    def cleanup(self):
        pass

class IAnalyzer(ABC):
    @abstractmethod
    def analyze(self, data: Dict, context: Dict) -> Dict:
        pass