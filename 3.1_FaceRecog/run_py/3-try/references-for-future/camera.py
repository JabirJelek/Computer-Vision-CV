import cv2
from config import CAMERA_ID, FRAME_WIDTH, FRAME_HEIGHT

class Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture(CAMERA_ID)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        
    def get_frame(self):
        """Get frame from camera"""
        ret, frame = self.cap.read()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None
    
    def release(self):
        """Release camera resources"""
        self.cap.release()