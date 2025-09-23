import cv2
import time
from datetime import datetime

# DVR RTSP connection parameters
DVR_IP = "192.168.0.8"
USERNAME = "admin"
PASSWORD = "Admin888"
CHANNEL = "Channels"  # Camera channel number
STREAM_TYPE = 101  # 0 = main stream, 1 = sub stream

# Construct RTSP URL "rtsp://admin:Admin888@192.168.0.8:554/Streaming/Channels/101" f"rtsp://{USERNAME}:{PASSWORD}@{DVR_IP}:554/Streaming/Channels/{CHANNEL}0{STREAM_TYPE+1}"
rtsp_url = ""

# Alternative formats to try if above doesn't work:
# rtsp_url = f"rtsp://{USERNAME}:{PASSWORD}@{DVR_IP}:554/cam/realmonitor?channel={CHANNEL}&subtype={STREAM_TYPE}"
# rtsp_url = f"rtsp://{USERNAME}:{PASSWORD}@{DVR_IP}:554/ch0{CHANNEL}.264"

# Initialize video capture with buffer optimization
cap = cv2.VideoCapture(rtsp_url)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Reduce buffer to minimize delay
cap.set(cv2.CAP_PROP_FPS, 15)        # Set expected FPS

# Check if connection was successful
if not cap.isOpened():
    print("Error: Could not connect to DVR stream")
    exit()

print("Successfully connected to DVR stream")

# Generate unique filename with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f'cctv_captured/output_{timestamp}.avi'


# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'DIVX') # In Windows: DIVX (More to be tested and added). DIVX is for FourCC, a 
                                        # 4-byte code used to specify the video codec.
out = cv2.VideoWriter(output_filename, fourcc, 20.0, (1024, 576))
 
try:
    while True:
        # Read frame
        ret, frame = cap.read()
        
        if not ret:
            print("Frame read error, attempting to reconnect...")
            cap.release()
            time.sleep(2)  # Wait before reconnecting
            cap = cv2.VideoCapture(rtsp_url)
            continue
            
        # Process frame (resize for better performance if needed)
        frame = cv2.resize(frame, (1024, 576))
        
        # Display frame and Write the frame into video
        cv2.imshow('DVR Stream', frame)
        out.write(frame)        
         
        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        

finally:
    cap.release()
    cv2.destroyAllWindows()