import cv2
import time

# Construct RTSP URL f"rtsp://{USERNAME}:{PASSWORD}@{DVR_IP}:554/Streaming/Channels/{CHANNEL}0{STREAM_TYPE+1}"
rtsp_url = "rtsp://{USERNAME}:{PASSWORD}@{DVR_IP}:554/ch0{CHANNEL}.264"

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
        
        # Display frame
        cv2.imshow('DVR Stream', frame)
        
        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
finally:
    cap.release()
    cv2.destroyAllWindows()