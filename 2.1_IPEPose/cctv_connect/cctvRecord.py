import cv2
import time
from datetime import datetime

rtsp_url = input("Enter the RTSP URL: ")

if not rtsp_url.startswith('rtsp://'):
    print("That doesn't look like a valid RTSP URL. Please try again.")
    exit()
    
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