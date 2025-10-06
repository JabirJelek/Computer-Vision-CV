# capture using rtsp protocol
import cv2

rtsp_url = input("Enter the RTSP URL: ")

if not rtsp_url.startswith('rtsp://'):
    print("That doesn't look like a valid RTSP URL. Please try again.")
    exit()
    
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
        print("Error: Could not open RTSP stream.")
else:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from stream.")
                break

            cv2.imshow("RTSP Stream", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()