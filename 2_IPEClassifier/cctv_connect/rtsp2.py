# capture using rtsp protocol
import cv2

rtsp_url = "rtsp://{USERNAME}:{PASSWORD}@{DVR_IP}:554/ch0{CHANNEL}.264"

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