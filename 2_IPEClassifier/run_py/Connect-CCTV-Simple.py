# capture using rtsp protocol
import cv2

rtsp_url = "rtsp://admin:CemaraMas2025!@192.168.2.190:554/Streaming/Channels/101"

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