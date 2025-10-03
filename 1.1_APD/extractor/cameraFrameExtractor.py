import cv2
import os
import time
from datetime import datetime
from pathlib import Path

class VideoFrameExtractor:
    def __init__(self, output_dir="captured_frames", capture_interval=1.0,
                 resolution=(1280, 720), prefix="frame", format="jpg"):
        """
        Initialize the video frame extractor

        Args:
            output_dir (str): Directory to save captured frames
            capture_interval (float): Time interval between frame captures in seconds
            resolution (tuple): Camera resolution (width, height)
            prefix (str): Prefix for saved frame filenames
            format (str): Image format (jpg, png)
        """
        self.output_dir = output_dir
        self.capture_interval = capture_interval
        self.resolution = resolution
        self.prefix = prefix
        self.format = format
        self.cap = None

        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    def start_capture(self, duration=None, camera_index=0):
        """
        Start capturing frames from the camera

        Args:
            duration (float): Total capture duration in seconds (None for indefinite)
            camera_index (int): Camera device index
        """
        # Initialize video capture
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            print("Error: Could not open camera")
            return

        # Set camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

        print(f"Starting video capture. Press 'q' to stop.")
        print(f"Frames will be saved to: {self.output_dir}")
        print(f"Capture interval: {self.capture_interval} seconds")

        frame_count = 0
        last_capture_time = time.time()
        start_time = time.time()

        try:
            while True:
                # Check if duration limit reached
                if duration and (time.time() - start_time) >= duration:
                    break

                # Read frame from camera
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Failed to capture frame")
                    break

                current_time = time.time()

                # Check if it's time to capture a frame
                if current_time - last_capture_time >= self.capture_interval:
                    # Generate filename with timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = f"{self.prefix}_{timestamp}.{self.format}"
                    filepath = os.path.join(self.output_dir, filename)

                    # Save frame
                    if self.format.lower() == "jpg":
                        cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    else:
                        cv2.imwrite(filepath, frame)

                    print(f"Captured frame: {filename}")
                    frame_count += 1
                    last_capture_time = current_time

                # Display preview (optional)
                cv2.imshow('Frame Capture', frame)

                # Check for quit command
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("\nCapture interrupted by user")
        finally:
            # Clean up
            self.cap.release()
            cv2.destroyAllWindows()
            print(f"Capture completed. Saved {frame_count} frames.")

    def set_interval(self, interval):
        """Change the capture interval during operation"""
        self.capture_interval = interval
        print(f"Capture interval changed to {interval} seconds")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Capture frames from webcam at regular intervals")
    parser.add_argument("-o", "--output", default="captured_frames", help="Output directory")
    parser.add_argument("-i", "--interval", type=float, default=0.5, help="Capture interval in seconds") # Change this default if want to directly change the number interval between capture
    parser.add_argument("-d", "--duration", type=float, help="Total capture duration in seconds")
    parser.add_argument("-r", "--resolution", nargs=2, type=int, default=[1280, 720],
                        help="Camera resolution (width height)")
    parser.add_argument("-p", "--prefix", default="frame", help="Filename prefix")
    parser.add_argument("-f", "--format", default="jpg", choices=["jpg", "png"], help="Image format")
    parser.add_argument("-c", "--camera", type=int, default=0, help="Camera index")

    args = parser.parse_args()

    # Create frame extractor
    extractor = VideoFrameExtractor(
        output_dir=args.output,
        capture_interval=args.interval,
        resolution=tuple(args.resolution),
        prefix=args.prefix,
        format=args.format
    )

    # Start capture
    extractor.start_capture(duration=args.duration, camera_index=args.camera)

if __name__ == "__main__":
    main()