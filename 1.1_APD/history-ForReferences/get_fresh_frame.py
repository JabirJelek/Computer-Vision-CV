# freshest_frame.py
"""
A thread-based video capture class for always retrieving the latest frame.
"""

from __future__ import print_function
import os
import sys
import time
import threading
import numpy as np
import cv2 as cv


class FreshestFrame(threading.Thread):
    """
    A thread that continuously reads frames from a cv2.VideoCapture object
    and provides the latest available frame on demand, acting as a drop-in
    replacement for part of cv2.VideoCapture's functionality.
    """

    def __init__(self, capture, name='FreshestFrame'):
        self.capture = capture
        if not self.capture.isOpened():
            raise RuntimeError("VideoCapture object is not opened.")

        self._cond = threading.Condition()
        self._running = False
        self._frame = None
        self._latestnum = 0
        self.callback = None

        super().__init__(name=name)
        self.start()

    def start(self):
        self._running = True
        super().start()

    def release(self, timeout=None):
        self._running = False
        self.join(timeout=timeout)
        self.capture.release()

    def __del__(self):
        """Ensures resources are released on object deletion."""
        self.release()

    def run(self):
        counter = 0
        while self._running:
            (rv, img) = self.capture.read()
            if not rv:
                break
            counter += 1

            with self._cond:
                self._frame = img
                self._latestnum = counter
                self._cond.notify_all()

            if self.callback:
                self.callback(img)

    def read(self, wait=True, seqnumber=None, timeout=None):
        """
        Reads the latest frame.

        Args:
            wait (bool): If True, blocks for a fresh frame.
            seqnumber (int): Block until this frame sequence number is available.
            timeout (float): Maximum time in seconds to wait.

        Returns:
            tuple: (frame_sequence_number, frame_image)
        """
        with self._cond:
            if wait:
                if seqnumber is None:
                    seqnumber = self._latestnum + 1
                if seqnumber < 1:
                    seqnumber = 1

                rv = self._cond.wait_for(
                    lambda: self._latestnum >= seqnumber,
                    timeout=timeout
                )
                if not rv:
                    return (self._latestnum, self._frame)

            return (self._latestnum, self._frame)

    # Property to safely access the latest frame number
    @property
    def latestnum(self):
        with self._cond:
            return self._latestnum

    # Property to safely access the latest frame
    @property
    def frame(self):
        with self._cond:
            return self._frame


def main():
    """Main function to demonstrate the usage of the FreshestFrame class."""
    cv.namedWindow("frame")
    cv.namedWindow("realtime")

    cap = cv.VideoCapture('rtsp://admin:CemaraMas2025!@192.168.2.190:554/Streaming/Channels/101')
    cap.set(cv.CAP_PROP_FPS, 30)

    fresh = FreshestFrame(cap)

    def callback(img):
        cv.imshow("realtime", img)

    fresh.callback = callback

    cnt = 0
    try:
        while True:
            t0 = time.perf_counter()
            cnt, img = fresh.read(seqnumber=cnt + 1)
            dt = time.perf_counter() - t0

            if dt > 0.010:
                print("NOTICE: read() took {dt:.3f} secs".format(dt=dt))

            print("processing {cnt}...".format(cnt=cnt), end=" ", flush=True)
            cv.imshow("frame", img)

            key = cv.waitKey(200)
            if key == 27:  # Escape key
                break
            print("done!")
    finally:
        fresh.release()
        cv.destroyAllWindows()


if __name__ == '__main__':
    main()