# from threading import Thread
from multiprocessing import Lock, Process, Queue
from multiprocessing.connection import Connection

import cv2
import time


class LastFrameCollector:
    def __init__(self, video_path: str) -> None:
        self.video_path = video_path
        self.batch_frame = None
        self.running = True
        self.lock = Lock()

    def start(self, childCollection: Connection):
        self.process: Process = Process(
            target=self._start_collection,
            args=[childCollection],
        )

    def stop(self):
        self.running = False

    def _start_collection(self, childConnection: Connection):
        cam = cv2.VideoCapture(self.video_path)
        while self.running:
            frame_running, frame = cam.read()
            if not frame_running:
                self.batch_frame = None
                self.stop()
                break
            # queue.put(frame)
            childConnection.send(frame)
            time.sleep(0.01)
        cam.release()


class MockUpCollector:
    def __init__(self, image_path: str) -> None:
        self.image_path = image_path
        self.batch_frame = None
        self.running = True
        self.lock = Lock()

    def start(self, childCollection: Connection):
        self.process: Process = Process(
            target=self._start_collection,
            args=[childCollection],
        )

    def stop(self):
        self.running = False

    def _start_collection(self, childConnection: Connection):
        while self.running:
            frame = cv2.imread(self.image_path)
            childConnection.send(frame)
            time.sleep(0.01)
