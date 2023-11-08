# from threading import Thread
from multiprocessing import Lock, Process, Queue

import cv2


class LastFrameCollector:
    def __init__(self, video_path: str) -> None:
        self.video_path = video_path
        self.batch_frame = None
        self.running = True
        self.lock = Lock()

    def start(self, queue: Queue):
        self.process: Process = Process(target=self._start_collection, args=[queue])

    def stop(self):
        self.running = False

    def _start_collection(self, queue: Queue):
        cam = cv2.VideoCapture(self.video_path)
        while self.running:
            frame_running, frame = cam.read()
            if not frame_running:
                self.batch_frame = None
                self.stop()
                break
            queue.put(frame)
        cam.release()
