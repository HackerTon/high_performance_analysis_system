# from threading import Thread
from multiprocessing import Process
from typing import Any

import cv2


class FrameCollector:
    def __init__(self, video_path: str) -> None:
        self.video_path = video_path
        self.batch_frame = []
        self.running = True

    def start(self):
        self.process: Process = Process(target=self._start_collection, args=())
        self.process.start()

    def get_earliest_batch(self, range_of_images) -> Any:
        if len(self.batch_frame) != 0:
            list_images = self.batch_frame[:range_of_images]
            del self.batch_frame[:range_of_images]
            return list_images
        else:
            return None

    def get_frames_left(self) -> int:
        return len(self.batch_frame)

    def stop(self):
        self.running = False

    def _start_collection(self):
        cam = cv2.VideoCapture(self.video_path)
        while self.running:
            is_running, frame = cam.read()
            if not is_running:
                self.running = False
                break
            self.batch_frame.append(frame)
        cam.release()
