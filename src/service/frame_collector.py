from threading import Thread
import cv2
from typing import List, Any
from copy import deepcopy


class FrameCollector:
    def __init__(self, video_path: str) -> None:
        self.video_path = video_path
        self.batch_frame = []
        self.running = True

    def start(self):
        self.thread: Thread = Thread(target=self._start_collection, args=())
        self.thread.start()

    def get_earliest_batch(self, range_of_images) -> Any:
        if len(self.batch_frame) != 0:
            list_images = self.batch_frame[:range_of_images]
            del self.batch_frame[:range_of_images]
            return list_images
        else:
            return None

    def stop(self):
        self.running = False
        self.thread.join()

    def _start_collection(self):
        cam = cv2.VideoCapture(self.video_path)
        while self.running:
            is_running, frame = cam.read()
            if not is_running:
                self.running = False
                break

            self.batch_frame.append(frame)

        cam.release()
