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


    def add_batch_frames(self, batch_frames: List[Any]):
        self.batch_frame.append(deepcopy(batch_frames))
        batch_frames.clear()

    def get_earliest_batch(self) -> Any:
        if len(self.batch_frame) is not 0:
            return self.batch_frame.pop(0)
        else:
            return None

    def stop(self):
        self.running = False
        self.thread.join()

    def _start_collection(self):
        cam = cv2.VideoCapture(self.video_path)

        stored_frame = []
        while self.running:
            is_running, frame = cam.read()

            if not is_running:
                self.batch_frame.append(stored_frame)
                self.running = False
                break

            stored_frame.append(frame)
            if len(stored_frame) == 30:
                self.add_batch_frames(stored_frame)

        cam.release()

