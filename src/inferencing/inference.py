import time
from typing import List

import cv2
import numpy as np
import torch
from deep_sort_realtime.deep_sort.track import Track
from deep_sort_realtime.deepsort_tracker import DeepSort
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
)


class Statistics:
    def __init__(self) -> None:
        self.number_of_person: int = 0
        self.fps: float = 0


class Inferencer:
    def __init__(self, device, box_score_thresh=0.9) -> None:
        self.weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
        self.preprocess = self.weights.transforms().to(device)
        self.model = fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=self.weights, box_score_thresh=box_score_thresh
        )
        self.model = self.model.eval().to(device=device)
        self.tracker = DeepSort(max_age=10)
        self.running = True

    @staticmethod
    def findCentroid(bounding_box):
        xmin, ymin, xmax, ymax = bounding_box
        return xmin + ((xmax - xmin) / 2), ymin + ((ymax - ymin) / 2)

    @staticmethod
    def convertToLTWH(bounding_box):
        width = bounding_box[2] - bounding_box[0]
        height = bounding_box[3] - bounding_box[1]
        return torch.Tensor([bounding_box[0], bounding_box[1], width, height])

    def stop(self):
        self.running = False

    def infer(self, capture_url, statistics: Statistics):  # type: ignore
        number_person = 0
        human_set = {}
        cam = cv2.VideoCapture(capture_url)
        while self.running:
            initial_time = time.time()
            is_running, frame = cam.read()

            img = torch.permute(torch.Tensor(frame[:, :, [2, 1, 0]]), [2, 0, 1]).to(
                torch.uint8
            )
            batch = self.preprocess(img).unsqueeze(0)

            with torch.no_grad():
                prediction = self.model(batch)[0]
                only_human = []
                for i in range(len(prediction["boxes"])):
                    if prediction["labels"][i] == 1:
                        only_human.append(
                            (
                                self.convertToLTWH(prediction["boxes"][i]).numpy(),
                                prediction["labels"][i].numpy(),
                                prediction["scores"][i].numpy(),
                            ),
                        )

                img = torch.permute(batch[0], [1, 2, 0]).numpy()[..., [2, 1, 0]]
                img = (img * 255).astype(np.uint8)

                tracks: List[Track] = self.tracker.update_tracks(only_human, frame=img)
                for track in tracks:
                    if not track.is_confirmed():
                        continue

                    track_id = track.track_id
                    ltrb = track.to_ltrb()
                    centroid = self.findCentroid(ltrb)

                    if centroid[0] > 0 and centroid[1] > 300 and track_id in human_set:
                        number_person += 1

            statistics.number_of_person = number_person
            statistics.fps = round(1 / (time.time() - initial_time), ndigits=3)
