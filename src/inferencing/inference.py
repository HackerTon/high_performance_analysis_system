import time
from typing import Any, Callable, List, Optional, Set

import numpy as np
import torch
from copy import deepcopy
from deep_sort_realtime.deep_sort.track import Track
from deep_sort_realtime.deepsort_tracker import DeepSort
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    RetinaNet_ResNet50_FPN_V2_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
    retinanet_resnet50_fpn_v2,
)

from service.frame_collector import FrameCollector
from service.logger_service import LoggerService
from service.metric_pushgateway import MetricPusher


class Statistics:
    def __init__(self) -> None:
        self.number_of_person: int = 0
        self.fps: float = 0


class Inferencer:
    def __init__(
        self,
        framecollector: FrameCollector,
        metricspusher: Optional[MetricPusher],
        batch_size,
        box_score_thresh=0.9,
    ) -> None:
        self._initModelAndProcess(box_score_thresh)
        self.tracker = DeepSort(max_age=30)
        self.running = True
        self.metricspusher = metricspusher
        self.framecollector = framecollector
        self.batch_size = batch_size

    def _initModelAndProcess(self, box_score_thresh: float):
        self.weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        self.preprocess = self.weights.transforms()
        self.model = fasterrcnn_mobilenet_v3_large_fpn(
            weights=self.weights,
            box_score_thresh=box_score_thresh,
        )
        self.model = self.model.eval()

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

    @staticmethod
    def check_and_delete_unique(condition_function: Callable, array: List[Any]):
        for idx in range(len(array)):
            if condition_function(array[idx]):
                array.pop(idx)
                break

    def process_each_frame(self, prediction, img, human_set, number_of_person):
        only_human = []
        for i in range(len(prediction["boxes"])):
            if prediction["labels"][i] == 1:
                only_human.append(
                    (
                        self.convertToLTWH(prediction["boxes"][i]).numpy(),
                        prediction["labels"][i].cpu().numpy(),
                        prediction["scores"][i].cpu().numpy(),
                    ),
                )

        tracks: List[Track] = self.tracker.update_tracks(only_human, frame=img)
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()
            centroid = self.findCentroid(ltrb)
            x, y = centroid[0], centroid[1]

            if x > 0 and y > 0 and not track_id in human_set:
                human_set.add(track_id)
                number_of_person += 1
        return number_of_person

    def infer(self, device, statistics: Statistics):
        LoggerService().logger.warn(f"Inference running on {device}")
        # self.model = torch.jit.script(self.model.to(device))
        # self.preprocess = torch.jit.script(self.preprocess.to(device))    
        self.model = self.model.to(device)
        self.preprocess = self.preprocess.to(device)

        number_person = 0
        human_set: Set = set()

        while self.running:
            initial_time = time.time()
            images: List[Any] = self.framecollector.get_earliest_batch(self.batch_size)

            original_images = deepcopy(images)

            if images is None:
                continue

            for i in range(len(images)):
                image = torch.tensor(images[i])
                image = torch.permute(image[..., [2, 1, 0]], [2, 0, 1])
                image = image.to(device)
                images[i] = self.preprocess(image)

            with torch.no_grad():
                prediction = self.model(images)[1]
                for idx in range(len(images)):
                    number_person = self.process_each_frame(
                        prediction=prediction[idx],
                        img=original_images[idx],
                        human_set=human_set,
                        number_of_person=number_person,
                    )

                    if self.metricspusher != None:
                        self.metricspusher.push(
                            number_of_person=number_person,
                            latency=(time.time() - initial_time) / self.batch_size,
                            frame_left=self.framecollector.get_frames_left()
                        )
