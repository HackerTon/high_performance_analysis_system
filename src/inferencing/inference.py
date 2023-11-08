from multiprocessing import Queue
import time
from threading import Thread
from typing import Any, Callable, List, Optional, Union

import numpy as np
import torch
from deep_sort_realtime.deep_sort.track import Track
from deep_sort_realtime.deepsort_tracker import DeepSort
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    RetinaNet_ResNet50_FPN_V2_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
    retinanet_resnet50_fpn_v2,
)
from torchvision.utils import draw_bounding_boxes, draw_keypoints
from torchvision.io import encode_jpeg

from service.frame_collector import LastFrameCollector
from service.logger_service import LoggerService
from service.metric_pushgateway import MetricPusher


class Statistics:
    def __init__(self) -> None:
        self.number_of_person: int = 0
        self.fps: float = 0


class Inferencer:
    def __init__(
        self,
        framecollector: LastFrameCollector,
        frame: List[Union[np.ndarray, None]],
        metricspusher: Optional[MetricPusher] = None,
        box_score_thresh=0.9,
    ) -> None:
        self._initModelAndProcess(box_score_thresh)
        self.tracker = DeepSort(max_age=10)
        self.running = True
        self.metricspusher = metricspusher
        self.framecollector = framecollector
        self.frame = frame

        self.human_set_right = []
        self.human_set_left = []
        self.human_set_tracked_right = []
        self.human_set_tracked_left = []

    def _initModelAndProcess(self, box_score_thresh: float):
        self.weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        self.preprocess = self.weights.transforms()
        self.model = fasterrcnn_mobilenet_v3_large_fpn(
            weights=self.weights,
            box_score_thresh=box_score_thresh,
        )
        self.model = self.model.eval()

    @staticmethod
    def bgr2rgb(image: torch.Tensor):
        return image[..., [2, 1, 0]]

    @staticmethod
    def hwc2chw(image: torch.Tensor):
        return torch.permute(image, [0, 2, 3, 1])

    @staticmethod
    def chw2hwc(image: torch.Tensor):
        return torch.permute(image, [0, 3, 1, 2])

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

    def process_each_frame(self, prediction, img):
        filter_only_human = []
        for i in range(len(prediction["boxes"])):
            if prediction["labels"][i] == 1:
                filter_only_human.append(
                    (
                        self.convertToLTWH(prediction["boxes"][i]).numpy(),
                        prediction["labels"][i].cpu().numpy(),
                        prediction["scores"][i].cpu().numpy(),
                    ),
                )

        tracks: List[Track] = self.tracker.update_tracks(
            filter_only_human,
            frame=img.cpu().numpy(),
        )

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()
            centroid = self.findCentroid(ltrb)
            x, y = centroid[0], centroid[1]
            x, y, velocity_x, velocity_y, _, _, _, _ = track.mean

            moving_to_right = velocity_x > 0
            cross_middle_line = x > (img.shape[1] // 2)

            if (
                not cross_middle_line
                and track_id not in self.human_set_left
                and track_id not in self.human_set_tracked_left
            ):
                self.human_set_left.append(track_id)
            elif (
                cross_middle_line
                and track_id in self.human_set_left
                and track_id not in self.human_set_tracked_left
            ):
                self.human_set_tracked_left.append(track_id)
            elif (
                cross_middle_line
                and track_id not in self.human_set_right
                and track_id not in self.human_set_tracked_right
            ):
                self.human_set_right.append(track_id)
            elif (
                not cross_middle_line
                and track_id in self.human_set_right
                and track_id not in self.human_set_tracked_right
            ):
                self.human_set_tracked_right.append(track_id)

    def run(self, device: str, queue: Queue):
        self.thread = Thread(
            target=self.infer,
            args=[device, queue],
            name="inference",
        )

    def infer(self, device: str, queue: Queue):
        LoggerService().logger.warn(f"Inference running on {device}")
        self.model = self.model.to(device)
        self.preprocess = self.preprocess.to(device)

        while self.running:
            image: np.ndarray = queue.get()

            if image is None:
                continue

            initial_time = time.time()
            images = torch.tensor(np.array([image]))
            images = images.to(device=device)
            input_images = self.bgr2rgb(images)
            input_images = self.chw2hwc(input_images)
            normalized_images = self.preprocess(input_images)

            with torch.no_grad():
                prediction = self.model(normalized_images)
                for idx in range(input_images.shape[0]):
                    self.process_each_frame(
                        prediction=prediction[idx],
                        img=images[idx],
                    )

                    # Filter only human to be drawn
                    only_idx = torch.where(prediction[idx]["labels"] == 1)
                    only_human_bbox = prediction[idx]["boxes"][only_idx]

                    visualization_image = draw_bounding_boxes(
                        image=input_images[idx],
                        boxes=only_human_bbox,
                        colors="red",
                    )

                    # Draw middle line
                    full_height = input_images.shape[2]
                    half_width = input_images.shape[3] // 2

                    visualization_image = draw_keypoints(
                        visualization_image,
                        torch.Tensor([[[half_width, 0], [half_width, full_height]]]),
                        connectivity=[(0, 1)],
                    )

                    encoded_image = encode_jpeg(visualization_image)
                    self.frame[0] = encoded_image.numpy()

            if self.metricspusher != None:
                self.metricspusher.push(
                    latency=(time.time() - initial_time),
                    person_right_to_left=len(self.human_set_tracked_right),
                    person_left_to_right=len(self.human_set_tracked_left),
                )
            else:
                print(f"Right to left: {len(self.human_set_tracked_right)}")
                print(f"Left to right: {len(self.human_set_tracked_left)}")
                print(f"Latency: {(time.time() - initial_time)}")
