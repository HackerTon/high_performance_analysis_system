import time
from copy import deepcopy
from typing import Any, Callable, List, Optional, Set, Union

import cv2
import numpy as np
import torch
from deep_sort_realtime.deep_sort.track import Track
from deep_sort_realtime.deepsort_tracker import DeepSort
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    RetinaNet_ResNet50_FPN_V2_Weights, fasterrcnn_mobilenet_v3_large_fpn,
    retinanet_resnet50_fpn_v2)
from torchvision.utils import draw_bounding_boxes

from service.frame_collector import FrameCollector, LastFrameCollector
from service.logger_service import LoggerService
from service.metric_pushgateway import MetricPusher


class Statistics:
    def __init__(self) -> None:
        self.number_of_person: int = 0
        self.fps: float = 0


class Inferencer:
    def __init__(
        self,
        framecollector: Union[FrameCollector, LastFrameCollector],
        metricspusher: Optional[MetricPusher],
        batch_size,
        box_score_thresh=0.9,
    ) -> None:
        self._initModelAndProcess(box_score_thresh)
        self.tracker = DeepSort(max_age=5)
        self.running = True
        self.metricspusher = metricspusher
        self.framecollector = framecollector
        self.batch_size = batch_size

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

    def infer(self, device, statistics: Statistics):
        LoggerService().logger.warn(f"Inference running on {device}")
        # self.model = torch.jit.script(self.model.to(device))
        # self.preprocess = torch.jit.script(self.preprocess.to(device))
        self.model = self.model.to(device)
        self.preprocess = self.preprocess.to(device)

        while self.running:
            initial_time = time.time()
            images: List[Any] = self.framecollector.get_earliest_batch(self.batch_size)

            if images is None:
                # print('frame not available')
                continue

            images = [images]
            original_images = deepcopy(images)

            for i in range(len(images)):
                image = torch.tensor(images[i])
                image = torch.permute(image[..., [2, 1, 0]], [2, 0, 1])
                image = image.to(device)
                images[i] = self.preprocess(image)

            with torch.no_grad():
                prediction = self.model(images)
                for idx in range(len(images)):
                    self.process_each_frame(
                        prediction=prediction[idx],
                        img=original_images[idx],
                    )

                    # box = draw_bounding_boxes(
                    #     torch.permute(
                    #         torch.Tensor(original_images[idx]).to(torch.uint8),
                    #         [2, 0, 1],
                    #     ),
                    #     boxes=prediction[idx]["boxes"],
                    #     colors="red",
                    # )

                    # full_height = box.shape[1]
                    # half_width = box.shape[2] // 2

                    # box = torch.permute(box, [1, 2, 0]).numpy().astype(np.uint8)
                    # box = cv2.line(
                    #     box,
                    #     (half_width, 0),
                    #     (half_width, full_height),
                    #     (255, 255, 0),
                    #     2,
                    # )

                    # cv2.imshow("video", box)
                    # if cv2.waitKey(1) & 0xFF == ord("q"):
                    #     break

            if self.metricspusher != None:
                print(f"Right to left: {len(self.human_set_tracked_right)}")
                print(f"Left to right: {len(self.human_set_tracked_left)}")
                self.metricspusher.push(
                    latency=(time.time() - initial_time) / self.batch_size,
                    person_right_to_left=len(self.human_set_tracked_right),
                    person_left_to_right=len(self.human_set_tracked_left),
                )
            # else:
            # print(f"Right to left: {len(self.human_set_tracked_right)}")
            # print(f"Left to right: {len(self.human_set_tracked_left)}")
            # print("STATISTICS")
            # print(number_person)
            # print((time.time() - initial_time) / self.batch_size)
            # print(self.framecollector.get_frames_left())
