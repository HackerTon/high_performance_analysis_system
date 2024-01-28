import time
from enum import Enum
from multiprocessing import Queue
from multiprocessing.connection import Connection
from threading import Thread
from typing import Any, Callable, List, Optional, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from deep_sort_realtime.deep_sort.track import Track
from deep_sort_realtime.deepsort_tracker import DeepSort
from service.frame_collector import LastFrameCollector, MockUpCollector
from service.logger_service import LoggerService
from service.metric_pushgateway import MetricPusher
from torchvision import transforms
from torchvision.io import encode_jpeg
from torchvision.models import ResNet34_Weights, ResNet50_Weights, resnet34, resnet50
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.transforms.functional import resize
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


class BackboneType(Enum):
    RESNET34 = 1
    RESNET50 = 2


class MultiNet(nn.Module):
    def __init__(self, numberClass, backboneType: BackboneType):
        super().__init__()

        if backboneType == BackboneType.RESNET34:
            backbone = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
            self.backbone = create_feature_extractor(
                backbone,
                {
                    # "relu": "feat1",
                    "layer1": "feat2",
                    "layer2": "feat3",
                    "layer3": "feat4",
                    "layer4": "feat5",
                },
            )
        elif backboneType == BackboneType.RESNET50:
            backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            self.backbone = create_feature_extractor(
                backbone,
                {
                    # "relu": "feat1",
                    "layer1": "feat2",
                    "layer2": "feat3",
                    "layer3": "feat4",
                    "layer4": "feat5",
                },
            )
        else:
            raise Exception(f"No {backboneType}")

        with torch.no_grad():
            outputs_prediction = self.backbone(torch.rand([1, 3, 256, 256])).values()
            backbone_dimensions = [output.size(1) for output in outputs_prediction]

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.upsampling_2x_bilinear = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upsampling_4x_bilinear = nn.UpsamplingBilinear2d(scale_factor=4)
        self.upsampling_8x_bilinear = nn.UpsamplingBilinear2d(scale_factor=8)
        self.conv5_1x1 = nn.Conv2d(
            in_channels=backbone_dimensions[-1],
            out_channels=256,
            kernel_size=1,
        )
        self.conv5_3x3_1 = nn.Conv2d(
            in_channels=256,
            out_channels=128,
            kernel_size=3,
            padding=1,
        )
        self.conv5_3x3_2 = nn.Conv2d(
            in_channels=128,
            out_channels=numberClass,
            kernel_size=3,
            padding=1,
        )
        self.conv4_1x1 = nn.Conv2d(
            in_channels=backbone_dimensions[-2],
            out_channels=256,
            kernel_size=1,
        )
        self.conv4_3x3_1 = nn.Conv2d(
            in_channels=256,
            out_channels=128,
            kernel_size=3,
            padding=1,
        )
        self.conv4_3x3_2 = nn.Conv2d(
            in_channels=128,
            out_channels=numberClass,
            kernel_size=3,
            padding=1,
        )
        self.conv3_1x1 = nn.Conv2d(
            in_channels=backbone_dimensions[-3],
            out_channels=256,
            kernel_size=1,
        )
        self.conv3_3x3_1 = nn.Conv2d(
            in_channels=256,
            out_channels=128,
            kernel_size=3,
            padding=1,
        )
        self.conv3_3x3_2 = nn.Conv2d(
            in_channels=128,
            out_channels=numberClass,
            kernel_size=3,
            padding=1,
        )
        self.conv2_1x1 = nn.Conv2d(
            in_channels=backbone_dimensions[-4],
            out_channels=256,
            kernel_size=1,
        )
        self.conv2_3x3_1 = nn.Conv2d(
            in_channels=256,
            out_channels=128,
            kernel_size=3,
            padding=1,
        )
        self.conv2_3x3_2 = nn.Conv2d(
            in_channels=128,
            out_channels=numberClass,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x):
        backbone_output = self.backbone(x)
        feat2, feat3, feat4, feat5 = (
            backbone_output["feat2"],
            backbone_output["feat3"],
            backbone_output["feat4"],
            backbone_output["feat5"],
        )

        conv5_mid = self.conv5_1x1(feat5).relu()
        conv5_prediction = self.conv5_3x3_1(conv5_mid).relu()
        conv5_prediction = self.conv5_3x3_2(conv5_prediction)

        conv4_lateral = self.conv4_1x1(feat4).relu()
        conv4_mid = conv4_lateral + self.upsampling_2x_bilinear(conv5_mid)
        conv4_prediction = self.conv4_3x3_1(conv4_mid).relu()
        conv4_prediction = self.conv4_3x3_2(conv4_prediction)

        conv3_lateral = self.conv3_1x1(feat3).relu()
        conv3_mid = conv3_lateral + self.upsampling_2x_bilinear(conv4_mid)
        conv3_prediction = self.conv3_3x3_1(conv3_mid).relu()
        conv3_prediction = self.conv3_3x3_2(conv3_prediction)

        conv2_lateral = self.conv2_1x1(feat2).relu()
        conv2_mid = conv2_lateral + self.upsampling_2x_bilinear(conv3_mid)
        conv2_prediction = self.conv2_3x3_1(conv2_mid).relu()
        conv2_prediction = self.conv2_3x3_2(conv2_prediction)

        final_prediction_5 = self.upsampling_8x_bilinear(conv5_prediction)
        final_prediction_4 = self.upsampling_4x_bilinear(conv4_prediction)
        final_prediction_3 = self.upsampling_2x_bilinear(conv3_prediction)
        final_prediction_2 = conv2_prediction

        return self.upsampling_4x_bilinear(
            final_prediction_5
            + final_prediction_4
            + final_prediction_3
            + final_prediction_2
        )


class OCRInferencer:
    def __init__(
        self,
        framecollector: Union[LastFrameCollector, MockUpCollector],
        frame: List[Union[np.ndarray, None]],
        metricspusher: Optional[MetricPusher] = None,
        box_score_thresh=0.98,
    ) -> None:
        self._initModelAndProcess(box_score_thresh)
        self.tracker = DeepSort(max_age=10)
        self.running = True
        self.metricspusher = metricspusher
        self.framecollector = framecollector
        self.frame = frame

    def _initModelAndProcess(self, box_score_thresh: float):
        self.localizerModel = MultiNet(
            numberClass=2, backboneType=BackboneType.RESNET50
        )
        self.localizerProcessor = transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )
        self.ocrProcessor = TrOCRProcessor.from_pretrained(
            "microsoft/trocr-base-printed"
        )
        self.ocrModel = VisionEncoderDecoderModel.from_pretrained(
            "microsoft/trocr-base-printed"
        )

        self.localizerModel.load_state_dict(
            torch.load("./src/81_model.pt", map_location=torch.device("cpu"))
        )
        self.ocrModel.eval()
        self.localizerModel.eval()

        # self.weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
        # self.preprocess = self.weights.transforms()
        # self.model = fasterrcnn_resnet50_fpn_v2(
        #     weights=self.weights,
        #     box_score_thresh=box_score_thresh,
        # )
        # self.model = self.model.eval()

    @staticmethod
    def predict_with_bounding_box(
        input_image: torch.Tensor,
        output: torch.Tensor,
        processor,
        model,
    ):
        image = (output.softmax(1)[0] > 0.5).numpy().astype(np.uint8)[0]
        image = cv2.morphologyEx(image, cv2.MORPH_DILATE, np.ones([3, 3]), iterations=5)

        # if debug:
        #     plt.imshow(image)

        contours = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        parsed_text = []

        for i in range(len(contours[0])):
            x, y, width, height = cv2.boundingRect(contours[0][i])
            # Rescale coordinate to original
            new_x = int(x * 608 / 256)
            new_y = int(y * 1080 / 256)
            new_x_2 = new_x + int(width * 608 / 256)
            new_y_2 = new_y + int(height * 1080 / 256)

            pixel_values = processor(
                transforms.ToPILImage()(
                    input_image[0, ..., new_y:new_y_2, new_x:new_x_2]
                ),
                return_tensors="pt",
            ).pixel_values

            # if debug:
            #     plt.figure()
            #     plt.imshow(torch.permute(pixel_values[0], [1, 2, 0]))

            generated_ids = model.generate(pixel_values)
            generated_text = processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]
            parsed_text.append(generated_text)

        return parsed_text

    def stop(self):
        self.running = False

    def run(self, device: str, parentConnection: Connection):
        self.thread = Thread(
            target=self.infer,
            args=[device, parentConnection],
            name="inference",
        )

    def infer(self, device: str, parentConnection: Connection):
        LoggerService().logger.warn(f"Inference running on {device}")
        self.localizerModel = self.localizerModel.to(device)
        self.ocrModel = self.ocrModel.to(device)
        self.localizerProcessor = self.localizerProcessor.to(device)

        while self.running:
            image: np.ndarray = parentConnection.recv()

            if image is None:
                continue

            initial_time = time.time()
            images = (torch.tensor(image) / 255).to(torch.float32)
            images = images[..., [2, 1, 0]]
            # Flip the WH to HW and then
            images = torch.permute(images, [2, 0, 1])
            images = images.to(device=device).unsqueeze(0)

            with torch.no_grad():
                resized_image = resize(images, [256, 256], antialias=False)
                normalized_images = self.localizerProcessor(resized_image)
                maskImage = self.localizerModel(normalized_images)
                print(
                    self.predict_with_bounding_box(
                        images,
                        maskImage,
                        processor=self.ocrProcessor,
                        model=self.ocrModel,
                    ),
                    flush=True,
                )

            print(f"Latency: {(time.time() - initial_time)}", flush=True)

            # print(
            #     self.predict_with_bounding_box(
            #         input_image,
            #         output,
            #         processor=self.ocrProcessor,
            #         model=self.ocrModel,
            #         debug=True,
            #     )
            # )

            # with torch.no_grad():
            #     prediction = self.model(normalized_images)
            #     for idx in range(input_images.shape[0]):
            #         # self.process_each_frame(
            #         #     prediction=prediction[idx],
            #         #     img=images[idx],
            #         # )

            #         # Filter only human to be drawn
            #         only_idx = torch.where(prediction[idx]["labels"] == 1)
            #         only_human_bbox = prediction[idx]["boxes"][only_idx]

            #         visualization_image = draw_bounding_boxes(
            #             image=input_images[idx],
            #             boxes=only_human_bbox,
            #             colors="red",
            #         )

            #         # Draw middle line
            #         full_height = input_images.shape[2]
            #         half_width = input_images.shape[3] // 2

            #         visualization_image = draw_keypoints(
            #             visualization_image,
            #             torch.Tensor([[[half_width, 0], [half_width, full_height]]]),
            #             connectivity=[(0, 1)],
            #         )

            #         encoded_image = encode_jpeg(visualization_image)
            #         self.frame[0] = encoded_image.numpy()

            # if self.metricspusher != None:
            #     self.metricspusher.push(
            #         latency=(time.time() - initial_time),
            #         person_right_to_left=len(self.human_set_tracked_right),
            #         person_left_to_right=len(self.human_set_tracked_left),
            #     )
            # else:
            #     print(f"Right to left: {len(self.human_set_tracked_right)}")
            #     print(f"Left to right: {len(self.human_set_tracked_left)}")
            #     print(f"Latency: {(time.time() - initial_time)}")


# class Inferencer:
#     def __init__(
#         self,
#         framecollector: LastFrameCollector,
#         frame: List[Union[np.ndarray, None]],
#         metricspusher: Optional[MetricPusher] = None,
#         box_score_thresh=0.98,
#     ) -> None:
#         self._initModelAndProcess(box_score_thresh)
#         self.tracker = DeepSort(max_age=10)
#         self.running = True
#         self.metricspusher = metricspusher
#         self.framecollector = framecollector
#         self.frame = frame

#         self.human_set_right = []
#         self.human_set_left = []
#         self.human_set_tracked_right = []
#         self.human_set_tracked_left = []

#     def _initModelAndProcess(self, box_score_thresh: float):
#         self.weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
#         self.preprocess = self.weights.transforms()
#         self.model = fasterrcnn_resnet50_fpn_v2(
#             weights=self.weights,
#             box_score_thresh=box_score_thresh,
#         )
#         self.model = self.model.eval()

#     @staticmethod
#     def bgr2rgb(image: torch.Tensor):
#         return image[..., [2, 1, 0]]

#     @staticmethod
#     def hwc2chw(image: torch.Tensor):
#         return torch.permute(image, [0, 2, 3, 1])

#     @staticmethod
#     def chw2hwc(image: torch.Tensor):
#         return torch.permute(image, [0, 3, 1, 2])

#     @staticmethod
#     def findCentroid(bounding_box):
#         xmin, ymin, xmax, ymax = bounding_box
#         return xmin + ((xmax - xmin) / 2), ymin + ((ymax - ymin) / 2)

#     @staticmethod
#     def convertToLTWH(bounding_box):
#         width = bounding_box[2] - bounding_box[0]
#         height = bounding_box[3] - bounding_box[1]
#         return torch.Tensor([bounding_box[0], bounding_box[1], width, height])

#     def stop(self):
#         self.running = False

#     def process_each_frame(self, prediction, img):
#         filter_only_human = []
#         for i in range(len(prediction["boxes"])):
#             if prediction["labels"][i] == 1:
#                 filter_only_human.append(
#                     (
#                         self.convertToLTWH(prediction["boxes"][i]).numpy(),
#                         prediction["labels"][i].cpu().numpy(),
#                         prediction["scores"][i].cpu().numpy(),
#                     ),
#                 )

#         tracks: List[Track] = self.tracker.update_tracks(
#             filter_only_human,
#             frame=img.cpu().numpy(),
#         )

#         for track in tracks:
#             if not track.is_confirmed():
#                 continue

#             track_id = track.track_id
#             ltrb = track.to_ltrb()
#             centroid = self.findCentroid(ltrb)
#             x, y = centroid[0], centroid[1]
#             x, y, velocity_x, velocity_y, _, _, _, _ = track.mean

#             moving_to_right = velocity_x > 0
#             cross_middle_line = x > (img.shape[1] // 2)

#             if (
#                 not cross_middle_line
#                 and track_id not in self.human_set_left
#                 and track_id not in self.human_set_tracked_left
#             ):
#                 self.human_set_left.append(track_id)
#             elif (
#                 cross_middle_line
#                 and track_id in self.human_set_left
#                 and track_id not in self.human_set_tracked_left
#             ):
#                 self.human_set_tracked_left.append(track_id)
#             elif (
#                 cross_middle_line
#                 and track_id not in self.human_set_right
#                 and track_id not in self.human_set_tracked_right
#             ):
#                 self.human_set_right.append(track_id)
#             elif (
#                 not cross_middle_line
#                 and track_id in self.human_set_right
#                 and track_id not in self.human_set_tracked_right
#             ):
#                 self.human_set_tracked_right.append(track_id)

#     def run(self, device: str, parentConnection: Connection):
#         self.thread = Thread(
#             target=self.infer,
#             args=[device, parentConnection],
#             name="inference",
#         )

#     def infer(self, device: str, parentConnection: Connection):
#         LoggerService().logger.warn(f"Inference running on {device}")
#         self.model = self.model.to(device)
#         self.preprocess = self.preprocess.to(device)

#         while self.running:
#             image: np.ndarray = parentConnection.recv()

#             if image is None:
#                 continue

#             initial_time = time.time()
#             images = torch.tensor(np.array([image]))
#             images = images.to(device=device)
#             input_images = self.bgr2rgb(images)
#             input_images = self.chw2hwc(input_images)
#             normalized_images = self.preprocess(input_images)

#             with torch.no_grad():
#                 prediction = self.model(normalized_images)
#                 for idx in range(input_images.shape[0]):
#                     # self.process_each_frame(
#                     #     prediction=prediction[idx],
#                     #     img=images[idx],
#                     # )

#                     # Filter only human to be drawn
#                     only_idx = torch.where(prediction[idx]["labels"] == 1)
#                     only_human_bbox = prediction[idx]["boxes"][only_idx]

#                     visualization_image = draw_bounding_boxes(
#                         image=input_images[idx],
#                         boxes=only_human_bbox,
#                         colors="red",
#                     )

#                     # Draw middle line
#                     full_height = input_images.shape[2]
#                     half_width = input_images.shape[3] // 2

#                     visualization_image = draw_keypoints(
#                         visualization_image,
#                         torch.Tensor([[[half_width, 0], [half_width, full_height]]]),
#                         connectivity=[(0, 1)],
#                     )

#                     encoded_image = encode_jpeg(visualization_image)
#                     self.frame[0] = encoded_image.numpy()

#             if self.metricspusher != None:
#                 self.metricspusher.push(
#                     latency=(time.time() - initial_time),
#                     person_right_to_left=len(self.human_set_tracked_right),
#                     person_left_to_right=len(self.human_set_tracked_left),
#                 )
#             else:
#                 print(f"Right to left: {len(self.human_set_tracked_right)}")
#                 print(f"Left to right: {len(self.human_set_tracked_left)}")
#                 print(f"Latency: {(time.time() - initial_time)}")
