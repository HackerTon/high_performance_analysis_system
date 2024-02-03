import time
from multiprocessing import Queue
from multiprocessing.connection import Connection
from threading import Thread
from typing import List, Optional, Union

import numpy as np
import torch
from inferencing.maskprocessor import MaskProcessor
from inferencing.model import MultiNet, BackboneType
from service.frame_collector import LastFrameCollector, MockUpCollector
from service.logger_service import LoggerService
from service.metric_pushgateway import MetricPusher
from torchvision import transforms
from torchvision.io import encode_jpeg
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


class OCRInferencer:
    def __init__(
        self,
        framecollector: Union[LastFrameCollector, MockUpCollector],
        frame: List[Union[np.ndarray, None]],
        metricspusher: Optional[MetricPusher] = None,
        box_score_thresh=0.98,
    ) -> None:
        self._initModelAndProcess(box_score_thresh)
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

        license_plate_processor = MaskProcessor(
            text_processor=self.ocrProcessor,
            text_model=self.ocrModel,
            detection_model=self.localizerModel,
            detection_processor=self.localizerProcessor,
        )

        while self.running:
            image: np.ndarray = parentConnection.recv()

            if image is None:
                continue

            initial_time = time.time()
            images = (torch.tensor(image) / 255).to(torch.float32)
            images = images[..., [2, 1, 0]]
            # Flip the WH to HW and then
            images = torch.permute(images, [2, 0, 1])
            images = images.to(device=device)
            print(
                license_plate_processor.obtain_text(single_image=images, device=device),
                flush=True,
            )
            print(f"Latency: {(time.time() - initial_time)}", flush=True)
