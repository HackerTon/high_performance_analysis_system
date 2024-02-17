import time
from multiprocessing.connection import Connection
from threading import Thread
from typing import Optional
import cv2

import numpy as np
import torch
from inferencing.number_plate_detector import NumberPlateProcessor
from inferencing.model import MultiNet, BackboneType
from service.logger_service import LoggerService
from service.metric_pushgateway import MetricPusher
from torchvision import transforms
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from torchvision.transforms import ToTensor


class OCRInferencer:
    def __init__(self, device: str, metricspusher: Optional[MetricPusher] = None) -> None:
        self._initModelAndProcess(device=device)
        self.running = True
        self.metricspusher = metricspusher

    def _initModelAndProcess(self, device: str):
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
            torch.load("./src/81_model.pt", map_location=torch.device(device))
        )

    def stop(self):
        self.running = False

    def run(
        self,
        device: str,
        frame_down_connection: Connection,
        streaming_up_connection: Connection,
    ):
        self.thread = Thread(
            target=self.infer,
            args=[device, frame_down_connection, streaming_up_connection],
            name="inference",
        )

    def infer(
        self,
        device: str,
        frame_down_connection: Connection,
        streaming_up_connection: Connection,
    ):
        LoggerService().logger.warn(f"Inference running on {device}")
        self.ocrModel.eval()
        self.localizerModel.eval()

        self.localizerModel = self.localizerModel.to(device)
        self.ocrModel = self.ocrModel.to(device)
        self.localizerProcessor = self.localizerProcessor.to(device)

        license_plate_processor = NumberPlateProcessor(
            text_processor=self.ocrProcessor,
            text_model=self.ocrModel,
            detection_model=self.localizerModel,
            detection_processor=self.localizerProcessor,
        )

        to_tensor = ToTensor()
        while self.running:
            image: np.ndarray = frame_down_connection.recv()

            if image is None:
                continue

            initial_time = time.time()
            images = to_tensor(image)
            images = images[[2, 1, 0]]
            images = images.to(device=device)
            result = license_plate_processor.obtain_text(
                single_image=images, device=device
            )

            if result != "":
                image = cv2.putText(
                    image,
                    result,
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            # Write latency into image
            image = cv2.putText(
                image,
                f"Latency: {(time.time() - initial_time):0.4f}",
                (0, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            streaming_up_connection.send_bytes(cv2.imencode(".jpg", image)[1].tobytes())
