import argparse
from contextlib import asynccontextmanager

from typing import List
import numpy as np
import os

# from threading import Thread
from time import sleep

# import torch
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse

from inferencing.inference import Inferencer, Statistics
from service.frame_collector import FrameCollector, LastFrameCollector
from service.logger_service import LoggerService
from service.metric_pushgateway import MetricPusher

statistics = Statistics()


class App:
    def __init__(self) -> None:
        self.logger = LoggerService()
        self.logger().warning("Initialization of application")
        # self.trainer = Trainer(train_report_rate=5)

    def run(self) -> None:
        uvicorn.run(app="app:App.webserver_factory", factory=True, reload=True)

    @staticmethod
    def webserver_factory() -> FastAPI:
        parser: argparse.ArgumentParser = argparse.ArgumentParser()
        parser.add_argument(
            "-d", "--device", help="device", default=os.environ.get("device")
        )
        parser.add_argument(
            "-vp",
            "--video_path",
            help="Video path file or url",
            default=os.environ.get("video_path"),
        )
        parser.add_argument(
            "-b",
            "--batch_size",
            help="Batch size",
            default=os.environ.get("batch_size"),
            type=int,
        )
        parsed_args: argparse.Namespace = parser.parse_args()

        print(f"Run on {parsed_args.device}")
        print(f"with video of {parsed_args.video_path}")
        print(f"and batch size of {parsed_args.batch_size}")

        frame: List[np.ndarray] = []

        @asynccontextmanager
        async def deepengine(app: FastAPI):
            collector = LastFrameCollector(parsed_args.video_path)
            collector.start()

            #  Spawn thread of CCTV monitoring and tracking
            metricspusher = MetricPusher(gateway_address="pushgateway:9091")
            inferencer = Inferencer(
                framecollector=collector,
                batch_size=parsed_args.batch_size,
                metricspusher=None,
                # metricspusher=metricspusher,
                frame=frame,
            )
            inferencer.run(device=parsed_args.device, statistics=statistics)
            yield
            inferencer.stop()
            collector.stop()

        app = FastAPI(lifespan=deepengine)
        app.mount("/static", StaticFiles(directory="src/static"), name="static")

        @app.get("/status")
        def status_path():
            return "Hello world, I am online"

        @app.get("/streaming")
        def streaming_path():
            def iterfile():
                yield frame[0].tobytes()
            return StreamingResponse(iterfile(), media_type="video/mjpeg")

        return app

    # def run_train(self, device):
    #     self.logger().warning(f"Run on {device}")
    #     self.trainer.run_trainer(device=device)
