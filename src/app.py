import argparse
from contextlib import asynccontextmanager

from typing import List
import numpy as np
import os

# from threading import Thread
from time import sleep

# import torch
import uvicorn
from fastapi import FastAPI, Header
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, Response

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
        parser: argparse.ArgumentParser = argparse.ArgumentParser()
        parser.add_argument(
            "-d",
            "--device",
            help="device",
            default=os.environ.get("device"),
        )
        parser.add_argument(
            "-r",
            "--reload",
            help="reload",
            default=False,
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
        uvicorn.run(
            app="app:App.webserver_factory",
            factory=True,
            host="0.0.0.0",
            port=8000,
            reload=parsed_args.reload,
        )

    @staticmethod
    def webserver_factory() -> FastAPI:
        parser: argparse.ArgumentParser = argparse.ArgumentParser()
        parser.add_argument(
            "-d",
            "--device",
            help="device",
            default=os.environ.get("device"),
        )
        parser.add_argument(
            "-r",
            "--reload",
            help="reload",
            default=False,
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
                metricspusher=metricspusher,
                frame=frame,
            )
            inferencer.run(device=parsed_args.device, statistics=statistics)
            yield
            LoggerService().logger.warning("Stopping inferencer")
            inferencer.stop()
            LoggerService().logger.warning("Stopping collector")
            collector.stop()
            LoggerService().logger.warning("Done")

        app = FastAPI(lifespan=deepengine)

        @app.get("/status")
        def status_path():
            return "Hello world, I am online"

        @app.get("/")
        async def streaming_path():
            def iterfile():
                while True:
                    try:
                        yield b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame[
                            0
                        ].tobytes() + b"\r\n"
                        sleep(0.05)
                    except:
                        sleep(1)

            return StreamingResponse(
                iterfile(), media_type="multipart/x-mixed-replace;boundary=frame"
            )

        return app

    # def run_train(self, device):
    #     self.logger().warning(f"Run on {device}")
    #     self.trainer.run_trainer(device=device)
