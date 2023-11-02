from contextlib import asynccontextmanager
from threading import Thread
from time import sleep

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

from inferencing.inference import Inferencer, Statistics
from service.frame_collector import FrameCollector
from service.logger_service import LoggerService

statistics = Statistics()


class App:
    def __init__(self) -> None:
        self.logger = LoggerService()
        self.logger().warning("Initialization of application")
        # self.trainer = Trainer(train_report_rate=5)

    def run(self, device, video_path, batch_size) -> None:
        collector = FrameCollector(video_path)
        collector.start()
        #  Spawn thread of CCTV monitoring and tracking
        inferencer = Inferencer(framecollector=collector, batch_size=batch_size)
        inferencer.infer(device=device, statistics=statistics)
        collector.stop()

    # @staticmethod
    # def webserver_factory() -> FastAPI:
    #     @asynccontextmanager
    #     async def deepengine(app: FastAPI):
    #         collector = FrameCollector('./video2.mp4')
    #         collector.start()

    #         #  Spawn thread of CCTV monitoring and tracking
    #         device = "cuda" if torch.cuda.is_available() else "cpu"
    #         inferencer = Inferencer(device=device, framecollector=collector)
    #         inference_thread: Thread = Thread(
    #             target=inferencer.infer,
    #             args=[statistics],
    #         )
    #         inference_thread.start()
    #         yield
    #         inferencer.stop()
    #         inference_thread.join()
    #         collector.stop()

    #     app = FastAPI(lifespan=deepengine)
    #     @app.get("/")
    #     async def root():
    #         return "hello world"
    #     @app.get("/metrics", response_class=PlainTextResponse)
    #     async def metrics():
    #         return "\n".join(
    #             [
    #                 "# HELP number_of_person",
    #                 f"number_of_person {statistics.number_of_person}",
    #                 "# HELP fps",
    #                 f"fps {statistics.fps}",
    #             ]
    #         )
    #     return app

    # def run_train(self, device):
    #     self.logger().warning(f"Run on {device}")
    #     self.trainer.run_trainer(device=device)
