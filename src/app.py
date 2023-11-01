from contextlib import asynccontextmanager
from threading import Thread
from typing import Union

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

from inferencing.inference import Inferencer, Statistics
from service.logger_service import LoggerService
from trainer.trainer import Trainer

statistics = Statistics()


class App:
    def __init__(self) -> None:
        self.logger = LoggerService()
        self.logger().warning("Initialization of application")
        # self.trainer = Trainer(train_report_rate=5)

    def run(self, reload) -> None:
        # Start webserver
        uvicorn.run(
            app="app:App.webserver_factory",
            factory=True,
            reload=reload,
            port=9090,
            host="0.0.0.0",
        )

    @staticmethod
    def webserver_factory() -> FastAPI:
        @asynccontextmanager
        async def deepengine(app: FastAPI):
            # Spawn thread of CCTV monitoring and tracking
            # device = "cuda" if torch.cuda.is_available() else "cpu"
            device = 'cuda'
            inferencer = Inferencer(device=device)
            inference_thread: Thread = Thread(
                target=inferencer.infer,
                args=(
                    "http://tamperehacklab.tunk.org:38001/nphMotionJpeg?Resolution=640x480&Quality=Clarity",
                    statistics,
                ),
            )
            inference_thread.start()
            yield
            inferencer.stop()
            inference_thread.join()

        app = FastAPI(lifespan=deepengine)

        @app.get("/")
        async def root():
            return "hello world"

        @app.get("/metrics", response_class=PlainTextResponse)
        async def metrics():
            return "\n".join(
                [
                    "# HELP number_of_person",
                    f"number_of_person {statistics.number_of_person}",
                    "# HELP fps",
                    f"fps {statistics.fps}",
                ]
            )

        return app

    # def run_train(self, device):
    #     self.logger().warning(f"Run on {device}")
    #     self.trainer.run_trainer(device=device)
