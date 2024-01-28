import os
from contextlib import asynccontextmanager
from multiprocessing import Pipe
from multiprocessing.connection import Connection
from time import sleep
from typing import List, Union

import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from inferencing.inference import OCRInferencer
from service.frame_collector import LastFrameCollector, MockUpCollector
from service.logger_service import LoggerService
from service.metric_pushgateway import MetricPusher

logger = LoggerService()
visualFrameQueue: List[Union[np.ndarray, None]] = [None]
parentConnection: Connection
childConnection: Connection
parentConnection, childConnection = Pipe()


device = os.getenv("DEVICE", "cpu")
video_path = os.getenv("VIDEO_PATH")

if video_path == None:
    logger.logger.warning("video_path ENV not provided")
    quit()

logger().warning("Initialization of application")


@asynccontextmanager
async def deepengine(app: FastAPI):
    collector = MockUpCollector(image_path=video_path)
    # metricspusher = MetricPusher(gateway_address="pushgateway:9091")
    inferencer = OCRInferencer(
        framecollector=collector,
        # metricspusher=metricspusher,
        frame=visualFrameQueue,
    )
    inferencer.run(device=device, parentConnection=parentConnection)
    collector.start(childCollection=childConnection)
    inferencer.thread.start()
    collector.process.start()
    yield
    collector.running = False
    inferencer.running = False
    inferencer.thread.join()
    collector.process.join()
    LoggerService().logger.warning("Done stopping inference and collector")


app = FastAPI(lifespan=deepengine)


@app.get("/status")
async def status_path():
    return "Hello world, I am online😀"


@app.get("/")
async def streaming_path():
    def iterfile():
        while True:
            if type(visualFrameQueue[0]) == np.ndarray:
                yield b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + visualFrameQueue[
                    0
                ].tobytes() + b"\r\n"
                sleep(0.016)

    return StreamingResponse(
        iterfile(),
        media_type="multipart/x-mixed-replace;boundary=frame",
    )


if __name__ == "__main__":
    uvicorn.run(
        app="app:app",
        app_dir="src/",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
