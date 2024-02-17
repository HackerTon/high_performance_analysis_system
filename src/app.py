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

logger = LoggerService()
streaming_down_connection, streaming_up_connection = Pipe(False)
parentConnection, childConnection = Pipe()


device = os.getenv("DEVICE", "cpu")
video_path = os.getenv("VIDEO_PATH")

if video_path == None:
    logger.logger.warning("video_path ENV not provided")
    quit()

logger().warning("Initialization of application")


@asynccontextmanager
async def deepengine(app: FastAPI):
    # collector = MockUpCollector(image_path=video_path)
    collector = LastFrameCollector(video_path=video_path)
    inferencer = OCRInferencer(device=device)
    inferencer.run(
        device=device,
        frame_down_connection=parentConnection,
        streaming_up_connection=streaming_up_connection,
    )
    collector.start(upstream_connection=childConnection)
    inferencer.thread.start()
    collector.process.start()
    yield
    collector.running = False
    inferencer.running = False
    inferencer.thread.join()
    collector.process.terminate()
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
            image_bytes = streaming_down_connection.recv_bytes()
            if image_bytes is None:
                continue
            yield b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + image_bytes + b"\r\n"
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
