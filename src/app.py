from threading import Thread
from typing import Union

import uvicorn
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

from inferencing.inference import Inferencer, Statistics
from service.logger_service import LoggerService
from trainer.trainer import Trainer

statistics = Statistics()

class App:
    def __init__(self, device) -> None:
        self.logger = LoggerService()
        self.logger().warning("Initialization of application")
        # self.trainer = Trainer(train_report_rate=5)
        self.inferencer = Inferencer(device=device)

    def run(self) -> None:
        # Spawn thread of CCTV monitoring and tracking
        # inference_thread: Thread = Thread(target=self.inferencer.infer, args=('/Users/babi/Downloads/video.mp4', statistics))
        # inference_thread.start()

        # Start webesrver
        self.webserver()

    @staticmethod
    def webserver_factory() -> FastAPI:
        app = FastAPI()

        @app.get('/')
        async def root():
            return 'hello world'

        @app.get('/metrics', response_class=PlainTextResponse)
        async def metrics():
            return '\n'.join(['# HELP number_of_person', f'number_of_person {statistics.number_of_person}'])
        
        return app


    @staticmethod
    def webserver() -> None:
        uvicorn.run(app='app:App.webserver_factory', factory=True, reload=True, port=9090, host='0.0.0.0')


    # def run_train(self, device):
    #     self.logger().warning(f"Run on {device}")
    #     self.trainer.run_trainer(device=device)

