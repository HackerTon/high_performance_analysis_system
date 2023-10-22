from logging import ERROR, FileHandler, Formatter, getLogger, Logger, INFO, StreamHandler, WARNING
from pathlib import Path
from dataloader.dataloader import UAVIDDataset
from torch.utils.data import DataLoader
import torch
from torchmetrics import Dice
from model.model import UNETNetwork
from trainer.trainer import Trainer

class App:
    def __init__(self) -> None:
        self._initializeLogging()
        self.logger.warning('Initialization of application')
        self.trainer = Trainer(train_report_rate=5)

    def _initializeLogging(self) -> None:
        self.logger: Logger = getLogger(__name__)

        # Log directory
        self.log_path: Path = Path('log/')
        if not self.log_path.exists():
            self.log_path.mkdir()

        file_handler: FileHandler  = FileHandler('log/log.txt')
        stream_handler: StreamHandler = StreamHandler()
        file_handler.setLevel(WARNING)
        stream_handler.setLevel(WARNING)
        file_formatter: Formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(fmt=file_formatter)
        stream_handler.setFormatter(fmt=file_formatter)
        self.logger.addHandler(hdlr=file_handler)
        self.logger.addHandler(hdlr=stream_handler)

    def run_train(self):
        training_data = UAVIDDataset(path="data/processed_dataset/", is_train=True)
        train_dataloader = DataLoader(training_data, batch_size=4, shuffle=True)

        loss_fn1 = torch.nn.CrossEntropyLoss().to('mps')
        loss_fn2 = Dice().to('mps')
        model = UNETNetwork(numberClass=8)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

        def loss_fn(outputs, labels):
            return loss_fn1(outputs, labels) + (1 - loss_fn2(outputs, labels.to(torch.int32)))

        self.trainer.train(epochs=1, model=model, dataloader=train_dataloader, optimizer=optimizer, loss_fn=loss_fn)
