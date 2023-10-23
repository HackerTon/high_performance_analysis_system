from dataloader.dataloader import UAVIDDataset
from torch.utils.data import DataLoader
import torch
from torchmetrics import Dice
from model.model import UNETNetwork
from service.logger_service import LoggerService
from trainer.trainer import Trainer


class App:
    def __init__(self) -> None:
        self.logger = LoggerService()
        self.logger().warning("Initialization of application")
        self.trainer = Trainer(train_report_rate=5)

    def run_train(self):
        training_data = UAVIDDataset(path="data/processed_dataset/", is_train=True)
        train_dataloader = DataLoader(training_data, batch_size=4, shuffle=True)

        loss_fn1 = torch.nn.CrossEntropyLoss().to("mps")
        loss_fn2 = Dice().to("mps")
        model = UNETNetwork(numberClass=8)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

        def loss_fn(outputs, labels):
            return loss_fn1(outputs, labels) + (
                1 - loss_fn2(outputs, labels.to(torch.int32))
            )

        self.trainer.train(
            epochs=5,
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            loss_fn=loss_fn,
        )
