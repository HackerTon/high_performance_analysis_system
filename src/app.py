from dataloader.dataloader import UAVIDDataset
from torch.utils.data import DataLoader, Dataset
import torch
from torchmetrics import Dice
from model.model import UNETNetwork
from service.logger_service import LoggerService
from trainer.trainer import Trainer

class LimitDataset(Dataset):
    def __init__(self, dataset: Dataset, n):
        self.dataset = dataset
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        return self.dataset[index]


class App:
    def __init__(self) -> None:
        self.logger = LoggerService()
        self.logger().warning("Initialization of application")
        self.trainer = Trainer(train_report_rate=5)

    def run_train(self, device):
        self.logger().warning(f'Run on {device}')
        training_data = UAVIDDataset(path="data/processed_dataset/", is_train=True)
        train_dataloader = DataLoader(LimitDataset(training_data, 1500), batch_size=4, shuffle=True)

        loss_fn1 = torch.nn.CrossEntropyLoss().to(device)
        loss_fn2 = Dice().to(device)
        model = UNETNetwork(numberClass=8)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)

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
            device=device
        )
