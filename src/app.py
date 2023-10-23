import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Normalize

from dataloader.dataloader import UAVIDDataset
from model.model import UNETNetwork
from service.logger_service import LoggerService
from trainer.trainer import Trainer


class App:
    def __init__(self) -> None:
        self.logger = LoggerService()
        self.logger().warning("Initialization of application")
        self.trainer = Trainer(train_report_rate=5)

    def run_train(self, device):
        self.logger().warning(f"Run on {device}")
        training_data = UAVIDDataset(path="data/processed_dataset/", is_train=True)
        train_dataloader = DataLoader(training_data, batch_size=4, shuffle=True)
        model = UNETNetwork(numberClass=8)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

        def dice_loss(pred: torch.Tensor, target: torch.Tensor):
            pred_flat = pred.flatten()
            target_flat = target.flatten()
            nominator = 2 * torch.mul(pred_flat, target_flat)
            denominator = torch.add(pred_flat, target_flat)
            return 1 - torch.mean((nominator + 1e-9) / (denominator + 1e-9))

        def total_loss(pred: torch.Tensor, target: torch.Tensor):
            return torch.nn.functional.cross_entropy(pred, target) + dice_loss(
                pred.softmax(1), target
            )

        preprocess = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        self.trainer.train(
            epochs=5,
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            loss_fn=total_loss,
            preprocess=preprocess,
            device=device,
        )
