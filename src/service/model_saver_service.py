import torch
from pathlib import Path
from os import remove

from service.logger_service import LoggerService


class ModelSaverService:
    def __init__(self, path: Path) -> None:
        self.model_directory = path
        self.latest_model = []
        self.logger = LoggerService()

        if not self.model_directory.exists():
            self.model_directory.mkdir()

    def _generate_save_name(self, epoch: int):
        return f"{epoch}_model.pt"

    def save(self, model: torch.nn.Module, epoch: int):
        if len(self.latest_model) > 2:
            first_epoch_to_delete = self.latest_model.pop(0)
            model_to_delete = self.model_directory.joinpath(
                self._generate_save_name(first_epoch_to_delete)
            )
            remove(model_to_delete)
            self.logger().warning(
                f"{self._generate_save_name(first_epoch_to_delete)} removed!"
            )

        torch.save(
            model.state_dict(),
            self.model_directory.joinpath(self._generate_save_name(epoch)),
        )
        self.latest_model.append(epoch)
