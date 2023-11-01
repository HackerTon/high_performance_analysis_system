from logging import (
    ERROR,
    FileHandler,
    Formatter,
    getLogger,
    Logger,
    INFO,
    StreamHandler,
    WARNING,
)
from pathlib import Path
from typing import Any


class LoggerService:
    _instance = None

    def _init(self) -> None:
        self.logger: Logger = getLogger(__name__)

        # Log directory
        self.log_path: Path = Path("log/")
        if not self.log_path.exists():
            self.log_path.mkdir()

        file_handler: FileHandler = FileHandler("log/log.txt")
        stream_handler: StreamHandler = StreamHandler()
        file_handler.setLevel(WARNING)
        stream_handler.setLevel(WARNING)
        file_formatter: Formatter = Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(fmt=file_formatter)
        stream_handler.setFormatter(fmt=file_formatter)
        self.logger.addHandler(hdlr=file_handler)
        self.logger.addHandler(hdlr=stream_handler)

    def __call__(self) -> Logger:
        return self.logger

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance
