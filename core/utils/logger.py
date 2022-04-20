import os
from typing import Tuple

from overboard import Logger as OBLogger
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, save_dir, config):
        self.save_dir = save_dir
        self.config = config
        self._logger = self._init_logger()

    def _init_logger(self):
        raise NotImplementedError

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if self._logger:
            try:
                self._logger.close()
            except:
                pass

    def append(self, data: dict, step):
        raise NotImplementedError

    def add_scalar(self, key, value, step):
        raise NotImplementedError


class TensorBoardLogger(Logger):
    def _init_logger(self):
        return SummaryWriter(os.path.join(self.save_dir, "tensorboard"))

    def append(self, data: dict, step):
        for k, v in data.items():
            self.add_scalar(k, v, step)

    def add_scalar(self, key, value, step):
        self._logger.add_scalar(key, value, step)


class OverboardLogger(Logger):
    def _init_logger(self):
        return OBLogger(self.save_dir, meta=self.config)

    def append(self, data: dict, step):
        self._logger.append(data)


class MetaLogger(Logger):
    def __init__(self, save_dir, config, loggers: Tuple[str] = ("overboard", "tensorboard")):
        super(MetaLogger, self).__init__(save_dir, config)
        mapping = {
            'overboard': OverboardLogger,
            'tensorboard': TensorBoardLogger
        }
        self._loggers = {name: mapping[name](save_dir, config) for name in loggers}
        self.tb: SummaryWriter = self._loggers['tensorboard']._logger

    def _init_logger(self):
        pass

    def close(self):
        for logger in self._loggers.values(): logger.close()

    def _method(self, method_name, *args, **kwargs):
        try:
            for logger in self._loggers.values(): getattr(logger, method_name)(*args, **kwargs)
        except NotImplementedError:
            pass

    def append(self, data: dict, step):
        self._method("append", data, step)

    def add_scalar(self, key, value, step):
        self._method("add_scalar", key, value, step)
