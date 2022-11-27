import sys
import logging
import torch.distributed as dist

class Logger():
    def __init__(self, logging_file, name, isopen):
        self.isopen = isopen

        if isopen is True:
            # 创建一个日志器logger并设置其日志级别为DEBUG
            logger = logging.getLogger(name)
            logger.handlers.clear()
            logger.setLevel(logging.DEBUG)
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

            # 创建一个流处理器handler并设置其日志级别为DEBUG，输出到命令窗口
            handler1 = logging.StreamHandler(sys.stdout)
            handler1.setLevel(logging.DEBUG)
            handler1.setFormatter(formatter)

            # 创建一个文件处理器handler并设置其日志级别为DEBUG，输出到文件
            handler2 = logging.FileHandler(logging_file)
            handler2.setLevel(logging.DEBUG)
            handler2.setFormatter(formatter)

            # 为日志器logger添加上面创建的处理器handler
            logger.addHandler(handler1)
            logger.addHandler(handler2)

            self.logger = logger

    def info(self, message):
        if dist.is_initialized() is False or dist.get_rank() == 0:
            if self.isopen:
                self.logger.info(message)

    def warning(self, message):
        if dist.is_initialized() is False or dist.get_rank() == 0:
            if self.isopen:
                self.logger.warning(message)

def get_logger(logging_file=None, name=None, isopen=False):
    if dist.is_initialized() is False:
        if not hasattr(get_logger, 'Logger_0'):
            assert logging_file is not None, "Logger hasn't been created."
            get_logger.Logger_0 = Logger(logging_file, name, isopen=isopen)
        return get_logger.Logger_0
    elif dist.get_rank() == 0:
        if not hasattr(get_logger, 'Logger_0'):
            assert logging_file is not None, "Logger_0 hasn't been created."
            get_logger.Logger_0 = Logger(logging_file, name, isopen=isopen)
        return get_logger.Logger_0
    else:
        if not hasattr(get_logger, 'Logger_1'):
            get_logger.Logger_1 = Logger(None, None, isopen=False)
        return get_logger.Logger_1