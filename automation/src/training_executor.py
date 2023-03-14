import logging
import os
from training_remote_runner import TrainingRemoteRunner
from task_executor import TaskExecutor

logger = logging.getLogger()


def configure_logger():
    logger.root.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


class TrainingExecutor(TaskExecutor):
    def __init__(self):
        super(TrainingExecutor, self).__init__(remote_runner=TrainingRemoteRunner())


if __name__ == '__main__':
    configure_logger()
    host = os.getenv('HOST_IP', default=None)
    TrainingExecutor().run(host=host)

