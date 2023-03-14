import os
from training_preprocessor_remote_runner import TrainingPreprocessorRemoteRunner
from task_executor import TaskExecutor


class TrainingPreprocessorExecutor(TaskExecutor):
    def __init__(self):
        super(TrainingPreprocessorExecutor, self).__init__(remote_runner=TrainingPreprocessorRemoteRunner())


if __name__ == '__main__':
    host = os.getenv('HOST_IP', default=None)
    TrainingPreprocessorExecutor().run(host=host)
