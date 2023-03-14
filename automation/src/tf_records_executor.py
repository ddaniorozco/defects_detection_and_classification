import os
from tf_records_remote_runner import TfRecordsRemoteRunner
from task_executor import TaskExecutor


class TfRecordsExecutor(TaskExecutor):
    def __init__(self):
        super(TfRecordsExecutor, self).__init__(remote_runner=TfRecordsRemoteRunner())


if __name__ == '__main__':
    host = os.getenv('HOST_IP', default=None)
    TfRecordsExecutor().run(host=host)
