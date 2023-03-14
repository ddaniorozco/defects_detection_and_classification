import os
from od_results_exporter_runner import OdResultsExporterRunner
from task_executor import TaskExecutor


class OdResultsExporterExecutor(TaskExecutor):
    def __init__(self):
        super(OdResultsExporterExecutor, self).__init__(remote_runner=OdResultsExporterRunner())


if __name__ == '__main__':
    host = os.getenv('HOST_IP', default=None)
    OdResultsExporterExecutor().run(host=host)
