import os
from cloud_provider import CloudProvider
from remote_runner import AsyncRemoteRunner


class OdResultsExporterRunner(AsyncRemoteRunner):

    def __init__(self, cloud_provider=None):
        super(OdResultsExporterRunner, self).__init__()

        self.cloud_provider = CloudProvider(cloud_provider).get_provider()

        self.aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        self.aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        self.region_name = 'eu-west-1'

        self.frozen_graph_path = os.getenv('FROZEN_GRAPH_PATH')
        self.labels_path = os.getenv('LABELS_PATH')
        # self.num_classes = os.getenv('NUM_CLASSES')
        self.min_thresh_percent = os.getenv('MIN_THRESH_PERCENT')
        self.images_path = os.getenv('IMAGES_PATH')
        self.results_json = os.getenv('RESULTS_JSON')
        self.results_images = os.getenv('RESULTS_IMAGES')

        self.training_scripts_local_relative_path = ''

    def set_training_scripts_local_relative_path(self, path):
        self.training_scripts_local_relative_path = path
        self.pem_file_name = path + self.pem_file_name

    async def run(self, connection):
        print('Creating directory structure on remote host...')
        commands = ['rm -rf workspace', 'mkdir workspace', 'cd workspace', 'mkdir automation']
        await AsyncRemoteRunner.run_commands(connection, commands)

        print('Copying od_results_exporter scripts to the remote host...')
        path = self.training_scripts_local_relative_path
        files_to_copy = [(path + 'docker_od_results_exporter.sh', '~/workspace/automation'),
                         (path + 'od_results_exporter.sh', '~/workspace/automation'),
                         (path + 'src/report/od_results_exporter.py', '~/workspace/automation'),
                         (path + 'src/report/tf_object_detector.py', '~/workspace/automation')]

        print(files_to_copy)

        await AsyncRemoteRunner.copy_files(connection, files_to_copy)

        env_variables_dict = {

            'AWS_ACCESS_KEY_ID': self.aws_access_key_id,
            'AWS_SECRET_ACCESS_KEY': self.aws_secret_access_key,
            'AWS_REGION': self.region_name,
            'FROZEN_GRAPH_PATH': self.frozen_graph_path,
            'LABELS_PATH': self.labels_path,
            # 'NUM_CLASSES': self.num_classes,
            'MIN_THRESH_PERCENT': self.min_thresh_percent,
            'IMAGES_PATH': self.images_path,
            'RESULTS_JSON': self.results_json,
            'RESULTS_IMAGES': self.results_images,
            'BUILD_ID': os.getenv('BUILD_ID', default='0'),
            'RUN_DISPLAY_URL': os.getenv('RUN_DISPLAY_URL', default='')
        }

        if self.cloud_provider == CloudProvider.AZURE:
            azure_docker_repository_password = os.getenv('AZURE_DOCKER_TRAINING_REPOSITORY_PASSWORD', default='')
            env_variables_dict.update({
                'AZURE_DOCKER_TRAINING_REPOSITORY_PASSWORD': azure_docker_repository_password
            })

        export_commands = ['export %s="%s"' % (key, value) for key, value in env_variables_dict.items()] + ['env']

        print('Starting od_results_exporter on the remote host...')

        od_results_exporter_commands = export_commands + [
            'chmod +x ~/workspace/automation/docker_od_results_exporter.sh',
            '~/workspace/automation/docker_od_results_exporter.sh']

        await AsyncRemoteRunner.run_commands(connection, od_results_exporter_commands)

        print('Remote od_results_exporter script finished')
