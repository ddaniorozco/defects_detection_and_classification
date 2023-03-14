import os
from cloud_provider import CloudProvider
from remote_runner import AsyncRemoteRunner


class TrainingPreprocessorRemoteRunner(AsyncRemoteRunner):

    def __init__(self, cloud_provider=None):
        super(TrainingPreprocessorRemoteRunner, self).__init__()

        self.cloud_provider = CloudProvider(cloud_provider).get_provider()

        self.aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        self.aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        self.region_name = 'eu-west-1'

        self.source_dir = os.getenv('SOURCE_DIR')
        self.train_dir = os.getenv('TRAIN_DIR')
        self.test_dir = os.getenv('TEST_DIR')
        self.val_dir = os.getenv('VAL_DIR')
        self.test_percentage = os.getenv('TEST_PERCENTAGE')
        self.val_percentage = os.getenv('VAL_PERCENTAGE')

        self.training_scripts_local_relative_path = ''

    def set_training_scripts_local_relative_path(self, path):
        self.training_scripts_local_relative_path = path
        self.pem_file_name = path + self.pem_file_name

    async def run(self, connection):
        print('Creating directory structure on remote host...')
        commands = ['rm -rf workspace', 'mkdir workspace', 'cd workspace', 'mkdir automation']
        await AsyncRemoteRunner.run_commands(connection, commands)

        print('Copying training_preprocess scripts to the remote host...')
        path = self.training_scripts_local_relative_path
        files_to_copy = [(path + 'docker_training_preprocessor.sh', '~/workspace/automation'),
                         (path + 'training_preprocessor.sh', '~/workspace/automation'),
                         (path + '../tools/labels_data/labelme2coco.py', '~/workspace/automation'),
                         (path + 'src/pre_processing/training_preprocessor.py', '~/workspace/automation')]

        print(files_to_copy)

        await AsyncRemoteRunner.copy_files(connection, files_to_copy)

        env_variables_dict = {

            'AWS_ACCESS_KEY_ID': self.aws_access_key_id,
            'AWS_SECRET_ACCESS_KEY': self.aws_secret_access_key,
            'AWS_REGION': self.region_name,
            'SOURCE_DIR': self.source_dir,
            'TRAIN_DIR': self.train_dir,
            'TEST_DIR': self.test_dir,
            'VAL_DIR': self.val_dir,
            'TEST_PERCENTAGE': self.test_percentage,
            'VAL_PERCENTAGE': self.val_percentage,
            'BUILD_ID': os.getenv('BUILD_ID', default='0'),
            'RUN_DISPLAY_URL': os.getenv('RUN_DISPLAY_URL', default='')
        }

        if self.cloud_provider == CloudProvider.AZURE:
            azure_docker_repository_password = os.getenv('AZURE_DOCKER_TRAINING_REPOSITORY_PASSWORD', default='')
            env_variables_dict.update({
                'AZURE_DOCKER_TRAINING_REPOSITORY_PASSWORD': azure_docker_repository_password
            })

        export_commands = ['export %s="%s"' % (key, value) for key, value in env_variables_dict.items()] + ['env']

        print('Starting training_preprocessor on the remote host...')

        training_preprocessor_exporter_commands = export_commands + [
            'chmod +x ~/workspace/automation/docker_training_preprocessor.sh',
            '~/workspace/automation/docker_training_preprocessor.sh']

        await AsyncRemoteRunner.run_commands(connection, training_preprocessor_exporter_commands)

        print('Remote training_preprocessor script finished')
