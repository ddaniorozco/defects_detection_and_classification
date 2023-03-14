import os
from cloud_provider import CloudProvider
from remote_runner import AsyncRemoteRunner


class TrainingRemoteRunner(AsyncRemoteRunner):

    def __init__(self, cloud_provider=None):
        super(TrainingRemoteRunner, self).__init__()

        self.cloud_provider = CloudProvider(cloud_provider).get_provider()

        self.aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        self.aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        self.region_name = 'eu-west-1'      # The region of hawkeye-tfrecord s3 bucket

        self.remote_pipeline_config_path = os.getenv('REMOTE_PIPELINE_CONFIG_PATH')
        self.model_dir = os.getenv('MODEL_DIR')

        self.training_scripts_local_relative_path = ''

    def set_training_scripts_local_relative_path(self, path):
        self.training_scripts_local_relative_path = path
        self.pem_file_name = path + self.pem_file_name

    async def run(self, connection):

        print('Creating directory structure on remote host...')
        commands = ['rm -rf workspace', 'mkdir workspace', 'cd workspace', 'mkdir automation']
        await AsyncRemoteRunner.run_commands(connection, commands)

        print('Copying training scripts to the remote host...')
        path = self.training_scripts_local_relative_path
        files_to_copy = [(path + 's3_sync.sh', '~/workspace/automation'),
                         (path + 'object_detection_training_configurator.py', '~/workspace/automation')]

        if self.cloud_provider == CloudProvider.AWS:
            files_to_copy += [(path + 'train_ec2.sh', '~/workspace/automation')]
        elif self.cloud_provider == CloudProvider.AZURE:
            files_to_copy += [(path + 'train_docker.sh', '~/workspace/automation'),
                              (path + 'train.sh', '~/workspace/automation')]

        await AsyncRemoteRunner.copy_files(connection, files_to_copy)

        env_variables_dict = {
            'AWS_ACCESS_KEY_ID': self.aws_access_key_id,
            'AWS_SECRET_ACCESS_KEY': self.aws_secret_access_key,
            'AWS_REGION': self.region_name,
            'REMOTE_PIPELINE_CONFIG_PATH': self.remote_pipeline_config_path,
            'MODEL_DIR': self.model_dir,
            'BUILD_ID': os.getenv('BUILD_ID', default='0'),
            'RUN_DISPLAY_URL': os.getenv('RUN_DISPLAY_URL', default=''),
        }

        if self.cloud_provider == CloudProvider.AZURE:
            azure_docker_repository_password = os.getenv('AZURE_DOCKER_TRAINING_REPOSITORY_PASSWORD', default='')
            env_variables_dict.update({
                'AZURE_DOCKER_TRAINING_REPOSITORY_PASSWORD': azure_docker_repository_password
            })

        export_commands = ['export %s="%s"' % (key, value) for key, value in env_variables_dict.items()] + ['env']

        print('Starting training on the remote host...')

        if self.cloud_provider == CloudProvider.AWS:
            training_commands = export_commands + [
                'chmod +x ~/workspace/automation/train_ec2.sh',
                '~/workspace/automation/train_ec2.sh'
            ]
        elif self.cloud_provider == CloudProvider.AZURE:
            training_commands = export_commands + [
                'chmod +x ~/workspace/automation/train_docker.sh',
                'echo "Tensorboard: http://' + self.host + ':6006/"',
                '~/workspace/automation/train_docker.sh'
            ]

        await AsyncRemoteRunner.run_commands(connection, training_commands)

        print('Remote training script finished')
