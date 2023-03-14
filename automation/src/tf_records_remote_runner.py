import os
from cloud_provider import CloudProvider
from remote_runner import AsyncRemoteRunner


class TfRecordsRemoteRunner(AsyncRemoteRunner):

    def __init__(self, cloud_provider=None):
        super(TfRecordsRemoteRunner, self).__init__()

        self.cloud_provider = CloudProvider(cloud_provider).get_provider()

        self.aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        self.aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        self.region_name = 'eu-west-1'

        self.train_image_dir = os.getenv('TRAIN_IMAGE_DIR')
        self.val_image_dir = os.getenv('VAL_IMAGE_DIR')
        self.test_image_dir = os.getenv('TEST_IMAGE_DIR')
        self.train_annotations_file = os.getenv('TRAIN_ANNOTATIONS_FILE')
        self.val_annotations_file = os.getenv('VAL_ANNOTATIONS_FILE')
        self.testdev_annotations_file = os.getenv('TESTDEV_ANNOTATIONS_FILE')
        self.output_dir = os.getenv('OUTPUT_DIR')

        self.training_scripts_local_relative_path = ''

    def set_training_scripts_local_relative_path(self, path):
        self.training_scripts_local_relative_path = path
        self.pem_file_name = path + self.pem_file_name

    async def run(self, connection):

        print('Creating directory structure on remote host...')
        commands = ['rm -rf workspace', 'mkdir workspace', 'cd workspace', 'mkdir automation']
        await AsyncRemoteRunner.run_commands(connection, commands)

        print('Copying tf_records scripts to the remote host...')
        path = self.training_scripts_local_relative_path
        files_to_copy = [(path + 'docker_tf_records.sh', '~/workspace/automation'),
                         (path + 'create_tf_records.sh', '~/workspace/automation'),
                         (path + 'src/od_files_to_update/create_coco_tf_records_fix.py', '~/workspace/automation')]

        print(files_to_copy)

        await AsyncRemoteRunner.copy_files(connection, files_to_copy)

        env_variables_dict = {
            'AWS_ACCESS_KEY_ID': self.aws_access_key_id,
            'AWS_SECRET_ACCESS_KEY': self.aws_secret_access_key,
            'AWS_REGION': self.region_name,
            'TRAIN_IMAGE_DIR': self.train_image_dir,
            'VAL_IMAGE_DIR': self.val_image_dir,
            'TEST_IMAGE_DIR': self.val_image_dir,
            'TRAIN_ANNOTATIONS_FILE': self.train_annotations_file,
            'VAL_ANNOTATIONS_FILE': self.val_annotations_file,
            'TESTDEV_ANNOTATIONS_FILE': self.testdev_annotations_file,
            'OUTPUT_DIR': self.output_dir,
            'BUILD_ID': os.getenv('BUILD_ID', default='0'),
            'RUN_DISPLAY_URL': os.getenv('RUN_DISPLAY_URL', default='')
        }

        if self.cloud_provider == CloudProvider.AZURE:
            azure_docker_repository_password = os.getenv('AZURE_DOCKER_TRAINING_REPOSITORY_PASSWORD', default='')
            env_variables_dict.update({
                'AZURE_DOCKER_TRAINING_REPOSITORY_PASSWORD': azure_docker_repository_password
            })

        export_commands = ['export %s="%s"' % (key, value) for key, value in env_variables_dict.items()] + ['env']

        print('Starting tf_records on the remote host...')

        tf_records_commands = export_commands + [
            'chmod +x ~/workspace/automation/docker_tf_records.sh',
            '~/workspace/automation/docker_tf_records.sh']

        await AsyncRemoteRunner.run_commands(connection, tf_records_commands)

        print('Remote tf_records script finished')
