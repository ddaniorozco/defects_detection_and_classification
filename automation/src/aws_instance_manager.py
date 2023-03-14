import os
import boto3
from botocore.exceptions import ClientError

from instance_manager import InstanceManager


class AwsInstanceManager(InstanceManager):

    def __init__(self):
        super().__init__()

        self.aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        self.aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        self.region_name = os.getenv('AWS_REGION')

        self.ec2_resource = None
        self.ec2_instance = None
        self.instance_id = None
        self.ip_address = None


    def launch_instance(self):
        try:
            self.ec2_resource = boto3.resource('ec2',
                                               region_name=self.region_name,
                                               aws_access_key_id=self.aws_access_key_id,
                                               aws_secret_access_key=self.aws_secret_access_key
                                               )

            launch_instance_params_dict_list = self.get_launch_instance_params_dict_list()
            print('Launching instance...')

            response = None
            for launch_instance_params_dict in launch_instance_params_dict_list:
                try:
                    response = self.ec2_resource.create_instances(**launch_instance_params_dict)
                    break
                except ClientError as err:
                    # Checking if the instance couldn't be launched in the current availability zone and trying the next one
                    if err.response['Error']['Code'] in ['Unsupported', 'InsufficientInstanceCapacity'] \
                            and 'Availability Zone' in err.response['Error']['Message']:
                        print(err)
                    else:
                        raise

            if response is None:
                raise Exception('Instance could not be launched')

            self.ec2_instance = response[0]
            self.instance_id = self.ec2_instance.id
            self.ip_address = self.get_ip_address()
            self.wait_for_instance()

        except ClientError as e:
            print('Error:', e)
            raise

    def get_ip_address(self):
        return self.ec2_instance.private_ip_address
        # return self.ec2_instance.public_ip_address

    def get_launch_instance_params_dict_list(self):
        # Virtual method
        return {}

    def wait_for_instance(self):
        ip_str = ''
        if self.get_ip_address() is not None:
            ip_str = ' (ip: ' + self.get_ip_address() + ')'

        print('Waiting for instance to be ready: ' + self.instance_id + ip_str)
        self.ec2_instance.wait_until_running()

        InstanceManager.wait_for_instance(self)

    def terminate_instance(self):
        try:
            self.ec2_instance.terminate()
            print('Instance terminated')
        except ClientError as e:
            print('Error:', e)
            raise

    def update_keep_alive_tag_for_instance(self, minutes):

        if self.ec2_instance.state['Name'] != 'terminated':
            print('Updating keep alive tag for the training instance')
            self.ec2_instance.create_tags(Tags=[self.get_keep_alive_tag_dict(minutes)])
        else:
            print('Instance is terminated')
