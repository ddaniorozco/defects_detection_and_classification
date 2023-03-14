import os


class CloudProvider:
    AWS = 'aws'
    AZURE = 'azure'

    def __init__(self, cloud_provider=None):
        if cloud_provider is None:
            cloud_provider = os.getenv('CLOUD_PROVIDER', default=CloudProvider.AWS)

        # The default cloud provider is aws
        self._cloud_provider = CloudProvider.AZURE if cloud_provider == CloudProvider.AZURE else CloudProvider.AWS

    def get_provider(self):
        return self._cloud_provider
