"""
Initializing the configuration is the first step users have to do before calling any DDS CloudAPI services.

This is done by initializing a :class:`Config <dds_cloudapi_sdk.config.Config Class>` class with the API token::

    from dds_cloudapi_sdk import Config

    token = "You API token here"
    config = Config(token)

"""

import enum
import os


class ServerEnv(enum.Enum):
    Dev = "dev"
    Test = "test"
    Prd = "prd"


class ServerEndpoint(enum.Enum):
    Dev = "apidev.deepdataspace.com"
    Test = "apitest.deepdataspace.com"
    Prd = "api.deepdataspace.com"


def _choose_endpoint():
    endpoint = os.environ.get("DDS_CLOUDAPI_ENDPOINT", ServerEndpoint.Prd.value)
    return endpoint


class Config:
    """
    The configuration representation for the SDK client.

    :param token: The API token of your DDS account. Currently, you can apply for an API token by sending email to weiliu@idea.edu.cn.

    """

    def __init__(self, token: str):
        """
        Initialize a configuration with API token.
        """

        self.endpoint: ServerEndpoint = _choose_endpoint()
        self.token: str = token
