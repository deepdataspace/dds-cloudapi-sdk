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
    env = os.environ.get("DDS_CLOUDAPI_ENV", ServerEnv.Prd)
    map_ = {
        ServerEnv.Dev : ServerEndpoint.Dev,
        ServerEnv.Test: ServerEndpoint.Test,
        ServerEnv.Prd : ServerEndpoint.Prd
    }
    return map_[env]


class Config:
    def __init__(self, token: str):
        self.endpoint: ServerEndpoint = _choose_endpoint()
        self.token: str = token
