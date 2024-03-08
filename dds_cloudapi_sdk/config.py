import enum


class ServerEnv(enum.Enum):
    Dev = "dev"
    Test = "test"
    Prd = "prd"


class ServerEndpoint(enum.Enum):
    Dev = "apidev.deepdataspace.com"
    Test = "apitest.deepdataspace.com"
    Prd = "api.deepdataspace.com"


def _choose_endpoint(env: ServerEnv):
    map_ = {
        ServerEnv.Dev : ServerEndpoint.Dev,
        ServerEnv.Test: ServerEndpoint.Test,
        ServerEnv.Prd : ServerEndpoint.Prd
    }

    return map_[env]


class Config:
    def __init__(self, token: str, env: ServerEnv = ServerEnv.Dev):
        self.endpoint: ServerEndpoint = _choose_endpoint(env)
        self.token: str = token
