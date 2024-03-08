from dds_cloudapi_sdk.config import Config
from dds_cloudapi_sdk.tasks.base import BaseTask

__all__ = [
    "Client"
]


class Client:
    def __init__(self, token: str):
        self.config = Config(token)

    def upload_file(self, local_path: str) -> str:
        pass

    def trigger_task(self, task: BaseTask):
        return task.trigger(self.config)

    def check_task(self, task: BaseTask):
        return task.check()

    def wait_task(self, task: BaseTask):
        return task.wait()

    def run_task(self, task: BaseTask):
        return task.run(self.config)
