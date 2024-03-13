import os.path

import requests

from dds_cloudapi_sdk.config import Config
from dds_cloudapi_sdk.tasks.base import BaseTask

__all__ = [
    "Client"
]


class Client:
    def __init__(self, token: str):
        """
        Initialize a client which can be used to call dds cloud APIs.
        :param token: The API token.
        """

        self.config = Config(token)

    def upload_file(self, local_path: str) -> str:
        """
        upload local image to dds server, return a visible url for calling dds algorithm API.
        :param local_path: The local file path.
        """

        # the token given before

        # request our server API to upload the image file
        sign_url = f"https://{self.config.endpoint.value}/upload_signature"
        headers = {"Token": self.config.token}

        file_name = os.path.basename(local_path)
        data = {"file_name": file_name}
        rsp = requests.post(sign_url, json=data, headers=headers, timeout=2)
        assert rsp.status_code == 200
        rsp_json = rsp.json()

        # parse the urls from API result
        upload_url = rsp_json["data"]["upload_url"]
        download_url = rsp_json["data"]["download_url"]

        # upload the image to our server by the upload_url
        with open(local_path, "rb") as fp:
            rsp = requests.put(upload_url, fp.read())
            assert rsp.status_code == 200

        return download_url

    def trigger_task(self, task: BaseTask):
        """
        Trigger a task.
        """
        return task.trigger(self.config)

    def check_task(self, task: BaseTask):
        """
        Check the task status.
        """
        return task.check()

    def wait_task(self, task: BaseTask):
        """
        Wait for the task to complete.
        This blocks the current thread until the task is done.
        """
        return task.wait()

    def run_task(self, task: BaseTask):
        """
        Trigger a task and wait for it to complete.
        """
        return task.run(self.config)
