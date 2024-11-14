"""
Client is the main entry point for users to interact with the DDS Cloud API.
After initializing it with a  :class:`Config <dds_cloudapi_sdk.config.Config>` class, users can use it to upload files,
trigger tasks, and check for the results.

A simple example illustrating the major interface::

    from dds_cloudapi_sdk import Config
    from dds_cloudapi_sdk import Client

    token = "Your API Token Here"
    config = Config(token)
    client = Client(config)

    # upload local file with client
    url = client.upload_file("/path/to/local_file.jpg")

    # run a task with client
    client.run(task)
    print(task.result)

"""

import os.path

import requests

from dds_cloudapi_sdk.config import Config
from dds_cloudapi_sdk.tasks.base import BaseTask

__all__ = [
    "Client"
]


class Client:
    """
    | This is the SDK client for dds cloud APIs.
    | It is initialized with the API token, and talks to the server to:

    - 1. upload files to get the visible url
    - 2. run tasks and wait for the results

    :param config: The :class:`Config <dds_cloudapi_sdk.config.Config>` object.

    """

    def __init__(self, config: Config):
        self.config = config

    def upload_file(self, local_path: str, timeout: int=10) -> str:
        """
        | Upload local file to dds server, return a visible url ready for calling dds cloud API.

        | Although users can trigger tasks with any publicly visible url, uploading file to dds server and use the dds
         hosted url as task parameter is necessary to conform the network security policy of the DDS server.

        :param local_path: The local file path.
        """

        # request our server API to upload the image file
        sign_url = f"https://{self.config.endpoint}/upload_signature"
        headers = {"Token": self.config.token}

        file_name = os.path.basename(local_path)
        data = {"file_name": file_name}
        rsp = requests.post(sign_url, json=data, headers=headers, timeout=timeout)
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
        Trigger a task and return immediately without waiting for the result.

        :param task: The task to trigger.
        """
        return task.trigger(self.config)

    def check_task(self, task: BaseTask):
        """
        Check the task's :class:`status <dds_cloudapi_sdk.tasks.base.TaskStatus>`.

        :param task: The task to check.
        """
        return task.check()

    def wait_task(self, task: BaseTask):
        """
        | Wait for the task to complete.
        | This blocks the current thread until the task is done.

        :param task: The task to wait.
        """
        return task.wait()

    def run_task(self, task: BaseTask):
        """
        | Trigger a task and wait for it to complete.
        | This blocks the current thread until the task is done.

        :param task: The task to run.
        """
        return task.run(self.config)
