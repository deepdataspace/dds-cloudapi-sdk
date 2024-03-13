import abc
import enum
import logging
import time
from typing import Union

import pydantic
import requests

from dds_cloudapi_sdk.config import Config

logger = logging.getLogger("dds_cloudapi_sdk")


class TaskStatus(enum.Enum):
    Triggering = "triggering"  # send request to
    Waiting = "waiting"  # wait for server to run this task
    Running = "running"  # server is running this task
    Success = "success"  # task is completed successfully
    Failed = "failed"  # task is failed


class LabelTypes(enum.Enum):
    BBox = "bbox"
    Mask = "mask"


class BaseTask(abc.ABC):
    def __init__(self):
        super().__init__()

        self.config = None
        self.task_uuid = None
        self.status = None
        self.error = None
        self._result = None

    @property
    @abc.abstractmethod
    def api_path(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def api_body(self):
        raise NotImplementedError

    @abc.abstractmethod
    def format_result(self, result: dict) -> pydantic.BaseModel:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def result(self):
        raise NotImplementedError

    @property
    def headers(self):
        return {"Token": self.config.token}

    @property
    def api_trigger_url(self):
        return f"https://{self.config.endpoint.value}/tasks/{self.api_path}"

    @property
    def api_check_url(self):
        return f"https://{self.config.endpoint.value}/task_statuses/{self.task_uuid}"

    def trigger(self, config: Config):
        if self.status is not None:
            raise RuntimeError("Task is already triggered, you can't triggered twice.")

        self.config = config
        self.status = TaskStatus.Triggering

        rsp = requests.post(self.api_trigger_url, json=self.api_body, headers=self.headers, timeout=2)

        rsp_json = rsp.json()
        if rsp_json["code"] != 0:
            raise RuntimeError(f"Failed to trigger task, error: {rsp_json['msg']}")
        self.task_uuid = rsp_json["data"]["task_uuid"]
        logger.info(f"Task {self.task_uuid} is triggered successfully")

    def check(self):
        if self.status is None:
            raise RuntimeError("Task is not triggered, you can't check it's status")

        api = self.api_check_url
        rsp = requests.get(api, timeout=2, headers=self.headers)
        rsp_json = rsp.json()
        if rsp_json["code"] != 0:
            raise RuntimeError(f"Failed to check task, error: {rsp_json['msg']}")

        task_data = rsp_json["data"]
        self.status = TaskStatus(task_data["status"])
        if self.status == TaskStatus.Success:
            result = task_data["result"]
            self._result = self.format_result(result)
        elif self.status == TaskStatus.Failed:
            self.error = task_data["error"]

    def wait(self):
        if self.status is None:
            raise RuntimeError("Task is not triggered, you can't wait for it's result")

        while True:
            if self.status not in {TaskStatus.Triggering, TaskStatus.Waiting, TaskStatus.Running}:
                return

            self.check()
            if self.status == TaskStatus.Waiting:
                logger.info(f"task {self.task_uuid} is waiting")
            elif self.status == TaskStatus.Running:
                logger.info(f"task {self.task_uuid} is running")
            elif self.status == TaskStatus.Success:
                logger.info(f"task {self.task_uuid} is success")
                return
            elif self.status == TaskStatus.Failed:
                logger.info(f"task {self.task_uuid} is failed")
                return
            time.sleep(0.5)

    def run(self, config: Config):
        self.trigger(config)
        self.wait()
