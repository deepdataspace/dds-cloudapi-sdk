"""
Every task represents an algorithm.

The Task class provides a unified interface for running different algorithms:

- Initialize the task object::

    from dds_cloudapi_sdk.tasks import IVPTask

    task = IVPTask(...)  # take IVP for example, parameters are left out

- Run the task with the client object::

    client.run(task)  # steps to initialize a client are left out

- Get the result from **task.result** property::

    print(task.result)

"""

from dds_cloudapi_sdk.tasks.base import LabelTypes
from dds_cloudapi_sdk.tasks.base import TaskStatus
from dds_cloudapi_sdk.tasks.gsam import BaseGSAMTask
from dds_cloudapi_sdk.tasks.gsam import TinyGSAMTask
from dds_cloudapi_sdk.tasks.ivp import IVPTask
from dds_cloudapi_sdk.tasks.prompt import *
from dds_cloudapi_sdk.tasks.trex_generic import TRexGenericInfer

__all__ = [
    "TaskStatus",
    "LabelTypes",
    "TextPrompt",
    "RectPrompt",
    "BatchPointPrompt",
    "BatchRectPrompt",
    "IVPTask",
    "TinyGSAMTask",
    "BaseGSAMTask",
    "TRexGenericInfer",
]
