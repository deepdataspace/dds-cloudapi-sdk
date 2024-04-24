"""
Each task class in the SDK represents algorithm, offering a unified interface to execute different algorithms.

Here's how to utilize a task class effectively:

- Initialize the task object::

    from dds_cloudapi_sdk.tasks import IVPTask

    task = IVPTask(...)  # take IVP for example, parameters are omitted for brevity

- Run the task with the client object::

    client.run(task)  # steps to initialize a client are omitted for brevity

- Get the result from **task.result** property::

    print(task.result)

"""

from dds_cloudapi_sdk.tasks.base import LabelTypes
from dds_cloudapi_sdk.tasks.base import TaskStatus
from dds_cloudapi_sdk.tasks.detection import DetectionModel
from dds_cloudapi_sdk.tasks.detection import DetectionTarget
from dds_cloudapi_sdk.tasks.detection import DetectionTask
from dds_cloudapi_sdk.tasks.gsam import BaseGSAMTask
from dds_cloudapi_sdk.tasks.gsam import TinyGSAMTask
from dds_cloudapi_sdk.tasks.ivp import IVPTask
from dds_cloudapi_sdk.tasks.prompt import *
from dds_cloudapi_sdk.tasks.trex_embd_customize import TRexEmbdCustomize
from dds_cloudapi_sdk.tasks.trex_embd_infer import TRexEmbdInfer
from dds_cloudapi_sdk.tasks.trex_generic import TRexGenericInfer
from dds_cloudapi_sdk.tasks.trex_interactive import TRexInteractiveInfer

__all__ = [
    "TaskStatus",
    "LabelTypes",
    "TextPrompt",
    "RectPrompt",
    "BatchPointPrompt",
    "BatchRectPrompt",
    "BatchEmbdPrompt",
    "BatchPointInfer",
    "BatchRectInfer",
    "BatchEmbdInfer",
    "IVPTask",
    "TinyGSAMTask",
    "BaseGSAMTask",
    "TRexGenericInfer",
    "TRexInteractiveInfer",
    "TRexEmbdCustomize",
    "TRexEmbdInfer",
    "DetectionTask",
    "DetectionModel",
    "DetectionTarget"
]
