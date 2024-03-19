"""
The TRex Generic Inference algorithm enables user prompting on multiple images and get the boxes, scores on one target image.

This algorithm hypothesis that there is only one category per batch image and it does not support batch inference.

This algorithm is available in DDS CloudAPI SDK in two variants:

- the "swint" for "tiny" model through :class:`TinyTRexGenericInfer <dds_cloudapi_sdk.tasks.trex_generic.TinyTRexGenericInfer>` class
- the "swinl" for "large" model through :class:`LargeTRexGenericInfer <dds_cloudapi_sdk.tasks.trex_generic.LargeTRexGenericInfer>` class
"""

import enum
from typing import List
from typing import Union

import pydantic

from dds_cloudapi_sdk.tasks.base import BaseTask
from dds_cloudapi_sdk.tasks.prompt import BatchPointPrompt
from dds_cloudapi_sdk.tasks.prompt import BatchRectPrompt
from dds_cloudapi_sdk.tasks.prompt import PromptType


class ModelType(enum.Enum):
    """
    | This enumerator represents the models the TRex Generic Inference algorithm uses.
    | The :class:`TinyTRexGenericInfer <dds_cloudapi_sdk.tasks.trex_generic.TinyTRexGenericInfer>` will use the SwinT model.
    | And the :class:`LargeTRexGenericInfer <dds_cloudapi_sdk.tasks.trex_generic.LargeTRexGenericInfer>` will use the SwinL model.
    """

    SwinT = "swint"  #: The swint model
    SwinL = "swinl"  #: The swinl model


class TRexObject(pydantic.BaseModel):
    """
    The object detected by TRexGenericInfer task.

    :param score: the prediction score
    :param bbox: the bounding box, [upper_left_x, upper_left_y, lower_right_x, lower_right_y]
    """

    score: float  #: the prediction score
    bbox: List[float]  #: the bounding box, [upper_left_x, upper_left_y, lower_right_x, lower_right_y]


class TaskResult(pydantic.BaseModel):
    """
    The task result of TRexGenericInfer task.

    :param objects: a list of detected objects of :class:`TRexObject <dds_cloudapi_sdk.tasks.trex_generic.TRexObject>`
    """

    objects: List[TRexObject]  #: a list of detected objects of :class:`TRexObject <dds_cloudapi_sdk.tasks.trex_generic.TRexObject>`


class _TRexGenericInfer(BaseTask):
    def __init__(self,
                 image_url: str,
                 batch_prompts: Union[List[BatchRectPrompt], List[BatchPointPrompt]],
                 model_type: ModelType,
                 ):
        super().__init__()

        self.image_url = image_url
        self.model_type = model_type
        self.batch_prompts = batch_prompts

    @property
    def api_path(self):
        return "trex_generic_infer"

    @property
    def api_body(self):
        batch_prompts = {
            "prompts": [prompt.dict() for prompt in self.batch_prompts]
        }
        if isinstance(self.batch_prompts[0], BatchPointPrompt):
            batch_prompts["type"] = PromptType.Point.value
        elif isinstance(self.batch_prompts[0], BatchRectPrompt):
            batch_prompts["type"] = PromptType.Rect.value

        data = {
            "image"        : self.image_url,
            "model_type"   : self.model_type.value,
            "batch_prompts": batch_prompts
        }

        return data

    @property
    def result(self) -> TaskResult:
        """
        Get the formatted :class:`TaskResult <dds_cloudapi_sdk.tasks.trex_generic.TaskResult>` object.
        """
        return self._result

    def format_result(self, result: dict) -> TaskResult:
        return TaskResult(**result)


class TinyTRexGenericInfer(_TRexGenericInfer):
    """
    Trigger the Trex Generic Inference algorithm with SwinL model.

    This task can process prompts from multiple images, and each image can have several prompts.
    However, each task is limited to one type of prompt, either point or rect.

    :param image_url: the image to be inferred on.
    :param batch_prompts: list of :class:`BatchRectPrompt <dds_cloudapi_sdk.tasks.prompt.BatchRectPrompt>` objects or :class:`BatchPointPrompt <dds_cloudapi_sdk.tasks.prompt.BatchPointPrompt>`.
    """

    def __init__(self,
                 image_url: str,
                 batch_prompts: Union[List[BatchRectPrompt], List[BatchPointPrompt]],
                 ):
        super().__init__(image_url, batch_prompts, ModelType.SwinT)


class LargeTRexGenericInfer(_TRexGenericInfer):
    """
    Trigger the Trex Generic Inference algorithm with SwinL model.

    This task can process prompts from multiple images, and each image can have several prompts.
    However, each task is limited to one type of prompt, either point or rect.

    :param image_url: the image to be inferred on.
    :param batch_prompts: list of :class:`BatchRectPrompt <dds_cloudapi_sdk.tasks.prompt.BatchRectPrompt>` objects or :class:`BatchPointPrompt <dds_cloudapi_sdk.tasks.prompt.BatchPointPrompt>`.
    """

    def __init__(self,
                 image_url: str,
                 batch_prompts: Union[List[BatchRectPrompt], List[BatchPointPrompt]],
                 ):
        super().__init__(image_url, batch_prompts, ModelType.SwinL)


def test():
    """
    python -m dds_cloudapi_sdk.tasks.trex_generic
    """
    import os
    test_token = os.environ["DDS_CLOUDAPI_TEST_TOKEN"]

    import logging

    logging.basicConfig(level=logging.INFO)

    from dds_cloudapi_sdk import Config
    from dds_cloudapi_sdk import Client

    config = Config(test_token)
    client = Client(config)

    batch_prompts = [
        BatchRectPrompt(
            image="https://dev.deepdataspace.com/static/04_a.ae28c1d6.jpg",
            rects=[[475.18413597733706, 550.1983002832861, 548.1019830028329, 599.915014164306]]
        )
    ]
    task = TinyTRexGenericInfer(
        image_url="https://dev.deepdataspace.com/static/04_a.ae28c1d6.jpg",
        batch_prompts=batch_prompts
    )

    client.run_task(task)
    for obj in task.result.objects:
        print(obj)
        break

    task = LargeTRexGenericInfer(
        image_url="https://dev.deepdataspace.com/static/04_a.ae28c1d6.jpg",
        batch_prompts=batch_prompts
    )

    client.run_task(task)
    for obj in task.result.objects:
        print(obj)
        break


if __name__ == "__main__":
    test()
