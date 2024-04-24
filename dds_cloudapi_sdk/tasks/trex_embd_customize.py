"""
The TRex Embedding Customization algorithm enables user customizing a visual embedding file from prompts on multiple images.

The embedding file then can be used to inference images by Trex Embedding Inference algorithm.

"""

from typing import List
from typing import Union

import pydantic

from dds_cloudapi_sdk.tasks.base import BaseTask
from dds_cloudapi_sdk.tasks.prompt import BatchPointPrompt
from dds_cloudapi_sdk.tasks.prompt import BatchRectPrompt
from dds_cloudapi_sdk.tasks.prompt import PromptType


class TaskResult(pydantic.BaseModel):
    """
    The task result of TRexEmbdCustomize task.

    :param visual embedding: the url of the embedding file
    """

    embd: str  #: the url of the visual embedding file


class TRexEmbdCustomize(BaseTask):
    """
    Trigger the Trex Embd Customization algorithm.

    This task generates an embedding file from prompts on multiple images.
    However, all prompts are limited to one type, either point or rect.

    The task is similar to :class:`TRexGenericInfer <dds_cloudapi_sdk.tasks.trex_generic.TRexGenericInfer>`, but it leaves the infer image out for it is generating an embedding file instead of inferring.

    :param batch_prompts: list of :class:`BatchRectPrompt <dds_cloudapi_sdk.tasks.prompt.BatchRectPrompt>` objects or :class:`BatchPointPrompt <dds_cloudapi_sdk.tasks.prompt.BatchPointPrompt>`.
    """

    def __init__(self,
                 batch_prompts: Union[List[BatchRectPrompt], List[BatchPointPrompt]],
                 ):
        super().__init__()

        self.batch_prompts = batch_prompts

    @property
    def api_path(self):
        return "trex_embd_customize"

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
            "batch_prompts": batch_prompts
        }

        return data

    @property
    def result(self) -> TaskResult:
        """
        Get the formatted :class:`TaskResult <dds_cloudapi_sdk.tasks.trex_embd_customize.TaskResult>` object.
        """
        return self._result

    def format_result(self, result: dict) -> TaskResult:
        return TaskResult(**result)


def test():
    """
    python -m dds_cloudapi_sdk.tasks.trex_embd_customize
    """

    import os
    test_token = os.environ["DDS_CLOUDAPI_TEST_TOKEN"]

    import logging

    logging.basicConfig(level=logging.INFO)

    from dds_cloudapi_sdk import Config
    from dds_cloudapi_sdk import Client

    config = Config(test_token)
    client = Client(config)

    image_url = "https://algosplt.oss-cn-shenzhen.aliyuncs.com/test_files/tasks/ivp/04_b.jpg"

    batch_prompts = [
        BatchRectPrompt(
            image=image_url,
            rects=[[475.18413597733706, 550.1983002832861, 548.1019830028329, 599.915014164306]]
        )
    ]
    task = TRexEmbdCustomize(
        batch_prompts=batch_prompts
    )

    client.run_task(task)
    embd_url = task.result.embd
    print(embd_url)


if __name__ == "__main__":
    test()
