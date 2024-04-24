"""
The TRex Embedding Inference algorithm enables user inferring image and get the boxes and scores on the same image by embd files
they trained from the :class:`Embedding Customization <dds_cloudapi_sdk.tasks.trex_embd_customize.TRexEmbdCustomize>`.

This algorithm supports batch inference for multiple images, and for every image, multiple embeddings are supported.
"""

from typing import List

import pydantic

from dds_cloudapi_sdk.tasks.base import BaseTask
from dds_cloudapi_sdk.tasks.prompt import BatchEmbdInfer
from dds_cloudapi_sdk.tasks.prompt import BatchEmbdPrompt
from dds_cloudapi_sdk.tasks.prompt import BatchRectPrompt
from dds_cloudapi_sdk.tasks.prompt import PromptType
from dds_cloudapi_sdk.tasks.trex_embd_customize import TRexEmbdCustomize


class TRexObject(pydantic.BaseModel):
    """
    The object detected by TRexEmbdInfer task.

    :param score: the prediction score
    :param bbox: the bounding box, [upper_left_x, upper_left_y, lower_right_x, lower_right_y]
    :param category_id: the category id of the object
    """

    score: float  #: the prediction score
    bbox: List[float]  #: the bounding box, [upper_left_x, upper_left_y, lower_right_x, lower_right_y]
    category_id: int  #: the category id of the object


class TaskResult(pydantic.BaseModel):
    """
    The task result of TRexInteractiveInfer task.

    :param object_batches: a 2D list of detected objects of :class:`TRexObject <dds_cloudapi_sdk.tasks.trex_embd_infer.TRexObject>`, each inner list is the detected objects of one image
    """

    object_batches: List[List[TRexObject]]  #: a 2D list of detected objects of :class:`TRexObject <dds_cloudapi_sdk.tasks.trex_embd_infer.TRexObject>`, each inner list is the detected objects of one image


class TRexEmbdInfer(BaseTask):
    """
    Trigger the Trex Embedding Inference algorithm.

    This task can process prompts from multiple images, and each image can have several embedding prompts.

    :param batch_infers: list of :class:`BatchPointInfer <dds_cloudapi_sdk.tasks.prompt.BatchPointInfer>` objects or :class:`BatchRectInfer <dds_cloudapi_sdk.tasks.prompt.BatchRectInfer>`.
    """

    def __init__(self,
                 batch_infers: List[BatchEmbdInfer],
                 ):
        super().__init__()

        self.batch_infers = batch_infers

    @property
    def api_path(self):
        return "trex_embd_infer"

    @property
    def api_body(self):
        batch_infers = []

        for infer in self.batch_infers:
            infer_data = infer.dict()
            infer_data["prompt_type"] = PromptType.Embd.value
            batch_infers.append(infer_data)

        print(batch_infers)
        return {"batch_infers": batch_infers}

    @property
    def result(self) -> TaskResult:
        """
        Get the formatted :class:`TaskResult <dds_cloudapi_sdk.tasks.trex_embd_infer.TaskResult>` object.
        """
        return self._result

    def format_result(self, result: dict) -> TaskResult:
        return TaskResult(**result)


def test():
    """
    python -m dds_cloudapi_sdk.tasks.trex_embd
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

    infer_1 = BatchEmbdInfer(
        image=image_url,
        prompts=[
            BatchEmbdPrompt(embd=embd_url, category_id=1)
        ]
    )
    task = TRexEmbdInfer([infer_1])
    client.run_task(task)

    for image_objects in task.result.object_batches:
        for obj in image_objects:
            print(obj.score)
            print(obj.bbox)
            print(obj.category_id)
            break
        break


if __name__ == "__main__":
    test()
