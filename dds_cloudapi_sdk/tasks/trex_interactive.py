"""
The TRex Interactive Inference algorithm enables user prompting on image and get the boxes and scores on the same image.

This algorithm supports batch inference for multiple images, and for every image, multiple prompts are supported.

However, the prompt type for every image is limited to either point or rect.

"""

from typing import List
from typing import Union

import pydantic

from dds_cloudapi_sdk.tasks.base import BaseTask
from dds_cloudapi_sdk.tasks.prompt import BatchPointInfer
from dds_cloudapi_sdk.tasks.prompt import BatchRectInfer
from dds_cloudapi_sdk.tasks.prompt import BatchRectPrompt
from dds_cloudapi_sdk.tasks.prompt import PromptType


class TRexObject(pydantic.BaseModel):
    """
    The object detected by TRexInteractiveInfer task.

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

    :param object_batches: a 2D list of detected objects of :class:`TRexObject <dds_cloudapi_sdk.tasks.trex_interactive.TRexObject>`, each inner list is the detected objects of one image
    """

    object_batches: List[List[TRexObject]]  #: a 2D list of detected objects of :class:`TRexObject <dds_cloudapi_sdk.tasks.trex_interactive.TRexObject>`, each inner list is the detected objects of one image


class TRexInteractiveInfer(BaseTask):
    """
    Trigger the Trex Interactive Inference algorithm.

    This task can process prompts from multiple images, and each image can have several prompts.
    However, each task is limited to one type of prompt, either point or rect.

    :param batch_infers: list of :class:`BatchPointInfer <dds_cloudapi_sdk.tasks.prompt.BatchPointInfer>` objects or :class:`BatchRectInfer <dds_cloudapi_sdk.tasks.prompt.BatchRectInfer>`.
    """

    def __init__(self,
                 batch_infers: Union[List[BatchPointInfer], List[BatchRectInfer]],
                 ):
        super().__init__()

        self.batch_infers = batch_infers

    @property
    def api_path(self):
        return "trex_interactive_infer"

    @property
    def api_body(self):
        batch_infers = []

        for infer in self.batch_infers:
            infer_data = infer.dict()
            if isinstance(infer, BatchPointInfer):
                infer_data["prompt_type"] = PromptType.Point.value
            elif isinstance(infer, BatchRectInfer):
                infer_data["prompt_type"] = PromptType.Rect.value
            batch_infers.append(infer_data)

        return {"batch_infers": batch_infers}

    @property
    def result(self) -> TaskResult:
        """
        Get the formatted :class:`TaskResult <dds_cloudapi_sdk.tasks.trex_generic.TaskResult>` object.
        """
        return self._result

    def format_result(self, result: dict) -> TaskResult:
        return TaskResult(**result)


def test():
    """
    python -m dds_cloudapi_sdk.tasks.trex_interactive
    """
    import os
    test_token = os.environ["DDS_CLOUDAPI_TEST_TOKEN"]

    import logging

    logging.basicConfig(level=logging.INFO)

    from dds_cloudapi_sdk import Config
    from dds_cloudapi_sdk import Client

    config = Config(test_token)
    client = Client(config)

    infer_image = "https://algosplt.oss-cn-shenzhen.aliyuncs.com/test_files/tasks/ivp/04_b.jpg"

    infer_1 = BatchRectInfer(
        image=infer_image,
        prompts=[
            BatchRectPrompt(category_id=1, rects=[[475.18413597733706, 550.1983002832861, 548.1019830028329, 599.915014164306]])
        ]
    )
    task = TRexInteractiveInfer([infer_1])

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
