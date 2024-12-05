from typing import List
from typing import Optional

import pydantic

from dds_cloudapi_sdk.tasks.base import BaseTask
from dds_cloudapi_sdk.tasks.prompt import TextPrompt
from dds_cloudapi_sdk.tasks.types import BBox
from dds_cloudapi_sdk.tasks.types import DetectionTarget
from dds_cloudapi_sdk.tasks.types import HandKeypoints
from dds_cloudapi_sdk.tasks.types import ObjectMask
from dds_cloudapi_sdk.tasks.types import PoseKeypoints


class DinoxObject(pydantic.BaseModel):
    """
    The object detected by Dinox task.

    :param category: the category name of the object
    :param score: the predict score of the object
    :param bbox: the bbox of the object, [xmin, ymin, xmax, ymax]
    :param rle: the detected :class:`Mask <dds_cloudapi_sdk.tasks.dinox.ObjectMask>` object
    :param pose: the pose of the person, [x, y, visibility, score, ...]
    :param hand: the hand of the person, [x, y, visibility, score, ...]
    """

    category: str
    score: float
    bbox: Optional[BBox] = None
    mask: Optional[ObjectMask] = None
    pose: Optional[PoseKeypoints] = None
    hand: Optional[HandKeypoints] = None


class TaskResult(pydantic.BaseModel):
    """
    The task result of Dinox task.

    """

    objects: List[
        DinoxObject]  #: a list of detected objects of :class:`DinoxObject <dds_cloudapi_sdk.tasks.dinox.DinoxObject>`


class DinoxTask(BaseTask):
    def __init__(
        self,
        image_url: str,
        prompts: List[TextPrompt],
        bbox_threshold: float = 0.25,
        iou_threshold: float = 0.8,
        targets: List[DetectionTarget] = None,

    ):
        self.image_url = image_url
        self.prompts = prompts
        self.bbox_threshold = bbox_threshold
        self.iou_threshold = iou_threshold
        self.targets = targets
        super().__init__()

    @property
    def api_path(self):
        return "dinox"

    @property
    def api_body(self):
        data = {
            "image"         : self.image_url,
            "prompts"       : [p.dict() for p in self.prompts],
            "bbox_threshold": self.bbox_threshold,
            "iou_threshold" : self.iou_threshold,
            "targets"       : [t.value for t in self.targets] if self.targets else None

        }
        return data

    @property
    def result(self) -> TaskResult:
        return self._result

    def format_result(self, result: dict) -> TaskResult:
        return TaskResult(**result)


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
    prompt = TextPrompt(text="<prompt_free>")
    task = DinoxTask(
        image_url="https://algosplt.oss-cn-shenzhen.aliyuncs.com/test_files/tasks/dinox/08.jpg",
        prompts=[prompt],
        targets=[DetectionTarget.BBox, DetectionTarget.Mask, DetectionTarget.Hand, DetectionTarget.Pose],
        bbox_threshold=0.55,
        iou_threshold=0.8
    )

    client.run_task(task)
    for obj in task.result.objects:
        print(obj)


if __name__ == "__main__":
    test()
