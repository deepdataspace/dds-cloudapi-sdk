"""
Run a detection task with text prompts for bbox or mask.

Supported models:
 - Grounding-Dino-1
 - Grounding-Dino-1.5-Edge
 - Grounding-Dino-1.5-Pro
 - Grounding-Dino-1.6-Edge
 - Grounding-Dino-1.6-Pro
"""

import enum
import sys
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
import pydantic
from PIL import Image

from dds_cloudapi_sdk.tasks.base import BaseTask
from dds_cloudapi_sdk.tasks.prompt import TextPrompt


class DetectionTarget(enum.Enum):
    BBox = "bbox"  #:
    Mask = "mask"  #:


class DetectionModel(enum.Enum):
    GDino1 = "GroundingDino-1"  #:
    GDino1_5_Edge = "GroundingDino-1.5-Edge"  #:
    GDino1_5_Pro = "GroundingDino-1.5-Pro"  #:
    GDino1_6_Edge = "GroundingDino-1.6-Edge"  #:
    GDino1_6_Pro = "GroundingDino-1.6-Pro"  #:


class DetectionObjectMask(pydantic.BaseModel):
    """
    | The mask detected by detection task.
    | It's a format borrow COCO which compressing the mask image array in RLE format.
    | You can restore it back to a png image array by :func:`DetectionTask.rle2rgba <dds_cloudapi_sdk.tasks.detection.DetectionTask.rle2rgba>`:

    :param counts: the compressed mask array in RLE format
    :param size: the 2d size of the array, (h, w)
    """

    counts: str  #: the compressed mask array in RLE format
    size: Tuple[int, int]  #: the 2d size of the array, (h, w)


class DetectionObject(pydantic.BaseModel):
    """
    The object detected by detection task.

    :param score: the prediction score
    :param bbox: the bounding box, [upper_left_x, upper_left_y, lower_right_x, lower_right_y]
    :param mask: the detected :class:`Mask <dds_cloudapi_sdk.tasks.detection.DetectionObjectMask>` object
    """

    score: float  # : the prediction score
    category: str  #: the category of the object
    bbox: List[float] = None  #: the bounding box, [upper_left_x, upper_left_y, lower_right_x, lower_right_y]
    mask: Union[DetectionObjectMask, None] = None  #: the detected :class:`Mask <dds_cloudapi_sdk.tasks.detection.DetectionObjectMask>` object


class TaskResult(pydantic.BaseModel):
    """
    The task result of detection task.

    :param mask_url: an image url with all objects' mask drawn on
    :param objects: a list of detected objects of :class:`DetectionObject <dds_cloudapi_sdk.tasks.detection.DetectionObject>`
    """

    mask_url: Union[str, None] = None
    objects: List[DetectionObject] = []


class DetectionTask(BaseTask):
    """
    Trigger a detection task.

    :param image_url: the image url for detection.
    :param prompts: list of :class:`TextPrompt <dds_cloudapi_sdk.tasks.prompt.TextPrompt>`.
    :param targets: detection targets, list of :class:`DetectionTarget <dds_cloudapi_sdk.tasks.detection.DetectionTarget>`.
    :param model: the model to be used for detection, supported models are enumerated by :class:`DetectionModel <dds_cloudapi_sdk.tasks.detection.DetectionModel>`.
    :param bbox_threshold: the detection threshold for bbox
    :param iou_threshold: the detection threshold for iou
    """

    def __init__(self,
                 image_url: str,
                 prompts: List[TextPrompt],
                 targets: List[DetectionTarget],
                 model: DetectionModel,
                 bbox_threshold: float = 0.25,
                 iou_threshold: float = 0.8

                 ):
        super().__init__()

        self.image_url = image_url
        self.prompts = prompts
        self.targets = targets
        self.model = model
        self.bbox_threshold = bbox_threshold
        self.iou_threshold = iou_threshold

    @property
    def api_path(self):
        return "detection"

    @property
    def api_body(self):
        data = {
            "image"  : self.image_url,
            "prompts": [p.dict() for p in self.prompts],
            "targets": [t.value for t in self.targets],
            "model"  : self.model.value,
            "bbox_threshold": self.bbox_threshold,
            "iou_threshold": self.iou_threshold
        }

        return data

    @property
    def result(self) -> TaskResult:
        """
        Get the formatted :class:`TaskResult <dds_cloudapi_sdk.tasks.detection.TaskResult>` object.
        """
        return self._result

    @staticmethod
    def string2rle(rle_str: str) -> List[int]:
        p = 0
        cnts = []

        while p < len(rle_str) and rle_str[p]:
            x = 0
            k = 0
            more = 1

            while more:
                c = ord(rle_str[p]) - 48
                x |= (c & 0x1f) << 5 * k
                more = c & 0x20
                p += 1
                k += 1

                if not more and (c & 0x10):
                    x |= -1 << 5 * k

            if len(cnts) > 2:
                x += cnts[len(cnts) - 2]
            cnts.append(x)
        return cnts

    @staticmethod
    def rle2mask(cnts: List[int], size: Tuple[int, int], label=1):
        img = np.zeros(size, dtype=np.uint8)

        ps = 0
        for i in range(0, len(cnts), 2):
            ps += cnts[i]

            for j in range(cnts[i + 1]):
                x = (ps + j) % size[1]
                y = (ps + j) // size[1]

                if y < size[0] and x < size[1]:
                    img[y, x] = label
                else:
                    break

            ps += cnts[i + 1]

        return img

    def rle2rgba(self, mask_obj: DetectionObjectMask) -> Image.Image:
        """
        Convert the compressed RLE string of mask object to png image object.

        :param mask_obj: The :class:`Mask <dds_cloudapi_sdk.tasks.ivp.IVPObjectMask>` object detected by this task
        """

        # convert rle counts to mask array
        rle = self.string2rle(mask_obj.counts)
        mask_array = self.rle2mask(rle, mask_obj.size)

        # convert the array to a 4-channel RGBA image
        mask_alpha = np.where(mask_array == 1, 255, 0).astype(np.uint8)
        mask_rgba = np.stack((255 * np.ones_like(mask_alpha),
                              255 * np.ones_like(mask_alpha),
                              255 * np.ones_like(mask_alpha),
                              mask_alpha),
                             axis=-1)
        image = Image.fromarray(mask_rgba, "RGBA")
        return image

    def format_result(self, result: dict) -> TaskResult:
        return TaskResult(**result)


def _test_specific_model(model: DetectionModel):
    import os
    test_token = os.environ["DDS_CLOUDAPI_TEST_TOKEN"]

    import logging

    logging.basicConfig(level=logging.INFO)

    from dds_cloudapi_sdk import Config
    from dds_cloudapi_sdk import Client

    # test with gdino 1.5 pro, for both bbox and mask
    config = Config(test_token)
    client = Client(config)
    task = DetectionTask(
        image_url="https://algosplt.oss-cn-shenzhen.aliyuncs.com/test_files/tasks/detection/iron_man.jpg",
        prompts=[TextPrompt(text="iron man")],
        targets=[DetectionTarget.Mask, DetectionTarget.BBox],
        model=model,
    )
    client.run_task(task)

    assert task.result.mask_url is not None
    for obj in task.result.objects:
        assert obj.score is not None
        assert obj.category is not None
        assert obj.bbox is not None
        assert obj.mask is not None
        mask = task.rle2rgba(obj.mask)
        mask.save("mask.png")
        break

    # test with gdino 1.5 pro, for both bbox only
    task = DetectionTask(
        image_url="https://algosplt.oss-cn-shenzhen.aliyuncs.com/test_files/tasks/detection/iron_man.jpg",
        prompts=[TextPrompt(text="iron man")],
        targets=[DetectionTarget.BBox],
        model=model,
    )
    client.run_task(task)

    assert task.result.mask_url is None
    for obj in task.result.objects:
        assert obj.score is not None
        assert obj.category is not None
        assert obj.bbox is not None
        assert obj.mask is None

    # test with gdino 1.5 pro, for mask only
    config = Config(test_token)
    client = Client(config)
    task = DetectionTask(
        image_url="https://algosplt.oss-cn-shenzhen.aliyuncs.com/test_files/tasks/detection/iron_man.jpg",
        prompts=[TextPrompt(text="iron man")],
        targets=[DetectionTarget.Mask],
        model=model,
    )
    client.run_task(task)

    assert task.result.mask_url is not None
    for obj in task.result.objects:
        assert obj.score is not None
        assert obj.category is not None
        assert obj.bbox is None
        assert obj.mask is not None
        mask = task.rle2rgba(obj.mask)
        mask.save("mask.png")
        break


def test_gdino_1():
    return _test_specific_model(DetectionModel.GDino1)


def test_gdino_1_5_edge():
    return _test_specific_model(DetectionModel.GDino1_5_Edge)


def test_gdino_1_5_pro():
    return _test_specific_model(DetectionModel.GDino1_5_Pro)

def test_gdino_1_6_edge():
    return _test_specific_model(DetectionModel.GDino1_6_Edge)

def test_gdino_1_6_pro():
    return _test_specific_model(DetectionModel.GDino1_6_Pro)

def test():
    """
    python -m dds_cloudapi_sdk.tasks.detection
    """

    target = None
    if len(sys.argv) >= 2:
        target = sys.argv[1]

    target_map = {
        "gdino_1_5_pro" : test_gdino_1_5_pro,
        "gdino_1_5_edge": test_gdino_1_5_edge,
        "gdino_1_6_pro" : test_gdino_1_6_pro,
        "gdino_1_6_edge": test_gdino_1_6_edge,
        "gdino_1"       : test_gdino_1,
    }

    target_tests = target_map.values() if target is None else [target_map[target]]
    for t in target_tests:
        t()


if __name__ == "__main__":
    test()
