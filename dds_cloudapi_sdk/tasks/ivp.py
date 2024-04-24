"""
Interactive Visual Prompting (IVP) is an interactive object detection and counting system based on the T-Rex model independently developed by the IDEA CVR team.

It enables object detection and counting through visual prompts without any training, truly realizing a single visual model applicable to multiple scenarios.

It particularly excels in counting objects in dense or overlapping scenes.

This algorithm is available in DDS CloudAPI SDK through IVPTask.
"""

from typing import List
from typing import Tuple

import numpy as np
import pydantic
from PIL import Image

from dds_cloudapi_sdk.tasks.base import BaseTask
from dds_cloudapi_sdk.tasks.base import LabelTypes
from dds_cloudapi_sdk.tasks.prompt import RectPrompt


class IVPObjectMask(pydantic.BaseModel):
    """
    | The mask detected by IVP task.
    | It's a format borrow COCO which compressing the mask image array in RLE format.
    | You can restore it back to a png image array by :func:`IVPTask.rle2rgba <dds_cloudapi_sdk.tasks.ivp.IVPTask.rle2rgba>`:

    :param counts: the compressed mask array in RLE format
    :param size: the 2d size of the array, (h, w)
    """

    counts: str  #: the compressed mask array in RLE format
    size: Tuple[int, int]  #: the 2d size of the array, (h, w)


class IVPObject(pydantic.BaseModel):
    """
    The object detected by IVP task.

    :param score: the prediction score
    :param bbox: the bounding box, [upper_left_x, upper_left_y, lower_right_x, lower_right_y]
    :param mask: the detected :class:`Mask <dds_cloudapi_sdk.tasks.ivp.IVPObjectMask>` object
    """

    score: float  # : the prediction score
    bbox: List[float] = None  #: the bounding box, [upper_left_x, upper_left_y, lower_right_x, lower_right_y]
    mask: IVPObjectMask = None  #: the detected :class:`Mask <dds_cloudapi_sdk.tasks.ivp.IVPObjectMask>` object


class TaskResult(pydantic.BaseModel):
    """
    The task result of IVP task.

    :param mask_url: an image url with all objects' mask drawn on
    :param objects: a list of detected objects of :class:`IVPObject <dds_cloudapi_sdk.tasks.ivp.IVPObject>`
    """

    mask_url: str = None
    objects: List[IVPObject] = []


class IVPTask(BaseTask):
    """
    Trigger the Interactive Visual Prompting algorithm.

    :param prompt_image_url: the image the prompts are acting on.
    :param prompts: list of :class:`RectPrompt <dds_cloudapi_sdk.tasks.prompt.RectPrompt>` objects which are drawn on the prompt image.
    :param infer_image_url: the image to be inferred on.
    :param infer_label_types: list of target :class:`LabelTypes <dds_cloudapi_sdk.base.LabelTypes>` to return.
    """

    def __init__(self,
                 prompt_image_url: str,
                 prompts: List[RectPrompt],
                 infer_image_url: str,
                 infer_label_types: List[LabelTypes],
                 ):
        super().__init__()

        self.infer_image = infer_image_url
        self.prompt_image = prompt_image_url
        self.label_types = infer_label_types
        self.prompts = prompts

    @property
    def api_path(self):
        return "ivp"

    @property
    def api_body(self):
        data = {
            "infer_image" : self.infer_image,
            "prompt_image": self.prompt_image,
            "label_types" : [label_type.value for label_type in self.label_types],
            "prompts"     : [prompt.dict() for prompt in self.prompts]
        }

        return data

    @property
    def result(self) -> TaskResult:
        """
        Get the formatted :class:`TaskResult <dds_cloudapi_sdk.tasks.ivp.TaskResult>` object.
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

    def rle2rgba(self, mask_obj: IVPObjectMask) -> Image.Image:
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


def test():
    """
    python -m dds_cloudapi_sdk.tasks.ivp
    """
    import os
    test_token = os.environ["DDS_CLOUDAPI_TEST_TOKEN"]

    import logging

    logging.basicConfig(level=logging.INFO)

    from dds_cloudapi_sdk import Config
    from dds_cloudapi_sdk import Client

    config = Config(test_token)
    client = Client(config)
    task = IVPTask(
        prompt_image_url="https://algosplt.oss-cn-shenzhen.aliyuncs.com/test_files/tasks/ivp/04_b.jpg",
        prompts=[RectPrompt(rect=[475.18413597733706, 550.1983002832861, 548.1019830028329, 599.915014164306])],
        infer_image_url="https://algosplt.oss-cn-shenzhen.aliyuncs.com/test_files/tasks/ivp/04_b.jpg",
        infer_label_types=[LabelTypes.BBox],
    )

    client.run_task(task)
    print(task.result)

    task = IVPTask(
        prompt_image_url="https://algosplt.oss-cn-shenzhen.aliyuncs.com/test_files/tasks/ivp/04_b.jpg",
        prompts=[RectPrompt(rect=[475.18413597733706, 550.1983002832861, 548.1019830028329, 599.915014164306])],
        infer_image_url="https://algosplt.oss-cn-shenzhen.aliyuncs.com/test_files/tasks/ivp/04_b.jpg",
        infer_label_types=[LabelTypes.Mask],
    )

    client.run_task(task)
    print(task.result)

    task = IVPTask(
        prompt_image_url="https://algosplt.oss-cn-shenzhen.aliyuncs.com/test_files/tasks/ivp/04_b.jpg",
        prompts=[RectPrompt(rect=[475.18413597733706, 550.1983002832861, 548.1019830028329, 599.915014164306])],
        infer_image_url="https://algosplt.oss-cn-shenzhen.aliyuncs.com/test_files/tasks/ivp/04_b.jpg",
        infer_label_types=[LabelTypes.Mask, LabelTypes.BBox],
    )

    client.run_task(task)
    print(task.result)
    for obj in task.result.objects:
        if obj.mask is not None:
            mask = task.rle2rgba(obj.mask)
            mask.save("mask.png")
            break


if __name__ == "__main__":
    test()
