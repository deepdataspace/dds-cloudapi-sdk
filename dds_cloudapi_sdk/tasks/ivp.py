from io import BytesIO
from typing import Any
from typing import List
from typing import Tuple

import numpy as np
import pydantic
import requests
from PIL import Image

from dds_cloudapi_sdk.tasks.base import BaseTask
from dds_cloudapi_sdk.tasks.base import LabelTypes
from dds_cloudapi_sdk.tasks.prompt import RectPrompt


class IVPObjectMask(pydantic.BaseModel):
    counts: str
    size: List[int] = None

    def model_post_init(self, __context: Any) -> None:
        pass


class IVPObject(pydantic.BaseModel):
    score: float
    bbox: List[float] = None
    mask: IVPObjectMask = None


class TaskResult(pydantic.BaseModel):
    mask_url: str = None
    objects: List[IVPObject] = []


class IVPTask(BaseTask):
    def __init__(self,
                 prompt_image_url: str,
                 prompts: List[RectPrompt],
                 infer_image_url: str,
                 infer_label_types: List[LabelTypes],
                 ):
        """
        Initialize an IVP task.

        :param prompt_image_url: The image the prompts are acting on. The url muse be public accessible.
        :param prompts: List of rect prompts which are drawn on the prompt image.
        :param infer_image_url: The image to be inferred on. The url muse be public accessible.
        :param infer_label_types: The label types to be inferred, possible values are LabelTypes.BBox and LabelTypes.Mask.
        """

        super().__init__()

        self.infer_image = infer_image_url
        self.prompt_image = prompt_image_url
        self.label_types = infer_label_types
        self.prompts = prompts

        self._infer_image_width = None
        self._infer_image_height = None

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
        return self._result

    def get_infer_image_size(self):
        if self._infer_image_width is None or self._infer_image_height is None:
            rsp = requests.get(self.infer_image, timeout=2)
            img = Image.open(BytesIO(rsp.content))
            width, height = img.size
            self._infer_image_width = width
            self._infer_image_height = height

    @property
    def infer_image_width(self):
        self.get_infer_image_size()
        return self._infer_image_width

    @property
    def infer_image_height(self):
        self.get_infer_image_size()
        return self._infer_image_height

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

    def rle2rgba(self, rle_counts: str, width: int = None, height: int = None) -> Image.Image:
        if width is None or height is None:
            width = self.infer_image_width
            height = self.infer_image_height

        # convert rle counts to mask array
        rle = self.string2rle(rle_counts)
        shape = (height, width)  # height, width
        mask_array = self.rle2mask(rle, shape)

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

    import logging

    logging.basicConfig(level=logging.INFO)

    from dds_cloudapi_sdk import Client

    client = Client("dds-app-free", )
    task = IVPTask(
        prompt_image_url="https://dev.deepdataspace.com/static/04_a.ae28c1d6.jpg",
        prompts=[RectPrompt(rect=[475.18413597733706, 550.1983002832861, 548.1019830028329, 599.915014164306])],
        infer_image_url="https://dev.deepdataspace.com/static/04_b.ae28c1d6.jpg",
        infer_label_types=[LabelTypes.BBox],
    )

    client.run_task(task)
    print(task.result)

    task = IVPTask(
        prompt_image_url="https://dev.deepdataspace.com/static/04_a.ae28c1d6.jpg",
        prompts=[RectPrompt(rect=[475.18413597733706, 550.1983002832861, 548.1019830028329, 599.915014164306])],
        infer_image_url="https://dev.deepdataspace.com/static/04_b.ae28c1d6.jpg",
        infer_label_types=[LabelTypes.Mask],
    )

    client.run_task(task)
    print(task.result)

    task = IVPTask(
        prompt_image_url="https://dev.deepdataspace.com/static/04_a.ae28c1d6.jpg",
        prompts=[RectPrompt(rect=[475.18413597733706, 550.1983002832861, 548.1019830028329, 599.915014164306])],
        infer_image_url="https://dev.deepdataspace.com/static/04_b.ae28c1d6.jpg",
        infer_label_types=[LabelTypes.Mask, LabelTypes.BBox],
    )

    client.run_task(task)
    print(task.result)
    for obj in task.result.objects:
        if obj.mask is not None:
            mask = task.rle2rgba(obj.mask.counts)
            mask.save("mask.png")
            break


if __name__ == "__main__":
    test()
