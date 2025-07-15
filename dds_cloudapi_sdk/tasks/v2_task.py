import logging
from typing import Any
from typing import Dict
from typing import List

import cv2
import pycocotools.mask as maskUtils

from dds_cloudapi_sdk.image_resizer import image_to_base64
from dds_cloudapi_sdk.image_resizer import resize_image
from dds_cloudapi_sdk.rle_util import mask_to_rle
from dds_cloudapi_sdk.rle_util import rle_to_array
from dds_cloudapi_sdk.tasks.base import BaseTask


class MaskFormat:
    DDS_RLE = "dds_rle"
    COCO_RLE = "coco_rle"


class ResizeHelper:
    _original_width = None
    _original_height = None
    _ratio = None
    RESIZE_TARGETS = (
        "bbox",
        "mask",
        "pose_keypoints",
        "hand_keypoints",
    )
    SUPPORTED_MASK_FORMATS = (
        MaskFormat.DDS_RLE,
        MaskFormat.COCO_RLE,
    )

    def __init__(self, original_width: int, original_height: int, ratio: float):
        self._original_width = original_width
        self._original_height = original_height
        self._ratio = ratio

    @classmethod
    def is_resizable(cls, api_body: dict) -> bool:
        targets = api_body.get('targets')
        mask_format = api_body.get('mask_format')
        if targets and any(target not in cls.RESIZE_TARGETS for target in targets):
            return False
        if mask_format and mask_format not in cls.SUPPORTED_MASK_FORMATS:
            return False
        return True

    @classmethod
    def image_max_size(cls, api_path: str) -> int:
        if api_path == "/v2/task/trex/detection":
            return 1333
        elif api_path == "/v2/task/application/change_cloth_color":
            return 2048
        else:
            return 1536

    def format_result(self, result: dict) -> dict:
        try:
            logging.debug(f"resize original result: {result}")
            for item in result['objects']:
                if item.get('bbox'):
                    item['bbox'] = self.resize_bbox(item['bbox'])
                if item.get('mask'):
                    item['mask'] = self.resize_mask(item['mask'])
                if item.get('pose'):
                    item['pose'] = self.resize_keypoints(item['pose'])
                if item.get('hand'):
                    item['hand'] = self.resize_keypoints(item['hand'])
            return result
        except Exception as e:
            logging.exception(
                f"Error formatting result: {result}, \n"
                f"error: {e}\n"
                f"_original_width: {self._original_width}, \n"
                f"_original_height: {self._original_height}, \n"
                f"_ratio: {self._ratio}"
            )
            return result

    def resize_bbox(self, bbox: list) -> list:
        return [int(coord / self._ratio) for coord in bbox]

    def resize_mask(self, mask: dict) -> dict:
        mask_format = mask.get('format', MaskFormat.DDS_RLE)
        if mask_format == MaskFormat.DDS_RLE:
            return self.resize_dds_rle_mask(mask)
        elif mask_format == MaskFormat.COCO_RLE:
            return self.resize_coco_rle_mask(mask)
        else:
            logging.error(f"ResizeHelper: Unsupported mask format: {mask_format}")
            return mask

    def resize_dds_rle_mask(self, mask: dict) -> dict:
        img = rle_to_array(
            mask['counts'],
            mask['size'][0] * mask['size'][1]
        ).reshape(mask['size'])
        img = cv2.resize(
            img,
            (self._original_width, self._original_height)
        )
        return {
            'counts': mask_to_rle(img, encode=True),
            'size': [
                self._original_height,
                self._original_width
            ],
            'mask_format': MaskFormat.DDS_RLE,
        }

    def resize_coco_rle_mask(self, mask: dict) -> dict:
        img = maskUtils.decode(mask)
        img = cv2.resize(
            img,
            (self._original_width, self._original_height)
        )
        rle = maskUtils.encode(img)
        return {
            'counts': rle['counts'].decode('utf-8'),
            'size': rle['size'],
            'mask_format': MaskFormat.COCO_RLE,
        }

    def resize_keypoints(self, keypoints: list) -> list:
        return [
            int(v / self._ratio) if i % 4 <= 1 else v
            for i, v in enumerate(keypoints)
        ]


class V2Task(BaseTask):
    _api_path = None
    _api_body = None
    result = None
    _resize_helper = None

    def __init__(
        self,
        api_path: str,
        api_body: dict = None,
        resize_helper: ResizeHelper = None,
    ):
        super().__init__()
        self._api_path = api_path
        self._api_body = api_body
        self._resize_helper = resize_helper

    @property
    def api_path(self):
        return self._api_path

    @property
    def api_body(self):
        return self._api_body or {}

    def format_result(self, result: dict) -> dict:
        if self._resize_helper:
            return self._resize_helper.format_result(result)
        else:
            return result

    @property
    def result(self):
        return self._result

    @property
    def api_trigger_url(self):
        if self.config.endpoint.startswith("http"):
            return f"{self.config.endpoint}{self.api_path}"
        else:
            return f"https://{self.config.endpoint}{self.api_path}"

    @property
    def api_check_url(self):
        if self.config.endpoint.startswith("http"):
            return f"{self.config.endpoint}/v2/task_status/{self.task_uuid}"
        else:
            return f"https://{self.config.endpoint}/v2/task_status/{self.task_uuid}"


def create_task_with_local_image_auto_resize(
    api_path: str,
    api_body_without_image: Dict[str, Any],
    image_path: str,
    max_size: int = None,
) -> V2Task:
    api_body = api_body_without_image or {}

    if ResizeHelper.is_resizable(api_body):
        max_size = max_size or ResizeHelper.image_max_size(api_path)
        image_data, resize_info = resize_image(image_path, max_size)
        resize_helper = ResizeHelper(**resize_info)
        api_body['image'] = image_to_base64(image_data)
    else:
        api_body['image'] = image_to_base64(image_path)
        resize_helper = None

    return V2Task(api_path, api_body, resize_helper)
