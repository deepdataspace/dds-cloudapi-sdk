import logging
from typing import Any
from typing import Dict
from typing import Optional

from dds_cloudapi_sdk.image_resizer import image_to_base64
from dds_cloudapi_sdk.image_resizer import resize_image
from dds_cloudapi_sdk.tasks.base import BaseTask


class V2Task(BaseTask):
    _api_path = None
    _api_body = None
    result = None

    def __init__(self, api_path: str, api_body: dict = None, scale: float = None):
        super().__init__()
        self._api_path = api_path
        self._api_body = api_body
        self._scale = scale

    @property
    def api_path(self):
        return self._api_path

    @property
    def api_body(self):
        return self._api_body or {}

    def format_result(self, result: dict) -> dict:
        try:
            if not self._scale:
                return result

            for item in result['data']['result']['objects']:
                if 'bbox' in item:
                    item['bbox'] = [int(coord / self._scale) for coord in item['bbox']]
            return result
        except Exception as e:
            logging.error(f"Error formatting result: {result}, scale: {self._scale}, error: {e}")
            return result

    @property
    def result(self):
        return self._result

    @property
    def api_trigger_url(self):
        return f"https://{self.config.endpoint}{self.api_path}"

    @property
    def api_check_url(self):
        return f"https://{self.config.endpoint}/v2/task_status/{self.task_uuid}"


def create_task(
    api_path: str,
    api_body: Optional[Dict[str, Any]] = None,
    image_path: Optional[str] = None,
    max_size: int = 1536
) -> V2Task:
    """Create task with local image"""
    api_body = api_body or {}

    if image_path:
        image_data, scale = resize_image(image_path, max_size)
        api_body['image'] = image_to_base64(image_data)
    else:
        scale = None

    return V2Task(api_path, api_body, scale)
