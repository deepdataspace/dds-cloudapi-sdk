"""
The Grounded SAM algorithm integrates the capabilities of both GroundingDINO and SAM algorithms, utilizing text prompts to
efficiently detect bounding boxes and segmentation masks.

This powerful algorithm is available in DDS CloudAPI SDK in two variants:

- the "tiny" model through TinyGSAMTask
- the "base" model through BaseGSAMTask
"""

import enum
from typing import List

import pydantic

from dds_cloudapi_sdk.tasks.base import BaseTask
from dds_cloudapi_sdk.tasks.prompt import TextPrompt


class ModelType(enum.Enum):
    """
    | This enumerator represents the models the GSAM algorithm uses.
    | The :class:`TinyGSAMTask <dds_cloudapi_sdk.tasks.gsam.TinyGSAMTask>` will use the Tiny model by default.
    | And the :class:`BaseGSAMTask <dds_cloudapi_sdk.tasks.gsam.BaseGSAMTask>` will use the Base model by default.
    """

    Tiny = "swint"  #: The tiny model
    Base = "swinb"  #: The base model


class GSAMObject(pydantic.BaseModel):
    """
    The object detected by GSAM tasks.

    :param category: the category name of the object
    :param score: the predict score of the object
    :param bbox: the bbox of the object, [xmin, ymin, xmax, ymax]
    """

    category: str  #: the category name of the object
    score: float  #: the predict score of the object
    bbox: List[float]  #: the bbox of the object, [xmin, ymin, xmax, ymax]


class TaskResult(pydantic.BaseModel):
    """
    The task result of the GSAM tasks.

    :param mask_url: an image url with all objects' mask drawn on
    :param objects: a list of detected objects of :class:`GSAMObject <dds_cloudapi_sdk.tasks.gsam.GSAMObject>`
    """

    mask_url: str  #: an image url with all objects' mask drawn on
    objects: List[GSAMObject]  #: a list of detected objects of :class:`GSAMObject <dds_cloudapi_sdk.tasks.gsam.GSAMObject>`


class _GroundedSAMTask(BaseTask):

    def __init__(self,
                 image_url: str,
                 model_type: ModelType,
                 prompts: List[TextPrompt] = None,
                 box_threshold: float = 0.3,
                 text_threshold: float = 0.2,
                 nms_threshold: float = 0.8
                 ):
        """
        Construct a task calls grounded_sam algorithm.
        """

        for p in prompts:
            if not p.is_positive:
                raise ValueError(f"Only positive text prompt is permitted for GSAM task.")

        super().__init__()

        self.image_url = image_url
        self.model_type = model_type
        self.prompts = prompts
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.nms_threshold = nms_threshold

    @property
    def api_path(self):
        return "grounded_sam"

    @property
    def api_body(self):
        data = {
            "image"         : self.image_url,
            "model_type"    : self.model_type.value,
            "prompts"       : [prompt.dict() for prompt in self.prompts],
            "box_threshold" : self.box_threshold,
            "text_threshold": self.text_threshold,
            "nms_threshold" : self.nms_threshold
        }
        return data

    @property
    def result(self) -> TaskResult:
        """
        Get the formatted :class:`TaskResult <dds_cloudapi_sdk.tasks.gsam.TaskResult>` object.
        """

        return self._result

    def format_result(self, result: dict) -> TaskResult:
        """
        Format the result of the task, return an instance of :class:`GSAM TaskResult <dds_cloudapi_sdk.tasks.gsam.TaskResult>`.

        An example of input result data::

            {
                'mask_url': 'https://image.png',
                 'objects': [
                        {'category': 'iron man', 'score': 0.4880097210407257, 'bbox': [653.0848388671875, 329.127685546875, 942.047119140625, 842.4909057617188]},
                        {'category': 'iron man', 'score': 0.3169572949409485, 'bbox': [481.4720153808594, 0.6030968427658081, 878.6416625976562, 650.6054077148438]}
                    ]
            }

        :param result: the raw python dict returned by the API.
        """
        return TaskResult(**result)


class TinyGSAMTask(_GroundedSAMTask):
    """
    Trigger the Grounded-SegmentAnything algorithm with the **tiny** :class:`ModelType <dds_cloudapi_sdk.tasks.gsam.ModelType>`.

    :param image_url: the segmenting image url.
    :param prompts: a list of :class:`TextPrompt <dds_cloudapi_sdk.tasks.prompt.TextPrompt>` object. But for This task, only positive prompts are permitted.
    :param box_threshold: a threshold to filter out objects by bbox score, default to 0.3.
    :param text_threshold: a threshold to filter out objects by text score, default to 0.2.
    :param nms_threshold: a threshold for nms to filter out overlapping boxes, default to 0.8.
    """

    def __init__(self,
                 image_url: str,
                 prompts: List[TextPrompt] = None,
                 box_threshold: float = 0.3,
                 text_threshold: float = 0.2,
                 nms_threshold: float = 0.8
                 ):
        super().__init__(image_url,
                         ModelType.Tiny,
                         prompts,
                         box_threshold,
                         text_threshold,
                         nms_threshold)


class BaseGSAMTask(_GroundedSAMTask):
    """
    Trigger the Grounded-SegmentAnything algorithm with the **base** :class:`ModelType <dds_cloudapi_sdk.tasks.gsam.ModelType>`.

    :param image_url: the segmenting image url.
    :param prompts: a list of :class:`TextPrompt <dds_cloudapi_sdk.tasks.prompt.TextPrompt>` object. But for This task, only positive prompts are permitted.
    :param box_threshold: a threshold to filter out objects by bbox score, default to 0.3.
    :param text_threshold: a threshold to filter out objects by text score, default to 0.2.
    :param nms_threshold: a threshold for nms to filter out overlapping boxes, default to 0.8.
    """

    def __init__(self,
                 image_url: str,
                 prompts: List[TextPrompt] = None,
                 box_threshold: float = 0.3,
                 text_threshold: float = 0.2,
                 nms_threshold: float = 0.8
                 ):
        """
        Construct a task calls grounded_sam algorithm with base model.
        """

        super().__init__(image_url,
                         ModelType.Base,
                         prompts,
                         box_threshold,
                         text_threshold,
                         nms_threshold)


def test():
    """
    python -m dds_cloudapi_sdk.tasks.gsam
    """

    import os
    test_token = os.environ["DDS_CLOUDAPI_TEST_TOKEN"]

    import logging

    logging.basicConfig(level=logging.INFO)

    from dds_cloudapi_sdk import Config
    from dds_cloudapi_sdk import Client
    from dds_cloudapi_sdk import TextPrompt

    config = Config(test_token)
    client = Client(config)
    task = TinyGSAMTask(
        "https://algosplt.oss-cn-shenzhen.aliyuncs.com/test_files/tasks/grounded_sam/iron_man.jpg",
        [TextPrompt(text="iron man")]
    )

    client.run_task(task)
    print(task.result)

    task = BaseGSAMTask(
        "https://algosplt.oss-cn-shenzhen.aliyuncs.com/test_files/tasks/grounded_sam/iron_man.jpg",
        [TextPrompt(text="iron man")]
    )

    client.run_task(task)
    print(task.result)


if __name__ == "__main__":
    test()
