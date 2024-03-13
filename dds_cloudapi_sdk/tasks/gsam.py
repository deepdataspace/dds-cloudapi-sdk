import enum
from typing import List

import pydantic

from dds_cloudapi_sdk.tasks.base import BaseTask
from dds_cloudapi_sdk.tasks.prompt import TextPrompt


class GSAMObject(pydantic.BaseModel):
    category: str
    score: float
    bbox: List[float]


class ModelType(enum.Enum):
    Tiny = "swint"
    Base = "swinb"


class TaskResult(pydantic.BaseModel):
    mask_url: str
    objects: List[GSAMObject]


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
        return self._result

    def format_result(self, result: dict) -> TaskResult:
        """
        Format the result of the task.
        An example of the result:
        {
            'mask_url': 'https://image.png',
             'objects': [
                    {'category': 'iron man', 'score': 0.4880097210407257, 'bbox': [653.0848388671875, 329.127685546875, 942.047119140625, 842.4909057617188]},
                    {'category': 'iron man', 'score': 0.3169572949409485, 'bbox': [481.4720153808594, 0.6030968427658081, 878.6416625976562, 650.6054077148438]}
                ]
        }
        """
        return TaskResult(**result)


class TinyGSAMTask(_GroundedSAMTask):
    def __init__(self,
                 image_url: str,
                 prompts: List[TextPrompt] = None,
                 box_threshold: float = 0.3,
                 text_threshold: float = 0.2,
                 nms_threshold: float = 0.8
                 ):
        """
        Construct a task calls grounded_sam algorithm with tiny model.
        """

        super().__init__(image_url,
                         ModelType.Tiny,
                         prompts,
                         box_threshold,
                         text_threshold,
                         nms_threshold)


class BaseGSAMTask(_GroundedSAMTask):
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
    import logging

    logging.basicConfig(level=logging.INFO)

    from dds_cloudapi_sdk import Client
    from dds_cloudapi_sdk import TextPrompt

    client = Client("dds-app-free", )
    task = TinyGSAMTask(
        "https://dds-frontend.oss-cn-shenzhen.aliyuncs.com/static_files/playground/grounded_sam/05.jpg",
        [TextPrompt(text="iron man")]
    )

    client.run_task(task)
    print(task.result)

    task = BaseGSAMTask(
        "https://dds-frontend.oss-cn-shenzhen.aliyuncs.com/static_files/playground/grounded_sam/05.jpg",
        [TextPrompt(text="iron man")]
    )

    client.run_task(task)
    print(task.result)


if __name__ == "__main__":
    test()
