from enum import Enum
from typing import List

import pydantic


class PromptType(Enum):
    Rect = "rect"
    Point = "point"
    Mask = "mask"
    Text = "text"
    Stroke = "stroke"
    Embd = "embd"


class TextPrompt(pydantic.BaseModel):
    """
    A text prompt.

    :param text: the str content of the prompt
    :param is_positive: whether the prompt is positive, default to True
    """

    text: str  #: the str content of the prompt
    is_positive: bool = True  #: whether the prompt is positive, default to True

    @property
    def type(self):
        """
        constant string 'text' for TextPrompt.
        """
        return PromptType.Text.value

    def dict(self, **kwargs):
        d = super().dict(**kwargs)
        d["type"] = self.type
        return d


class RectPrompt(pydantic.BaseModel):
    """
    A rectangle prompt.

    :param rect: the rect location in [upper_left_x, upper_left_y, lower_right_x, lower_right_y]
    :param is_positive: whether the prompt is positive, default to True
    """

    rect: List[float]  #: the rect location in [upper_left_x, upper_left_y, lower_right_x, lower_right_y]
    is_positive: bool = True  #: whether the prompt is positive, default to True

    @property
    def type(self):
        """
        constant string 'rect' for RectPrompt.
        """
        return PromptType.Rect.value

    def dict(self, **kwargs):
        d = super().dict(**kwargs)
        d["type"] = self.type
        return d


class BatchPointPrompt(pydantic.BaseModel):
    """
    A batch of point prompts.

    :param points: a list of point locations in [[x1, y1], [x2, y2]]
    :param image: the image url the point prompts are acting on, if not provided, the infer image url in context will be used
    :param category_id: the category id of the points
    """

    points: List[List[float]]  #: a list of point locations in [[x1, y1], [x2, y2]]

    image: str = None  #: the image url the point prompts are acting on
    category_id: int = None  #: the category id of the points, only required for :class:`TRexInteractiveInfer <dds_cloudapi_sdk.tasks.trex_interactive.TRexInteractiveInfer>` task.


class BatchRectPrompt(pydantic.BaseModel):
    """
    A batch of rectangle prompts.

    :param rects: a list of rect locations in [[[upper_left_x, upper_left_y, lower_right_x, lower_right_y], ...]
    :param image: the image url the rectangle prompts are acting on
    :param category_id: the category id of the rects
    """

    rects: List[List[float]]  #: a list of rect locations in [[[upper_left_x, upper_left_y, lower_right_x, lower_right_y], ...]

    image: str = None  #: the image url the rectangle prompts are acting on, if not provided, the infer image url in context will be used
    category_id: int = None  #: the category id of the rects, only required for :class:`TRexInteractiveInfer <dds_cloudapi_sdk.tasks.trex_interactive.TRexInteractiveInfer>` task.


class BatchEmbdPrompt(pydantic.BaseModel):
    """
    A batch of embd prompts.

    :param embd: the embedding file url
    :param category_id: the category id of the rects
    """

    embd: str = None  #: the embedding file url
    category_id: int  #: the category id of the objects inferred by this embedding file


class BatchPointInfer(pydantic.BaseModel):
    """
    An infer image with batch point prompts.

    :param image: the image url to be inferred on
    :param prompts: a list of :class:`BatchPointPrompt <dds_cloudapi_sdk.tasks.prompt.BatchPointPrompt>`
    """

    image: str  #: the image url to be inferred on
    prompts: List[BatchPointPrompt]  #: a list of :class:`BatchPointPrompt <dds_cloudapi_sdk.tasks.prompt.BatchPointPrompt>`


class BatchRectInfer(pydantic.BaseModel):
    """
    An infer image with batch rect prompts.

    :param image: the image url to be inferred on
    :param prompts: a list of :class:`BatchRectPrompt <dds_cloudapi_sdk.tasks.prompt.BatchRectPrompt>`
    """

    image: str  #: the image url to be inferred on
    prompts: List[BatchRectPrompt]  #: a list of :class:`BatchRectPrompt <dds_cloudapi_sdk.tasks.prompt.BatchRectPrompt>`


class BatchEmbdInfer(pydantic.BaseModel):
    """
    An infer image with batch embd prompts.

    :param image: the image url to be inferred on
    :param prompts: a list of :class:`BatchEmbdPrompt <dds_cloudapi_sdk.tasks.prompt.BatchEmbdPrompt>`
    """

    image: str  #: the image url to be inferred on
    prompts: List[BatchEmbdPrompt]  #: a list of :class:`BatchEmbdPrompt <dds_cloudapi_sdk.tasks.prompt.BatchEmbdPrompt>`
