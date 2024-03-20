from enum import Enum
from typing import List
from typing import Union

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
    category_id: str = None  #: the category id of the rects


class BatchRectPrompt(pydantic.BaseModel):
    """
    A batch of rectangle prompts.

    :param rects: a list of rect locations in [[[upper_left_x, upper_left_y, lower_right_x, lower_right_y], ...]
    :param image: the image url the rectangle prompts are acting on
    :param category_id: the category id of the rects
    """

    rects: List[List[float]]  #: a list of rect locations in [[[upper_left_x, upper_left_y, lower_right_x, lower_right_y], ...]

    image: str = None  #: the image url the rectangle prompts are acting on, if not provided, the infer image url in context will be used
    category_id: str = None  #: the category id of the rects


class BatchInfer(pydantic.BaseModel):
    """
    A batch of inferring images with prompts.

    :param image: the image url to be inferred on
    :param prompt_type: the type of the prompts, either :member:`Point <dds_cloudapi_sdk.tasks.prompt.PromptType.Point>` or :member:`Rect <dds_cloudapi_sdk.tasks.prompt.PromptType.Rect>`
    :param prompts: either a list of :class:`BatchPointPrompt <dds_cloudapi_sdk.tasks.prompt.BatchPointPrompt>` or a list of :class:`BatchRectPrompt <dds_cloudapi_sdk.tasks.prompt.BatchRectPrompt>`, this must match the prompt_type
    """

    image: str  #: the image url to be inferred on
    prompt_type: PromptType  #: the type of the prompts, either :member:`Point <dds_cloudapi_sdk.tasks.prompt.PromptType.Point>` or :member:`Rect <dds_cloudapi_sdk.tasks.prompt.PromptType.Rect>`
    prompts: Union[List[BatchPointPrompt], List[BatchRectPrompt]]  # either a list of :class:`BatchPointPrompt <dds_cloudapi_sdk.tasks.prompt.BatchPointPrompt>` or a list of :class:`BatchRectPrompt <dds_cloudapi_sdk.tasks.prompt.BatchRectPrompt>`, this must match the prompt_type
