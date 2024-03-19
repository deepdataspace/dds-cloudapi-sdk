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

    :param image: the image url the point prompts are acting on
    :param points: a list of point locations in [[x1, y1], [x2, y2]]
    """

    image: str  #: the image url the point prompts are acting on
    points: List[List[float]]  #: a list of point locations in [[x1, y1], [x2, y2]]


class BatchRectPrompt(pydantic.BaseModel):
    """
    A batch of rectangle prompts.

    :param image: the image url the rectangle prompts are acting on
    :param rects: a list of rect locations in [[[upper_left_x, upper_left_y, lower_right_x, lower_right_y], ...]
    """

    image: str  #: the image url the rectangle prompts are acting on
    rects: List[List[float]]  #: a list of rect locations in [[[upper_left_x, upper_left_y, lower_right_x, lower_right_y], ...]
