from typing import List

import pydantic


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
        return "text"

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
        return "rect"

    def dict(self, **kwargs):
        d = super().dict(**kwargs)
        d["type"] = self.type
        return d
