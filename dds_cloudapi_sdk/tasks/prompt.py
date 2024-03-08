from typing import List

import pydantic


class TextPrompt(pydantic.BaseModel):
    text: str
    is_positive: bool = True

    type: str = "text"


class RectPrompt(pydantic.BaseModel):
    rect: List[float]
    is_positive: bool = True

    type: str = "rect"
