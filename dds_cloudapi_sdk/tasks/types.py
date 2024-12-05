import enum
from typing import Tuple

import pydantic

# Format: x, y, visible, score
PoseKeypoints = Tuple[
    float, float, int, float,  # nose
    float, float, int, float,  # l-eye
    float, float, int, float,  # r-eye
    float, float, int, float,  # l-ear
    float, float, int, float,  # r-ear
    float, float, int, float,  # l-shoulder
    float, float, int, float,  # r-shoulder
    float, float, int, float,  # l-elbow
    float, float, int, float,  # r-elbow
    float, float, int, float,  # l-wrist
    float, float, int, float,  # r-wrist
    float, float, int, float,  # l-hip
    float, float, int, float,  # r-hip
    float, float, int, float,  # l-knee
    float, float, int, float,  # r-knee
    float, float, int, float,  # l-ankle
    float, float, int, float,  # r-ankle
]

# Format: x, y, visible, score
HandKeypoints = Tuple[
    float, float, int, float,  # wrist
    float, float, int, float,  # thumb-1
    float, float, int, float,  # thumb-2
    float, float, int, float,  # thumb-3
    float, float, int, float,  # thumb-4
    float, float, int, float,  # forefinger-1
    float, float, int, float,  # forefinger-2
    float, float, int, float,  # forefinger-3
    float, float, int, float,  # forefinger-4
    float, float, int, float,  # middle-finger-1
    float, float, int, float,  # middle-finger-2
    float, float, int, float,  # middle-finger-3
    float, float, int, float,  # middle-finger-4
    float, float, int, float,  # ring-finger-1
    float, float, int, float,  # ring-finger-2
    float, float, int, float,  # ring-finger-3
    float, float, int, float,  # ring-finger-4
    float, float, int, float,  # pinky-finger-1
    float, float, int, float,  # pinky-finger-2
    float, float, int, float,  # pinky-finger-3
    float, float, int, float,  # pinky-finger-4
]
# Format: xmin, ymin, xmax, ymax
BBox = Tuple[float, float, float, float]


class ObjectMask(pydantic.BaseModel):
    """
    The object mask format as RLE.

    :param counts: the compressed mask array in RLE format
    :param size: the 2d size of the array, (h, w)
    """

    counts: str
    size: Tuple[int, int]


class DetectionTarget(enum.Enum):
    BBox = "bbox"
    Mask = "mask"
    Hand = "hand"
    Pose = "pose"
