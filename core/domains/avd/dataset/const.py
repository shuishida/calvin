from enum import Enum


class SceneType(Enum):
    HOME = 1
    OFFICE = 2


class ImageType(Enum):
    RGB = 1
    DEPTH = 3


class AVDMove(Enum):
    forward = 'w'
    backward = 's'
    left = 'e'
    right = 'r'
    rotate_ccw = 'a'
    rotate_cw = 'd'
