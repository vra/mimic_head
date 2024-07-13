# coding: utf-8

"""
parameters used for crop faces
"""

from dataclasses import dataclass
from .base_config import PrintableConfig


@dataclass(repr=False)  # use repr from PrintableConfig
class CropConfig(PrintableConfig):
    dsize: int = 512  # crop size
    scale: float = 2.3  # scale factor
    vx_ratio: float = 0  # vx ratio
    vy_ratio: float = -0.125  # vy ratio +up, -down
