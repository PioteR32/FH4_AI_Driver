"""
Input Handlers Package for VGamepad and Xbox Controller Emulation
"""

from .FirstModel import ForzaH4Model
from .SecondModel import SecondFH4Model
from .OnlyTanhModel import OnlyTanhModel
from .Resnet18Model import Resnet18Model

__all__ = [
    'ForzaH4Model'
    'SecondFH4Model'
    'OnlyTanhModel'
    'Resnet18Model'
]