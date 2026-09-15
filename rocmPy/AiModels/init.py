"""
Input Handlers Package for VGamepad and Xbox Controller Emulation
"""

from .FirstModel import ForzaH4Model
from .SecondModel import SecondFH4Model
from .OnlyTanhModel import OnlyTanhModel

__all__ = [
    'ForzaH4Model'
    'SecondFH4Model'
    'OnlyTanhModel'
]