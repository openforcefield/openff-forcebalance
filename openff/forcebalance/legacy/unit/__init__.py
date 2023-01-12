"""
Physical quantities with units for dimensional analysis and automatic unit conversion.
"""
__docformat__ = "epytext en"

__author__ = "Christopher M. Bruns"
__copyright__ = "Copyright 2010, Stanford University and Christopher M. Bruns"
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Christopher M. Bruns"
__email__ = "cmbruns@stanford.edu"

from .constants import *
from .quantity import Quantity, is_quantity
from .unit import Unit, is_unit
from .unit_definitions import *
from .unit_math import *
