# -*- coding: utf-8 -*-

r"""
The data module contains all the function to preprocess, transform and load the data
 * load : Load a dataset 
 * path : Path for the data
 * tranformation : transformation for the data
"""

from .core import * 
from . import transformation
from .gaussian_synthetic_data import *
from . import fmap

