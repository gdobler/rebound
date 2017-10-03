#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd


# topdirectory for 2017 stacked HSI images
STACKED_HSI_FILEPATH = os.path.join(os.environ['REBOUND_DATA'], 'hsi')

# HSI stack output
HSI_OUT = os.path.join(os.environ['REBOUND_WRITE'],'hsi')

# dimensions of 2017 stacked images (after transpose: nwav, nrow, ncol)
STACKED_SHAPE = (848, 1600, 3194)

