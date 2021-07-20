import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy import signal
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

# To enable logging, e.g. for warning and errors
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Supress warnings, usually they just cloud jupyter-notebook, we can access all warnings using logging.
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, notebook_path + "./python_scripts")
import nanolyse as nl

from CollectData import *