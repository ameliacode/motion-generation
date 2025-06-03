import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.getcwd()))
from utils.AnimationPlot import animation_plot

test = np.load('../data/data_cmu.npz')
animation_plot([test["clips"][:1].swapaxes(1,2)])
