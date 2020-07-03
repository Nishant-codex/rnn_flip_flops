"""
Functions used to save model data and to perform analysis
"""

import numpy as np
from parameters import *
# from sklearn import svm
import time
import pickle
import stimulus
import os
import copy
import matplotlib.pyplot as plt
from itertools import product
from scipy import signal
from scipy.optimize import curve_fit



def get_perf(target, output, mask):

    """ Calculate task accuracy by comparing the actual network output to the desired output
        only examine time points when test stimulus is on, e.g. when y[:,:,0] = 0 """

    # mask_full = np.float32(mask > 0)
    # mask_test = mask_full*(target[:,:,0]==0)
    # mask_non_match = mask_full*(target[:,:,1]==1)
    # mask_match = mask_full*(target[:,:,2]==1)
    # target_max = np.argmax(target, axis = 2)
    # output_max = np.argmax(output, axis = 2)
    # accuracy = np.sum(np.float32(target_max == output_max)*mask_test)/np.sum(mask_test)

    # accuracy_non_match = np.sum(np.float32(target_max == output_max)*np.squeeze(mask_non_match))/np.sum(mask_non_match)
    # accuracy_match = np.sum(np.float32(target_max == output_max)*np.squeeze(mask_match))/np.sum(mask_match)

    # return accuracy, accuracy_non_match, accuracy_match

    target = np.reshape(np.float32(target),[-1,3])
    output = np.reshape(np.float32(output),[-1,3])

    accuracy = np.sum(np.float32(target==output))/np.size(target)

    return accuracy
