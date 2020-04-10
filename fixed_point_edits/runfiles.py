#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 19:51:43 2020

@author: joshi
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os

import tensorflow as tf
import sys 

# sys.path.insert(0,'/home/joshi/fixed_point_edits')
sys.path.insert(0,'/home/joshi/fixed_point_edits')
# %tensorflow_version 1.x magic
import matplotlib.pyplot as plt
#import numpy.random as nrand
np.random.seed(400)
# import numpy as np
#import time
from AdaptiveGradNormClip import AdaptiveGradNormClip
from AdaptiveLearningRate import AdaptiveLearningRate

from FlipFlop import FlipFlop
from FlipFlop_GPU import FlipFlop_GPU
from FixedPointSearch import *


n_bits = 3

rnn_object = FlipFlop(c_type='GRU')

rnn_object.data_batches(80000, 8, save=True)
print('created the dataset')

rnn_object.train_network(save=True)

lis = rnn_object.reload_from_checkpoints(os.getcwd()+'/GRU_hps(64_0.01)/')

plt.plot(lis['predictions'][:500,0])
plt.plot(np.reshape(lis['truth'],[-1,3])[:500,0])

plt.show()
plt.savefig('outputs.png')

inputs = np.zeros([1,n_bits])
fps = FixedPointSearch('GRU',
						lis['hiddens'],
						cell=rnn_object.graph['cell'],
						sess=rnn_object.sess)
sample = fps.sample_states(1000,lis['hiddens'],'GRU',0.04)
fps_unique, fps_all = fps.find_fixed_points(inputs, save = True)

