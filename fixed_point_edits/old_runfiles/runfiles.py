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


import matplotlib.pyplot as plt
np.random.seed(400)
import time
from AdaptiveGradNormClip import AdaptiveGradNormClip
from AdaptiveLearningRate import AdaptiveLearningRate

from FlipFlop import FlipFlop
from FlipFlop_GPU import FlipFlop_GPU
from FixedPointSearch import *


n_bits = 3

rnn_object = FlipFlop()

hps_array = rnn_object.hps_array()

for i in range(len(hps_array)//2):
	
	if rnn_object.is_root:
		print('the hps are ',hps_array[i])

	rnn_object.c_type = hps_array[i]['arch']
	rnn_object.activation = hps_array[i]['activ']
	rnn_object.state_size = hps_array[i]['units']
	rnn_object.l2_loss = hps_array[i]['l2_norm']

#------------------------------Create Dataset-------------------------------------------------

	# rnn_object.data_batches(80000, 8, save=True)
	# print('created the dataset')

#-----------------------------Train and visualize--------------------------------------------------
	
	rnn_object.train_network(save=True)

	chkpt = rnn_object.savedir[i] 
	lis = rnn_object.reload_from_checkpoints(chkpt)

	plt.plot(lis['predictions'][:500,0])
	plt.plot(np.reshape(lis['truth'],[-1,3])[:500,0])

	plt.show()
	plt.savefig(chkpt+'/outputs.png')


#--------------------------------------Finding the fixed points----------------------------------------#
	# inputs = np.zeros([1,n_bits])
	# fps = FixedPointSearch('GRU',
	# 						lis['hiddens'],
	# 						cell=rnn_object.graph['cell'],
	# 						sess=rnn_object.sess)
	# sample = fps.sample_states(1000,lis['hiddens'],'GRU',0.04)
	# fps_unique, fps_all = fps.find_fixed_points(inputs, save = True)

