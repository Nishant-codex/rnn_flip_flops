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
from plot_utils import plot_fps

sys.path.insert(0,'/home/joshi/fixed_point_edits')
import matplotlib.pyplot as plt
np.random.seed(400)
# import numpy as np
import time
from AdaptiveGradNormClip import AdaptiveGradNormClip
from AdaptiveLearningRate import AdaptiveLearningRate
from FixedPointStore import FixedPointStore

# fps = FixedPointStore(num_inits = 1,
# 					num_states= 1,
# 					num_inputs = 1	)
# dict_d = fps.restore('/home/joshi/fixed_point_edits/fps_saver/fixedPoint_unique.p')
# fps.__dict__ = dict_d

# plot_fps(fps,
# 		hid,
# 		plot_batch_idx=range(30),
# 	    plot_start_time=10)
from FlipFlop import FlipFlop
# from FlipFlop_GPU import FlipFlop_GPU
from FixedPointSearch import *



#------------------------------Create Dataset-------------------------------------------------

def create_data(rnn_object, save=True):
	rnn_object.data_batches(80000, 8, save=save)
	print('created the dataset')

#-----------------------------Train and visualize--------------------------------------------------
def train_network(rnn_object):
	rnn_object.train_network(save=True)

def reload_from_chkpt(rnn_object, chkpt, plot=False):
	chkpt = chkpt 
	# print(chkpt)
	lis = rnn_object.reload_from_checkpoints(chkpt)

	if plot:

		plt.plot(lis['predictions'][:500,0])
		plt.plot(np.reshape(lis['truth'],[-1,3])[:500,0])
		plt.savefig(chkpt+'/outputs.png')
		plt.close('all')
	return lis

#--------------------------------------Finding the fixed points----------------------------------------#
def fixed_points(rnn_object, lis, path):
	n_bits = 3
	inputs = np.zeros([1,n_bits])
	fps = FixedPointSearch(rnn_object.c_type,
							lis['hiddens'],
							path, 
							cell=rnn_object.graph['cell'],
							sess=rnn_object.sess)
	fps.sample_states(1000,lis['hiddens'],'GRU',0.5)
	fps.find_fixed_points(inputs, save = True)

def main():
	n_bits = 3

	rnn_object = FlipFlop()

	hps_array = rnn_object.hps_array()

	for i in range(len(hps_array)):
		
		if rnn_object.is_root or i==0:
			print('the hps are ',hps_array[i])

		# if hps_array[i]['arch'] =='Vanilla' or hps_array[i]['arch'] =='UGRNN':
		# 	continue

		# elif hps_array[i]['arch'] == 'GRU' and hps_array[i]['activ'] == 'tanh':
		# 	continue

		# else:
		rnn_object.c_type = hps_array[i]['arch']
		rnn_object.activation = hps_array[i]['activ']
		rnn_object.state_size = hps_array[i]['units']
		rnn_object.l2_loss = hps_array[i]['l2_norm']

		# create_data(rnn_object)

		# train_network(rnn_object)
		# pathlist = os.listdir(os.getcwd()+'/trained')
		rnn_object.get_path_for_saving()
		chkptpath =  rnn_object.path
		lis = reload_from_chkpt(rnn_object, chkptpath, plot=True)

		fixed_points(rnn_object, lis , chkptpath)


if __name__ == "__main__":
	main()

