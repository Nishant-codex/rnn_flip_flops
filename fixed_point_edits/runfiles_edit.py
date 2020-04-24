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
from FixedPointStore import *
import tensorflow as tf
import sys 
from plot_utils import plot_fps

sys.path.insert(0,'/home/joshi/fixed_point_edits')
import matplotlib.pyplot as plt
np.random.seed(200)
# import numpy as np
import time
from AdaptiveGradNormClip import AdaptiveGradNormClip
from AdaptiveLearningRate import AdaptiveLearningRate
from FixedPointStore import FixedPointStore

import pickle 
# from FlipFlop import FlipFlop
# from FlipFlop_GPU import FlipFlopGPU
# from FixedPointSearch import *



#------------------------------Create Dataset-------------------------------------------------

def create_data(rnn_object, save=True):
	if save == True:
		rnn_object.data_batches(80000, 32, save=True)
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
#------------------------------------HYPS_______________________________________________	
def hps_array():
  arch  = ['Vanilla', 'UGRNN', 'GRU', 'LSTM']
  activ = ['tanh' , 'relu']
  units = [64, 128, 256 ]
  l2_norm = [1e-5, 1e-4, 1e-3, 1e-2]
  hps_dict = {'arch':arch,
              'activ':activ,
              'units':units,
              'l2_norm':l2_norm}

  hps_array = []
  adp_lr = {'initial_rate': 1.0, 'min_rate': 1e-5}
  for i in arch:
    for j in activ:
      for k in units:
        for l in l2_norm:
          hps_array.append({'arch':i,
                            'activ':j,
                            'units':k,
                            'l2_norm':l,
                            'al_hps':adp_lr})
  return hps_array  

#--------------------------------------Finding the fixed points----------------------------------------#
def save_hidden_states(hid,path):

    print(path)
    dir_name = path+'/fps_saver/' 
    if not os.path.exists(os.path.dirname(dir_name)):
      os.makedirs(os.path.dirname(dir_name))
    
    filename  = dir_name+'hiddens_'+'.p'   
    f =  open(filename,'wb')
    # print(self.__dict__)
    pickle.dump(hid,f)
    f.close()
    print('saved')

def restore(path):
	file = open(path,'rb')
	restore_data = file.read()
	file.close()
	# print(type(pickle.loads(restore_data)))
	# print((self.__dict__))
	hid= pickle.loads(restore_data,encoding='latin1')
	return(hid)

def fixed_points(rnn_object):
	n_bits = 3
	inputs = np.zeros([1,n_bits])

	# pathlist = os.listdir(os.getcwd()+'/trained')
	rnn_object.get_path_for_saving()
	chkptpath =  rnn_object.path
	lis = reload_from_chkpt(rnn_object, chkptpath, plot=True)
	save_hidden_states(lis['hiddens'], rnn_object.path)
	fps = FixedPointSearch(rnn_object.c_type,
							lis['hiddens'],
							rnn_object.path, 
							cell=rnn_object.graph['cell'],
							sess=rnn_object.sess)
	fps.sample_states(1024,lis['hiddens'],rnn_object.c_type,0.5)
	fps.find_fixed_points(inputs, save = True)

def plot_fps():
	hid = restore('/home/joshi/fixed_point_edits/Vanilla_hps_states_64_l2_1e-05_tanh/fps_saver/hiddens_.p')

	fps = FixedPointStore(num_inits = 1,
					num_states= 1,
					num_inputs = 1)
	dict_d = fps.restore('/home/joshi/fixed_point_edits/Vanilla_hps_states_64_l2_1e-05_tanh/fps_saver/fixedPoint_unique.p')
	# print(dict_d)
	fps.__dict__ = dict_d

	plot_fps(fps,
			hid,
			plot_batch_idx=range(30),
		    plot_start_time=10)
	




def main():
	n_bits = 3



	hps_ = hps_array()
	for i in range(len(hps_)):

		if hps_[i]['arch'] =='LSTM':		

			rnn_object = FlipFlop(opt ='momentum',**hps_[i])

			train_network(rnn_object)
			





if __name__ == "__main__":
	main()

