#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 19:51:43 2020

@author: joshi
"""



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import seaborn as sns 
import numpy as np
import os
from FixedPointStore import *
import tensorflow as tf
import sys 

from FlipFlop import FlipFlop
from FixedPointSearch import *
import tf_utils

sys.path.insert(0,'/home/joshi/fixed_point_edits')
import matplotlib.pyplot as plt
np.random.seed(200)

import time
from AdaptiveGradNormClip import AdaptiveGradNormClip
from AdaptiveLearningRate import AdaptiveLearningRate
from FixedPointStore import FixedPointStore

import pickle 

import cca_core 

import numpy as np

from matplotlib import pyplot as plt

from matplotlib.collections import LineCollection

# from plot_utils import plot_fps
# from sklearn import manifold
# from sklearn.metrics import euclidean_distances
# from sklearn.decomposition import PCA


#------------------------------Create Dataset-------------------------------------------------

def create_data(rnn_object, save=True):
	if save == True:
		rnn_object.data_batches(20000, 32, save=True)
	print('created the dataset')

#-----------------------------Train and visualize--------------------------------------------------
def train_network(rnn_object):
	rnn_object.train_network(20000,save=True)

def reload_from_chkpt(rnn_object, chkpt, plot=False):
	chkpt = chkpt 
	# print(chkpt)
	lis = rnn_object.reload_from_checkpoints(chkpt)

	if plot:

		plt.plot(lis['predictions'][:500,0])
		plt.plot(np.reshape(lis['truth'],[-1,3])[:500,0])
		plt.plot(np.reshape(lis['inputs'],[-1,3])[:500,0])
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
  # adp_lr = {'initial_rate': 1.0, 'min_rate': 1e-5}
  for i in arch:
    for j in activ:
      for k in units:
        for l in l2_norm:
          hps_array.append({'arch':i,
                            'activ':j,
                            'units':k,
                            'l2_norm':l,
                            'seed':400})
  return hps_array  

#--------------------------------------Finding the fixed points----------------------------------------#
def save_hidden_states(hid,path):

    # print(path)
    dir_name = path+'/fps_saver/' 
    try:
	    if not os.path.exists(os.path.dirname(dir_name)):
	      os.makedirs(os.path.dirname(dir_name))
    except:
    	print('path exists')
    filename  = dir_name+'hiddens_'+'.p'   
    f =  open(filename,'wb')
    # print(self.__dict__)
    pickle.dump(hid,f)
    f.close()
    # print('saved')

def restore(path):
	file = open(path,'rb')
	restore_data = file.read()
	file.close()
	# print(type(pickle.loads(restore_data)))
	# print((self.__dict__))
	hid= pickle.loads(restore_data,encoding='latin1')
	return(hid)

def fixed_points(rnn_object, hps, chkpt = None):
	n_bits = 3
	inputs = np.zeros([1,n_bits])

	# pathlist = os.listdir(os.getcwd()+'/trained')

	if chkpt is not None:
		chkptpath = chkpt
		print(chkptpath)
	else:	
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

def order_fps(rnn):

	trans = rnn
	temp1 = []
	temp2 = []
	temp3 = []
	temp4 = []
	ind  = []
	
	for i in range(trans.shape[0]):
	  if np.count_nonzero(trans[i]) ==1:
	    temp1.append(trans[i])
	  elif np.count_nonzero(trans[i])==2:
	    temp2.append(trans[i])
	  elif np.count_nonzero(trans[i])==4:
	    temp3.append(trans[i])
	  else:
	    temp4.append(trans[i])
	glf = np.concatenate((np.array(temp1),np.array(temp2),np.array(temp3),np.array(temp4)))
	for i in range(glf.shape[1]):
	  if np.count_nonzero(glf[i]) ==1:
	    ind.append(glf[:,np.argmax(glf[i])])
	y = glf.shape[0]-len(ind)
	final = np.concatenate((np.array(ind).T,np.zeros((glf.shape[0],y))),axis=1)
	return final

def MDS():
	def save_array(distance_matrix):

	    filename  = os.getcwd()+'/distance_matrix'+'.p'   
	    f =  open(filename,'wb')
	    pickle.dump(distance_matrix,f)
	    f.close()

	hps1 = hps_array()
	hps2 = hps_array()
	distance_matrix = np.zeros((len(hps1),len(hps1)))


	for i in range(len(hps1)):
		for j in range(len(hps2)):

			rnn_object1 = FlipFlop(opt ='momentum',**hps1[i])
			rnn_object2 = FlipFlop(opt ='momentum',**hps2[j])
			
			rnn_object1.get_path_for_saving()
			rnn_object2.get_path_for_saving()
			print(hps1[i])
			print(hps2[j])
			path1 =  rnn_object1.path+'/fps_saver/'+'transit'+'.p'
			path2 =  rnn_object2.path+'/fps_saver/'+'transit'+'.p'

			trans1 = restore(path1)
			trans2 = restore(path2)

			if(trans1.shape[0]==27 and trans2.shape[0]==27):
				trans1 = order_fps(trans1)
				trans2 = order_fps(trans2)
				distance_matrix[i,j] = np.linalg.norm(trans1-trans2)
	save_array(distance_matrix)



def transition_graph(rnn,plot=False, save=True):
	def save_transitions(trans,path):

	    dir_name = path+'/fps_saver/' 
	    try:
		    if not os.path.exists(os.path.dirname(dir_name)):
		      os.makedirs(os.path.dirname(dir_name))
	    except:
	    	print('path exists')
	    filename  = dir_name+'transit'+'.p'   
	    f =  open(filename,'wb')
	    pickle.dump(trans,f)
	    f.close()
	    print('saved')

	rnn.get_path_for_saving()
	path =  rnn.path
	print(path)    
	lis = rnn.reload_from_checkpoints(path)
	fps = FixedPointStore(num_inits = 1,
	        num_states= 1,
	        num_inputs = 1)
	dict_d = fps.restore(rnn.path+'/fps_saver/fixedPoint_unique.p')
	fps.__dict__ = dict_d
	trans = np.zeros([fps.num_inits,fps.num_inits])

	# for k in range(fps.num_inits):
	for y in range(20):
	  fixed_points = fps.xstar

	  fps_w_noise = fps.xstar+1e-01*np.random.randn(fixed_points.shape[0],fixed_points.shape[1])

	  init_state = tf.convert_to_tensor(fps_w_noise, dtype=tf.float32)

	  x0 = np.zeros_like(rnn.test_data['inputs'])
	  x = tf.placeholder(tf.float32, [fps.num_inits, rnn.time, rnn.bits], name='input_placeholder')

	  if rnn.c_type =='LSTM':
	    init_state = tf_utils.convert_to_LSTMStateTuple(init_state)
	    rnn_outs= rnn.unroll_LSTM(rnn.cell, x,initial_state=init_state)
	  else:
	    rnn_outs,_= tf.nn.dynamic_rnn(rnn.cell,x,initial_state=init_state)

	  zero_inpt = np.zeros((fps.num_inits, rnn.time, rnn.bits))
	  hids = rnn.sess.run(rnn_outs, feed_dict={x:zero_inpt})
	  
	  if rnn.c_type =='LSTM':
	    hids = tf_utils.convert_from_LSTMStateTuple(hids)
	  for l in range(fps.num_inits):
	    for j in range(fps.num_inits):
	      index = slice(j,j+1)
	      inits_ = fixed_points[index]

	      if np.linalg.norm(inits_-hids[l,99,:])<0.01:
	        trans[l,j]+=1

	if plot:
		sns.heatmap(trans,cmap='rainbow')

	if save:

		save_transitions(trans,path)


def plot_fp():
	hid = restore('/home/joshi/fixed_point_edits/fps_saver/hiddens_.p')
	# plt.plot(hid[:,0,:])
	# plt.savefig('hid.png')
	# print("hidden shape",hid.shape)
	fps = FixedPointStore(num_inits = 1,
					num_states= 1,
					num_inputs = 1)
	dict_d = fps.restore('/home/joshi/fixed_point_edits/fps_saver/fixedPoint_unique.p')
	# print(dict_d)
	fps.__dict__ = dict_d
	print(fps.xstar[0].shape)
	plot_fps(fps,
			hid,
			plot_batch_idx=range(30),
		    plot_start_time=10)

def svcca():

	def save_array(svcca_matrix):

	    filename  = os.getcwd()+'/svcca_matrix'+'.p'   
	    f =  open(filename,'wb')
	    pickle.dump(svcca_matrix,f)
	    f.close()


	hps1 = hps_array()
	hps2 = hps_array()
	svcca_matrix = np.zeros((len(hps1),len(hps1)))
	m = 0

	for i in range(len(hps1)):
		for j in range(len(hps2)):

			rnn_object1 = FlipFlop(opt ='momentum',**hps1[i])
			rnn_object2 = FlipFlop(opt ='momentum',**hps2[j])	

			rnn_object1.get_path_for_saving()
			path1 =  rnn_object1.path
			lis1 = rnn_object1.reload_from_checkpoints(path1)

			rnn_object2.get_path_for_saving()
			path2 =  rnn_object2.path
			lis2 = rnn_object2.reload_from_checkpoints(path2)

			fps1 = FixedPointSearch(rnn_object1.c_type,
									lis1['hiddens'],
									rnn_object1.path, 
									cell=rnn_object1.graph['cell'],
									sess=rnn_object1.sess)
			fps1.sample_states(1000,lis1['hiddens'],rnn_object1.c_type,0.0)
			if rnn_object1.c_type == 'LSTM':
				sample1 = fps1.convert_from_lstm_tuples(fps1.sampled_states)
			else:
				sample1 = fps1.sampled_states

			fps2 = FixedPointSearch(rnn_object2.c_type,
									lis2['hiddens'],
									rnn_object2.path, 
									cell=rnn_object2.graph['cell'],
									sess=rnn_object2.sess)
			fps2.sample_states(1000,lis2['hiddens'],rnn_object2.c_type,0.0)
			if rnn_object2.c_type == 'LSTM':
				sample2 = fps2.convert_from_lstm_tuples(fps2.sampled_states)
			else:
				sample2 = fps2.sampled_states

			results = cca_core.robust_cca_similarity(sample1.T, sample2.T, epsilon=1e-10,compute_dirns=False)
			mean = np.mean(results["cca_coef1"])

			svcca_matrix[i,j] = mean
			print('running iter ',m)
			m+=1
	save_array(svcca_matrix)


def main():

	# hps_ = hps_array()
	# for i in range(len(hps_)):

	# 	if hps_[i]['arch'] =='LSTM' and hps_[i]['activ'] == 'tanh' and hps_[i]['units'] == 128 and hps_[i]['l2_norm'] ==1e-2:		

	# 		rnn_object = FlipFlop(opt ='momentum',**hps_[i])
			

			# train_network(rnn_object)
			
			# fixed_points(rnn_object, hps_[i])
			# transition_graph(rnn_object)

	# MDS()
	svcca()			
			

if __name__ == "__main__":
	main()

