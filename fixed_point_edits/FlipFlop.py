# -*- coding: utf-8 -*-
"""flip_flop_lstm_states

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1V9SlynnIlbvdAEHV1-D28mH45Z1c6D37
"""

import numpy as np
import os

import tensorflow as tf

import cProfile
# %tensorflow_version 1.x magic
import matplotlib.pyplot as plt
import numpy.random as nrand
# import horovod.tensorflow as hvd
np.random.seed(400)
# import numpy as np
import time
from AdaptiveGradNormClip import AdaptiveGradNormClip
from AdaptiveLearningRate import AdaptiveLearningRate

#from FixedPointSearch import * 
# if(tf.enable_eager_execution()):
#   print('true')
class FlipFlop:
#Hyperparameters
    
    hyp_dict = \
    {'time' : 500,
    'bits' : 3 ,
    'num_steps' : 6,
    'batch_size' : 200,
    'state_size' : 50 ,
    'num_classes' : 2,
    'p': 0.95,
    'learning_rate' :0.01,
    'c_type':'GRU'
    }
    '''
    Architectures: 
    Vanilla, UG-RNN, GRU, LSTM
    
    Activation: 
    Tanh, relu
    
    Num_units:
    64, 128, 256 
    
    L2 regularization: 
    1e-5, 1e-4, 1e-3, 1e-2
    '''
    def __init__(self,
        time = hyp_dict['time'],
        bits = hyp_dict['bits'],
        num_steps = hyp_dict['num_steps'],
        batch_size = hyp_dict['batch_size'],
        state_size = hyp_dict['state_size'],
        num_classes = hyp_dict['num_classes'],
        learning_rate = hyp_dict['learning_rate'],
        p = hyp_dict['p'],
        c_type = hyp_dict['c_type']):
        
        self.time = time
        self.bits = bits
        self.num_steps = num_steps
        self.num_steps =num_steps 
        self.batch_size = batch_size 
        self.state_size = state_size 
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.p = p  
        self.graph = 0
        self.c_type = c_type    
        self.sess = 0
        self.test_data = []

    def flip_flop(self, plot=False):
    
      inputs = []
      outputs= []
      for batch in range(self.batch_size):
        a = np.random.binomial(1,self.p,size=[self.bits, self.time])
        b = np.random.binomial(1,self.p,size=[self.bits, self.time])
        inp_= a-b
        last = 1
        out_ = np.ones_like(inp_)
        for i in range(self.bits):
          for m in range(self.time):
            a = inp_[i,m]
            if a !=0:
              last = a
            out_[i,m] = last
        # if(plot):
        inp_inv = inp_.T
        out_inv = out_.T
        # print(inp_[:,0])
        inputs.append(inp_inv)
        outputs.append(out_inv)
      if(plot):
        plt.plot(inputs[1][:,0])
        plt.plot(outputs[1][:,0])
        plt.xlabel("time")
        plt.ylabel("Bit 0")
        plt.show()
        plt.plot(inputs[1][:,1])
        plt.plot(outputs[1][:,1])
        plt.xlabel("time")
        plt.ylabel("Bit 1")
        plt.show()
        plt.plot(inputs[1][:,2])
        plt.plot(outputs[1][:,2])
        plt.xlabel("time")
        plt.ylabel("Bit 2")
        plt.show()
      return({'inputs':inputs,'outputs':outputs})
    
    def data_batches(self,num_batches):
      batch_list = []
      t1 = time.time()
      total = 0
      for i in range(num_batches):
        t2 = time.time()-t1
        total +=t2
        if i ==0:
          print('time to generate 1 batch %f', t2)
        t1 = t2
        batch_list.append(self.flip_flop())
      print('total time is %f', total)
      return batch_list
  
    def reset_graph(self):
        if 'sess' in globals() and self.sess:
            self.sess.close()
        tf.reset_default_graph()
    
    def setup_model(self):
      
      self.reset_graph()
      x = tf.placeholder(tf.float32, [self.batch_size, self.time, self.bits], name='input_placeholder')
      y = tf.placeholder(tf.float32, [self.batch_size, self.time, self.bits], name='labels_placeholder')
    
      if self.c_type == 'Vanilla':
        cell = tf.contrib.rnn.BasicRNNCell(self.state_size,reuse=tf.AUTO_REUSE)
    
      if self.c_type == 'GRU':
        cell = tf.contrib.rnn.GRUCell(self.state_size,reuse=tf.AUTO_REUSE)
    
      if self.c_type == 'LSTM':
        cell = tf.contrib.rnn.LSTMCell(self.state_size,reuse=tf.AUTO_REUSE,state_is_tuple=True)
    
      init_state = cell.zero_state(self.batch_size, dtype=tf.float32)
    
      rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, x, initial_state=init_state,)
    
      """
      rnn_outputs gives out rnn hidden states ht which is of the size [batch_size, timestep, state_size]
    
      """
#      self.cell = cell
      with tf.variable_scope('losses',reuse=tf.AUTO_REUSE):
          W = tf.get_variable('W', [self.state_size , self.bits])
          b = tf.get_variable('b', [self.bits], initializer=tf.constant_initializer(0.0))
      # plt.plot(W)
      rnn_outputs_ = tf.reshape(rnn_outputs, [-1, self.state_size])
    
      logits = tf.tensordot(rnn_outputs_,W,axes=1) + b
      # predictions = tf.nn.softmax(logits)
    
      y_as_list =tf.reshape(y, [-1, self.bits]) #shape is flattened_tensor x bits
      print(y_as_list.shape)
      losses = tf.squared_difference(logits,y_as_list)
      total_loss = tf.reduce_mean(losses)
      train_step = tf.train.AdagradOptimizer(self.learning_rate).minimize(total_loss)
      return {'losses':total_loss, 'train_step':train_step, 
              'hiddens':rnn_outputs,'finalstate':final_state , 
              'X':x, 'Y':y, 'predict':logits, 'init_state':init_state , 
              'saver' : tf.train.Saver(),'cell':cell, 'weights':W }
    
    
    
    def train_network(self, num_epochs, verbose=True, save=True):
      num_epochs = num_epochs
      act = self.setup_model()
      data = self.data_batches(num_epochs)
      epochs = 0
      training_loss = 0
      hidden = []
      # path = input('What should be the name of the file?')
      # writer = tf.summary.FileWriter('./graphs')
      saver = tf.train.Saver()
      path = self.c_type+str(num_epochs)
    
      with tf.Session() as sess:
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())
        training_losses = []
        for i in range(len(data)):
          ground_truth = data[i]['outputs']
          tr_losses, training_step_, training_state, outputs, predict = \
                      self.sess.run([act['losses'],
                                act['train_step'],
                                act['finalstate'],
                                act['hiddens'],
                                act['predict']],
                                    feed_dict={act['X']:data[i]['inputs'], act['Y']:data[i]['outputs']})
          training_loss += tr_losses
    
          if verbose:
              print("Average training loss for Epoch", epochs, ":", training_loss/100)
          training_losses.append(training_loss)
          training_loss = 0
          if epochs == 0:
            hidden.append(outputs)
          epochs +=1
        hidden.append(outputs)
        if(save):
          saver.save(self.sess, path)
      return {'losses':training_losses, 'hidden':hidden, 'predictions':predict, 'truth':ground_truth, 'training':training_state}

    def lstm_hiddens(self, graph):

      n_hidden = self.state_size
      [self.batch_size, self.time, self.bits] = np.array(self.test_data['inputs']).shape
      initial_state = graph['cell'].zero_state(a.batch_size, dtype=tf.float32)

      ''' Add ops to the graph for getting the complete LSTM state
      (i.e., hidden and cell) at every timestep.'''
      full_state_list = []
      # cur_state_min_one = 0
      for t in range(self.time):
          input_ = graph['X'][:,t,:]
          if t == 0:
              cur_state_min_one = initial_state
          else:
              cur_state_min_one = full_state_list[-1]

          _, states = graph['cell'](input_,cur_state_min_one)
          full_state_list.append(states)
      
      print(states.c)

      '''Evaluate those ops'''
      ops_to_eval = [full_state_list]
      feed_dict = {graph['X']: self.test_data['inputs']}
      ev_full_state_list= \
          self.sess.run(ops_to_eval, feed_dict=feed_dict)

      '''Package the results'''
      h = np.zeros([self.batch_size, self.time, self.state_size]) # hidden states: bxtxd
      c = np.zeros([self.batch_size, self.time, self.state_size]) # cell states: bxtxd
      for t in range(self.time):
          h[:,t,:] = ev_full_state_list[0][t].h
          c[:,t,:] = ev_full_state_list[0][t].c

      ev_LSTMCellState = tf.nn.rnn_cell.LSTMStateTuple(h=h, c=c)
      return(ev_LSTMCellState)


    def reload_from_checkpoints(self, chkpt):
      # Uncomment to run a saved network
      # Need to close the session manually 
      graph = self.setup_model()
      self.graph = graph
      saver = graph['saver']
      state = None
      self.test_data = self.flip_flop()
      feed_dict = {graph['X']:self.test_data['inputs'],graph['Y']:self.test_data['outputs']}
      self.sess = tf.Session()
      self.sess.run(tf.global_variables_initializer())
      saver.restore(self.sess, chkpt)
      hiddens, outputs, losses = self.sess.run([graph['hiddens'],graph['predict'],graph['losses']],feed_dict=feed_dict)
      
      if(self.c_type=='LSTM'):
        return({'hiddens':self.lstm_hiddens(graph),'predictions':outputs,'loss':losses})

      else:
        return({'hiddens':hiddens,'predictions':outputs,'loss':losses})


