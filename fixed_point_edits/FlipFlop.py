
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy.random as nrand
import horovod.tensorflow as hvd
from hpc4neuro.errors import MpiInitError
from hpc4neuro.distribution import DataDistributor
import mpi4py
# from data_utils import DataValidator

import time
import datetime as dt 
from AdaptiveGradNormClip import AdaptiveGradNormClip
from AdaptiveLearningRate import AdaptiveLearningRate

class FlipFlop:
#Hyperparameters
    
    hyp_dict = \
    {'time' : 100,
    'bits' : 3 ,
    'num_steps' : 6,
    'batch_size' : 64,
    'state_size' : 64 ,
    'num_classes' : 2,
    'p': 0.95,
    'learning_rate' :0.01,
    'c_type':'GRU',
    'l2_loss':0.01

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
        c_type = hyp_dict['c_type'],
        _seed = 200,
        l2_loss = hyp_dict['l2_loss'],
        opt = 'momentum',
        **hps):
      
        self.seed = hps['seed']
        self.time = time
        self.bits = bits

        # if hps is not None:
        self.activation = hps['activ']
        self.l2_loss = hps['l2_norm']
        self.state_size = hps['units'] 
        self.c_type = hps['arch']  

        self.num_steps = num_steps
        self.hps = hps
        self.batch_size = batch_size 
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.p = p 
        self.opt = opt 
        self.dtype = tf.float32
        self.graph = 0
        self.grad_norm_clip_val = None
        self.sess = 0
        self.is_root = False
        self.cell = None
        self.path = None
        self.learning_rate = None
        self.test_data = []
        self.merged_summary_op = 0
        self.global_step = None
        self.grad_global_norm = None
        self.logdir = 0
        self.savedir = []
        self.data = None

        np.random.seed(self.seed)

  

    def flip_flop(self, plot=False):
    
      unsigned_inp = np.random.binomial(1,0.2,[self.batch_size,self.time,self.bits])
      unsigned_out = 2*np.random.binomial(1,0.5,[self.batch_size,self.time,self.bits]) -1 



      inputs = np.multiply(unsigned_inp,unsigned_out)
      inputs[:,0,:] = 1
      output = np.zeros_like(inputs)
      for trial_idx in range(self.batch_size):
          for bit_idx in range(self.bits):
              input_ = np.squeeze(inputs[trial_idx,:, bit_idx])
              t_flip = np.where(input_ != 0)
              for flip_idx in range(np.size(t_flip)):
                  # Get the time of the next flip
                  t_flip_i = t_flip[0][flip_idx]

                  '''Set the output to the sign of the flip for the
                  remainder of the trial. Future flips will overwrite future
                  output'''
                  output[trial_idx, t_flip_i:, bit_idx] = \
                      inputs[trial_idx, t_flip_i, bit_idx]

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
      return({'inputs':inputs ,'outputs': output})
    

    def reset_graph(self):
        if 'sess' in globals() and self.sess:
            self.sess.close()
        tf.reset_default_graph()

    def setup_optimizer(self, loss):

      if self.opt == 'adam':
        optimizer = tf.train.AdagradOptimizer(0.001*hvd.size())
        optimizer = hvd.DistributedOptimizer(optimizer)
        train_op = optimizer.minimize(loss, global_step=self.global_step)

      elif self.opt == 'momentum':
        decay = tf.train.exponential_decay(1e-02, self.global_step, 1, 0.9)
        optimizer = tf.train.MomentumOptimizer(decay*hvd.size(), 0.5)
        # optimizer = tf.train.AdamOptimizer(decay*hvd.size(),epsilon=1e-1)
        optimizer = hvd.DistributedOptimizer( optimizer)
        gradients, variables = zip(*optimizer.compute_gradients(loss,tf.trainable_variables()))
        gradients = [None if gradient is None else tf.clip_by_norm(gradient, 2.0) for gradient in gradients]
        train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)

      return train_op

    def is_lstm(self,x):
      if isinstance(x, tf.nn.rnn_cell.BasicLSTMCell):

        return True
      if isinstance(x, tf.nn.rnn_cell.LSTMStateTuple):
        return True

      return False

    def unroll_LSTM(self,lstm_cell, inputs, initial_state):

      assert (self.is_lstm(lstm_cell)),('lstm_cell is not an LSTM.')
      assert (self.is_lstm(initial_state)),('initial_state is not an LSTMStateTuple.')

      ''' Add ops to the graph for getting the complete LSTM state
      (i.e., hidden and cell) at every timestep.'''
      n_time = inputs.shape[1].value
      hidden_list = []
      cell_list = []

      prev_state = initial_state

      for t in range(n_time):

          input_ = inputs[:,t,:]

          _, state = lstm_cell(input_, prev_state)

          hidden_list.append(state.h)
          cell_list.append(state.c)
          prev_state = state

      c = tf.stack(cell_list, axis=1)
      h = tf.stack(hidden_list, axis=1)

      return tf.nn.rnn_cell.LSTMStateTuple(c=c, h=h)    
    
    def setup_model(self):

      self.reset_graph()
      
      x = tf.placeholder(tf.float32, [self.batch_size, self.time, self.bits], name='input_placeholder')
      y = tf.placeholder(tf.float32, [self.batch_size, self.time, self.bits], name='labels_placeholder')
    
      if self.c_type == 'Vanilla':
        if self.activation == 'tanh':
          self.cell = tf.nn.rnn_cell.BasicRNNCell(self.state_size,reuse=tf.AUTO_REUSE, activation=tf.nn.tanh)
        else:
          self.cell = tf.nn.rnn_cell.BasicRNNCell(self.state_size,reuse=tf.AUTO_REUSE, activation=tf.nn.relu)

      if self.c_type == 'GRU':
        if self.activation == 'tanh':
          self.cell = tf.nn.rnn_cell.GRUCell(self.state_size,reuse=tf.AUTO_REUSE, activation=tf.nn.tanh)
        else:
          self.cell = tf.nn.rnn_cell.GRUCell(self.state_size,reuse=tf.AUTO_REUSE, activation=tf.nn.relu)

      if self.c_type == 'UGRNN':
        if self.activation == 'tanh':
          self.cell = tf.contrib.rnn.UGRNNCell(self.state_size,reuse=tf.AUTO_REUSE, activation=tf.nn.tanh)
        else:
          self.cell = tf.contrib.rnn.UGRNNCell(self.state_size,reuse=tf.AUTO_REUSE, activation=tf.nn.relu)

      if self.c_type == 'LSTM':
        if self.activation == 'tanh':
          self.cell = tf.nn.rnn_cell.BasicLSTMCell(self.state_size,reuse=tf.AUTO_REUSE,state_is_tuple=True, activation=tf.nn.tanh)
        else:
          self.cell = tf.nn.rnn_cell.BasicLSTMCell(self.state_size,reuse=tf.AUTO_REUSE,state_is_tuple=True, activation=tf.nn.relu)

      init_state = self.cell.zero_state(self.batch_size, dtype=tf.float32)

      if self.c_type == 'LSTM':
        rnn_outputs = self.unroll_LSTM(self.cell,x,init_state)
        hiddens = rnn_outputs.h
      else:
        rnn_outputs, final_state = tf.nn.dynamic_rnn(self.cell, x, initial_state=init_state,)
        hiddens = rnn_outputs
    
    
      scale = 1.0 / np.sqrt(self.state_size)
      W = np.multiply(scale,np.random.randn(self.state_size, self.bits))
      b = np.zeros(self.bits)
      # with tf.variable_scope('losses',reuse=tf.AUTO_REUSE):
      #     W = tf.get_variable('W', [self.state_size , self.bits])
      #     b = tf.get_variable('b', [self.bits], initializer=tf.constant_initializer(0.0))

      with tf.variable_scope('losses',reuse=tf.AUTO_REUSE):
          W = tf.Variable(W,dtype=tf.float32)
          b = tf.Variable(b,dtype=tf.float32)
          
      logits = tf.tensordot(hiddens,W,axes=1) + b
      self.global_step = tf.train.get_or_create_global_step()    
      y_as_list =tf.reshape(y, [-1, self.bits]) #shape is flattened_tensor x bits
      vars_   = tf.trainable_variables() 

      # lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in vars_ if 'b' not in v.name ]) * self.l2_loss
      lossL2 = tf.nn.l2_loss(W)* self.l2_loss

      losses = tf.squared_difference(logits, y) + lossL2
      total_loss = tf.reduce_mean(losses)

      train_op = self.setup_optimizer(total_loss)

      return {'losses':total_loss, 
              'train_step':train_op, 
              'hiddens':rnn_outputs,
              'X':x, 
              'Y':y, 
              'predict':logits, 
              'init_state':init_state , 
              'cell':self.cell, 
              'weights':W ,
              }

    def get_path_for_saving(self):

      name = str(self.c_type)+'_hps_states_'+str(self.state_size)+'_l2_'+str(self.l2_loss)+'_'+str(self.activation) 
      dir_name = os.getcwd()+'/trained'+str(self.seed)+'/'+name
      try:
        if not os.path.exists(os.path.dirname(dir_name)):
          os.makedirs(os.path.dirname(dir_name))
        self.path = dir_name 
      except: 
        self.path = dir_name 
        


    def train_network(self, n_iteration, verbose=True, save=True):
 
      hvd.init()

      config = tf.ConfigProto()
      config.gpu_options.allow_growth = True
      config.gpu_options.visible_device_list = str(hvd.local_rank())
      t1 = dt.datetime.now()
      # config = self.setup_mpi_horovod()
      self.is_root = hvd.rank() == 0
      act = self.setup_model()

      # self.loaddata()


      # epochs = 0
      training_loss = 0
      hidden = []
      num_steps = n_iteration//hvd.size()
      epochs = 2

      # if self.is_root: print('data loaded with size ',num_steps)

      hooks = [hvd.BroadcastGlobalVariablesHook(0),

      tf.train.StopAtStepHook(last_step = num_steps),

      tf.train.LoggingTensorHook(tensors={'step': self.global_step, 
                                          'loss': act['losses']},
                                 every_n_iter=10) ]
      self.get_path_for_saving()
      if self.is_root:
        print(self.path)

      checkpoint_dir = self.path if hvd.rank() == 0 else None
      with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir,
                                               hooks=hooks,
                                               config=config,
                                               summary_dir = self.path+'/tf_logs/') as sess:
        self.sess = sess

        # summary_writer = tf.summary.FileWriter(self.logdir, graph=tf.get_default_graph())
        i = 0
        training_losses = []
        iteration = 0
        if self.is_root: print('Training statrted with hps ', self.hps)
        # for i in range(epochs):
        while not self.sess.should_stop():
          if self.opt == 'norm' or self.opt == 'momentum' :
            data = self.flip_flop()
            ground_truth = data['outputs']  
            tr_losses, training_step_, outputs, predict = \
                        self.sess.run([act['losses'],
                                      act['train_step'],
                                      act['hiddens'],
                                      act['predict']],
                                      feed_dict={act['X']:data['inputs'], 
                                                act['Y']:data['outputs']
                                                })       
          training_loss += tr_losses

          # if self.is_root:
          #   summary_writer.add_summary(summary, iteration)
          
          if iteration%num_steps==0:
            i=0
          else:
            i+=1

          if self.is_root and verbose:
            if iteration%100==0:
              # ep_100 = training_loss/100.0
              print("ITERATION  {iter} epoch {epoch} loss {loss} ".format(iter=iteration,loss=tr_losses,epoch=iteration//num_steps))
            # training_loss = 0
          training_losses.append(training_loss)
            # if ep_100<0.1:
            #   break


          if iteration == 0:
            hidden.append(outputs)
          iteration +=1

          # if self.is_root:
          #   validation_data = self.flip_flop()
          #   sess = tf.Session(config = config)
          #   tr_losses= self.sess.run([act['losses']],feed_dict={act['X']:validation_data['inputs'], act['Y']:validation_data['outputs']})
          #   print('Validation loss is ',tr_losses)


        hidden.append(outputs)
        t2 = (dt.datetime.now()- t1).seconds

        if self.is_root: print('training finished with time ',t2) 
      return {'losses':training_losses, 'hidden':hidden, 'predictions':predict, 'truth':ground_truth}

    def lstm_hiddens(self, graph):

      n_hidden = self.state_size
      [self.batch_size, self.time, self.bits] = np.array(self.test_data['inputs']).shape
      initial_state = graph['cell'].zero_state(self.batch_size, dtype=tf.float32)

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
      
      # print(states.c)

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
      hvd.init()

      config = tf.ConfigProto()
      config.gpu_options.allow_growth = True
      config.gpu_options.visible_device_list = str(hvd.local_rank())
      
      model = self.setup_model()
 
      self.is_root = hvd.rank() == 0
      chkpt_ = tf.train.get_checkpoint_state(chkpt)

      self.graph = model

      saver = tf.train.Saver()

      self.test_data = self.flip_flop()

      feed_dict = {self.graph['X']:self.test_data['inputs'],self.graph['Y']:self.test_data['outputs']}

      self.sess = tf.Session(config = config)

      self.sess.run(tf.global_variables_initializer())

      saver.restore(self.sess, chkpt_.model_checkpoint_path)

      if self.is_root: print("successfully loaded from checkpoints")

      hiddens, outputs, losses = self.sess.run([self.graph['hiddens'],self.graph['predict'],self.graph['losses']],feed_dict=feed_dict)
    
      if(self.c_type=='LSTM'):

        return({'hiddens':self.lstm_hiddens(self.graph),'predictions':outputs,'loss':losses,'truth':self.test_data['outputs'],'inputs':self.test_data['inputs']})
        
      else:
        
        return({'hiddens':hiddens,'predictions':outputs,'loss':losses, 'truth':self.test_data['outputs'],'inputs':self.test_data['inputs']})


