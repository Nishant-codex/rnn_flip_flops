
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
    {'time' : 256,
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
      
        self.seed = _seed
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
        self.adaptive_learning_rate = AdaptiveLearningRate(**{})#hps['al_hps'])
        self.adaptive_grad_norm_clip = AdaptiveGradNormClip(**{})
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
        inp_inv = inp_.T
        out_inv = out_.T
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
      return({'inputs':np.array(inputs),'outputs':np.array(outputs)})
    
    def data_batches(self,num_batches, num_processes, save =False):
      batch_list = []
      t1 = round(time.time()//1e6)
      total = 0
      n = 0
      num_division = num_batches // num_processes
      print(num_division)
      for i in range(num_batches):
        print( 'iteration # ',i)
        single_batch = self.flip_flop()
        batch_list.append(single_batch)
        t2 = round((time.time()-t1)//1e6, 3)
        total +=t2

        t1 = t2
        if (i+1) % num_division==0:

          self.save_data(batch_list, n)
          n+=1
          batch_list = []

      print('total time %f' %total)
      return batch_list
      
    def save_data(self, array , i):
      dir_name = os.getcwd()+'/data_'+str(self.seed)+'/'
      print(dir_name)
      if not os.path.exists(os.path.dirname(dir_name)):
        os.makedirs(os.path.dirname(dir_name))

      filename  = dir_name+'data'+str(i)+'.npy'   
      f =  open(filename,'wb')
      np.save(f, np.array(array) ,allow_pickle=True)

    def restore_data(self, path):
      file = open(path,'rb')
      restore_data = np.load(file, allow_pickle=True)
      # file.close()
      # data = pickle.loads(restore_data,encoding='latin1')
      return restore_data

    def reset_graph(self):
        if 'sess' in globals() and self.sess:
            self.sess.close()
        tf.reset_default_graph()

    def setup_optimizer(self, loss):

      # with tf.variable_scope('record' , trainable = False, reuse = False):
      #   self.global_step = tf.Variable( 0, name='global_step', trainable=False, dtype=tf.int32)

      # vars_to_train = tf.trainable_variables()

      # with tf.variable_scope('optimizer', reuse=False):

      #   # Gradient clipping
      #   grads = tf.gradients(loss, vars_to_train)

      #   self.grad_norm_clip_val = tf.placeholder(self.dtype, name='grad_norm_clip_val')

      #   clipped_grads, self.grad_global_norm = tf.clip_by_global_norm(grads, self.grad_norm_clip_val)

      #   clipped_grad_global_norm = tf.global_norm(clipped_grads)

      #   clipped_grad_norm_diff = self.grad_global_norm - clipped_grad_global_norm

      #   zipped_grads = zip(clipped_grads, vars_to_train)

      #   self.learning_rate = tf.placeholder(self.dtype, name='learning_rate')
        
      #   adam_hps =  {'epsilon': 0.01,
      #               'beta1': 0.9,
      #               'beta2': 0.999,
      #               'use_locking': False,
      #               'name': 'Adam'}
      #   if self.opt == 'adam':
      #     optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate*hvd.size(), **adam_hps)
      #     optimizer = hvd.DistributedOptimizer(optimizer)
      #     train_op = optimizer.apply_gradients(zipped_grads, global_step=self.global_step)

      if self.opt == 'norm':
        optimizer = tf.train.AdagradOptimizer(0.001*hvd.size())
        optimizer = hvd.DistributedOptimizer(optimizer)
        train_op = optimizer.minimize(loss, global_step=self.global_step)

      elif self.opt == 'momentum':
        decay = tf.train.exponential_decay(0.1, self.global_step, 20000, 0.99)
        optimizer = tf.train.MomentumOptimizer(decay, 0.001)
        # optimizer = tf.train.AdamOptimizer(decay*hvd.size(),epsilon=1e-1)
        optimizer = hvd.DistributedOptimizer(optimizer)
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients = [None if gradient is None else tf.clip_by_norm(gradient, 10.0) for gradient in gradients]
        train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)

      return train_op

    
    def setup_model(self):

      self.reset_graph()
      
      x = tf.placeholder(tf.float32, [self.batch_size, self.time, self.bits], name='input_placeholder')
      y = tf.placeholder(tf.float32, [self.batch_size, self.time, self.bits], name='labels_placeholder')
    
      if self.c_type == 'Vanilla':
        if self.activation == 'tanh':
          self.cell = tf.contrib.rnn.BasicRNNCell(self.state_size,reuse=tf.AUTO_REUSE, activation=tf.nn.tanh)
        else:
          self.cell = tf.contrib.rnn.BasicRNNCell(self.state_size,reuse=tf.AUTO_REUSE, activation=tf.nn.relu)

      if self.c_type == 'GRU':
        if self.activation == 'tanh':
          self.cell = tf.contrib.rnn.GRUCell(self.state_size,reuse=tf.AUTO_REUSE, activation=tf.nn.tanh)
        else:
          self.cell = tf.contrib.rnn.GRUCell(self.state_size,reuse=tf.AUTO_REUSE, activation=tf.nn.relu)

      if self.c_type == 'UGRNN':
        if self.activation == 'tanh':
          self.cell = tf.contrib.rnn.UGRNNCell(self.state_size,reuse=tf.AUTO_REUSE, activation=tf.nn.tanh)
        else:
          self.cell = tf.contrib.rnn.UGRNNCell(self.state_size,reuse=tf.AUTO_REUSE, activation=tf.nn.relu)

      if self.c_type == 'LSTM':
        if self.activation == 'tanh':
          self.cell = tf.contrib.rnn.LSTMCell(self.state_size,reuse=tf.AUTO_REUSE,state_is_tuple=True, activation=tf.nn.tanh)
        else:
          self.cell = tf.contrib.rnn.LSTMCell(self.state_size,reuse=tf.AUTO_REUSE,state_is_tuple=True, activation=tf.nn.relu)


      init_state = self.cell.zero_state(self.batch_size, dtype=tf.float32)
    
      rnn_outputs, final_state = tf.nn.dynamic_rnn(self.cell, x, initial_state=init_state,)
    
      """
      rnn_outputs gives out rnn hidden states ht which is of the size [batch_size, timestep, state_size]
    
      """
      with tf.variable_scope('losses',reuse=tf.AUTO_REUSE):
          W = tf.get_variable('W', [self.state_size , self.bits])
          b = tf.get_variable('b', [self.bits], initializer=tf.constant_initializer(0.0))
      rnn_outputs_ = tf.reshape(rnn_outputs, [-1, self.state_size])
    
      logits = tf.tensordot(rnn_outputs_,W,axes=1) + b
      self.global_step = tf.train.get_or_create_global_step()    
      y_as_list =tf.reshape(y, [-1, self.bits]) #shape is flattened_tensor x bits
      vars_   = tf.trainable_variables() 

      # lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in vars_ if 'b' not in v.name ]) * self.l2_loss
      lossL2 = tf.nn.l2_loss(W)* self.l2_loss

      losses = tf.squared_difference(logits, y_as_list) + lossL2
      total_loss = tf.reduce_mean(losses)

      train_op = self.setup_optimizer(total_loss)

      return {'losses':total_loss, 
              'train_step':train_op, 
              'hiddens':rnn_outputs,
              'finalstate':final_state , 
              'X':x, 
              'Y':y, 
              'predict':logits, 
              'init_state':init_state , 
              'cell':self.cell, 
              'weights':W ,
              }

    def get_path_for_saving(self):

      name = str(self.c_type)+'_hps_states_'+str(self.state_size)+'_l2_'+str(self.l2_loss)+'_'+str(self.activation) 
      dir_name = os.getcwd()+'/trained/'+name
      if not os.path.exists(os.path.dirname(dir_name)):
          os.makedirs(os.path.dirname(dir_name))
          self.path = dir_name 
      else: 
          self.path = dir_name 
        

    def setup_tensorboard(self, dict):

      dir_name = self.path+'/tf_logs/'
      try:
        if not os.path.exists(os.path.dirname(dir_name)):
          os.makedirs(os.path.dirname(dir_name))
      
          self.logdir =  dir_name
      except:
          self.logdir =  dir_name

      tf.summary.scalar("loss",dict['losses'])
      self.merged_summary_op = tf.summary.merge_all()


    def setup_mpi_horovod(self):

      hvd.init()

      config = tf.ConfigProto()
      config.gpu_options.allow_growth = True
      config.gpu_options.visible_device_list = str(hvd.local_rank())

      if not hvd.mpi_threads_supported():
        raise MpiInitError(
            'MPI multi-threading is not supported. Horovod cannot work with mpi4py'
            'in this case. Please enable MPI multi-threading and try again.'
        )

      mpi4py.rc.initialize = False


      # Verify that Horovod and mpi4py are using the same number of ranks
      from mpi4py import MPI

      if hvd.size() != MPI.COMM_WORLD.Get_size():
          raise MpiInitError('Mismatch in hvd.size() and MPI.COMM_WORLD size.'
              f' No. of ranks in Horovod: {hvd.size()}.'
              f' No. of ranks in mpi4py: {MPI.COMM_WORLD.Get_size()}'
          )
      return config
    
    def get_filenames(self, path):

      absolute_path = os.path.join(os.path.abspath(f'{path}'))

      return os.listdir(absolute_path)

    def get_data(self, path, filenames):
      arrays = [np.load(os.path.join(path, f)) for f in filenames]
      return np.concatenate(arrays)

    def loaddata(self):
      dist_decorator = DataDistributor(mpi_comm=mpi4py.MPI.COMM_WORLD, shutdown_on_error=True)
  
      get_rank_local_filenames = dist_decorator(self.get_filenames)

      datapath = os.getcwd()+'/data_'+str(self.seed)
      filenames = get_rank_local_filenames(datapath)
      self.data = self.get_data(datapath, filenames)


    def train_network(self, verbose=True, save=True):
      t1 = dt.datetime.now()
      config = self.setup_mpi_horovod()
      self.is_root = hvd.rank() == 0

      act = self.setup_model()

      self.loaddata()


      epochs = 0
      training_loss = 0
      hidden = []
      num_steps = len(self.data)#//hvd.size()+1
      if self.is_root: print('data loaded with size ',num_steps)

      hooks = [hvd.BroadcastGlobalVariablesHook(0),

      tf.train.StopAtStepHook(last_step = num_steps),

      tf.train.LoggingTensorHook(tensors={'step': self.global_step, 
                                          'loss': act['losses']},
                                 every_n_iter=10) ]
      self.get_path_for_saving()
      self.setup_tensorboard(act)
      if self.is_root:
        print(self.path)

      checkpoint_dir = self.path if hvd.rank() == 0 else None
      with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir,
                                               hooks=hooks,
                                               config=config) as sess:
        self.sess = sess

        summary_writer = tf.summary.FileWriter(self.logdir, graph=tf.get_default_graph())

        training_losses = []
        # num_steps = len(data)//hvd.size()+1
        i = 0
        if self.is_root: print('Training statrted with hps ', self.hps)
        while not self.sess.should_stop():
          if self.opt == 'norm' or self.opt == 'momentum' :
            ground_truth = self.data[i]['outputs']  
            tr_losses, training_step_, training_state, outputs, predict , summary= \
                        self.sess.run([act['losses'],
                                      act['train_step'],
                                      act['finalstate'],
                                      act['hiddens'],
                                      act['predict'], 
                                      self.merged_summary_op],
                                      feed_dict={act['X']:self.data[i]['inputs'], 
                                                act['Y']:self.data[i]['outputs'],
                                                })
          # if tr_losses==np.nan or tr_losses ==np.inf:
          #   raise "gra"
          # elif self.opt == 'adam':
          #   ground_truth = self.data[i]['outputs']  
          #   tr_losses, training_step_, training_state, outputs, predict , summary , ev_grad_global_norm= \
          #               self.sess.run([act['losses'],
          #                             act['train_step'],
          #                             act['finalstate'],
          #                             act['hiddens'],
          #                             act['predict'], 
          #                             self.merged_summary_op,
          #                             self.grad_global_norm],
          #                             feed_dict={act['X']:self.data[i]['inputs'], 
          #                                       act['Y']:self.data[i]['outputs'],
          #                                       self.learning_rate:self.adaptive_learning_rate(),
          #                                       self.grad_norm_clip_val:self.adaptive_grad_norm_clip()
          #                                       })            
          training_loss += tr_losses

          # self.sess.run(self.global_step)
          if self.is_root:
            summary_writer.add_summary(summary, epochs)
          i +=1
          if self.is_root and verbose:
              print("Average training loss for ITERATION", epochs, ":", tr_losses)
          training_losses.append(training_loss)
          training_loss = 0
          # if epochs%1000:
          #   time_elapsed = time.time() - time_1
          #   print('Time after 1000 iteration',time_elapsed)
          #   time = time.time()
          if epochs == 0:
            hidden.append(outputs)
          epochs +=1
        # self.adaptive_learning_rate.update(tr_losses)
        # self.adaptive_grad_norm_clip.update(ev_grad_global_norm)
        hidden.append(outputs)
        t2 = (dt.datetime.now()- t1).seconds
        if self.is_root: print('training finished with time ',t2) 

        # if hvd.rank() == 0 :
        #   if(save):
        #     saver.save(self.sess, path)
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

        return({'hiddens':self.lstm_hiddens(graph),'predictions':outputs,'loss':losses})
        
      else:
        
        return({'hiddens':hiddens,'predictions':outputs,'loss':losses, 'truth':self.test_data['outputs']})


