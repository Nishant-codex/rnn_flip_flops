import numpy as np
import os
import absl
# %tensorflow_version 1.x
import tensorflow as tf
import pylab 
from tensorflow.python.ops import parallel_for as pfor
import matplotlib.pyplot as plt
# %matplotlib inline
import numpy.random as nrand 
import pickle
from FixedPointStore import FixedPointStore
from FixedPointSearch import FixedPointSearch
import pickle
import tables
import time
from AdaptiveGradNormClip import AdaptiveGradNormClip
from AdaptiveLearningRate import AdaptiveLearningRate
from collections import namedtuple

LIFStateTuple = namedtuple('LIFStateTuple', ('v', 'z', 'i_future_buffer', 'z_buffer'))
ALIFStateTuple = namedtuple('ALIFStateTuple', ('z','v','b','i_future_buffer','z_buffer'))

import sys 
sys.path.insert(0,'/content/LSNN-official/')
import lsnn.spiking_models as lsnn
from lsnn.toolbox.tensorflow_einsums.einsum_re_written import einsum_bij_jk_to_bik

# from lsnn.toolbox.rewiring_tools import rewiring_optimizer_wrapper
from lsnn.spiking_models import tf_cell_to_savable_dict, exp_convolve #ALIF, LIF
from lsnn.toolbox.rewiring_tools import weight_sampler, rewiring_optimizer_wrapper

class FlipFlop:
#Hyperparameters
    
    hyp_dict = \
    {'time' : 500,
    'bits' : 3 ,
    'num_steps' : 6,
    'batch_size' : 64,
    'neurons' : 64 ,
    'num_classes' : 2,
    'p': 0.2,
    'learning_rate' :0.01,
    'decay_steps': 100,
    'c_type':'UGRNN'
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
        neurons = hyp_dict['neurons'],
        num_classes = hyp_dict['num_classes'],
        learning_rate = hyp_dict['learning_rate'],
        p = hyp_dict['p'],
        c_type = hyp_dict['c_type'],
        l2 = 0.01,
        decay_steps = 0,
        _seed = 400,
        **hps):
      
        self.seed = _seed
        self.time = time
        self.new_lr = 0
        self.hps = hps
        self.cell = None
        self.activation = 'tanh'
        self.lr_update = 0
        self.bits = bits
        self.model = None
        self.fps = None
        self.alr_hps = {}#{'initial_rate': 1.0, 'min_rate': 1e-5}
        self.grad_global_norm = 0
        self.grad_norm_clip_val = 0
        self.dtype = tf.float32
        self.opt = 'norm'
        self.l2_loss = l2
        self.decay_steps = decay_steps
        # self.adaptive_learning_rate = AdaptiveLearningRate(**self.alr_hps)
        # self.adaptive_grad_norm_clip = AdaptiveGradNormClip(**{})
        self.num_steps = num_steps
        self.num_steps =num_steps 
        self.batch_size = batch_size 
        self.neurons = neurons 
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.p = p  
        self.graph = 0
        self.chkpt = None
        self.c_type = c_type    
        self.sess = 0
        self.test_data = []

        np.random.seed(self.seed)

    def flip_flop(self, p = 0, plot=False):

      unsigned_inp = np.random.binomial(1,p,[self.batch_size,self.time//10,self.bits])
      unsigned_out = 2*np.random.binomial(1,0.5,[self.batch_size,self.time//10,self.bits]) -1 



      inputs = np.multiply(unsigned_inp,unsigned_out)
      inputs[:,0,:] = 1
      inputs = np.repeat(inputs,10,axis=1)
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

      
      if plot:
        fig, ax = plt.subplots(3)

        ax[0].plot(inputs[0,:,0])
        ax[0].plot(output[0,:,0])
        ax[0].set_xlabel("time")
        ax[0].set_ylabel("Bit 0")
        ax[1].plot(inputs[0,:,1])
        ax[1].plot(output[0,:,1])
        ax[1].set_xlabel("time")
        ax[1].set_ylabel("Bit 1")
        ax[2].plot(inputs[0,:,2])
        ax[2].plot(output[0,:,2])
        ax[2].set_xlabel("time")
        ax[2].set_ylabel("Bit 2")
        plt.show()


      return {'inputs': inputs, 'output': output}

    
    def data_batches(self,num_batches,seed=None, save =False):
      batch_list = []
      if seed is not None:
        # self.seed = seed
        np.random.seed(seed)
      t1 = round(time.time()//1e6)
      total = 0
      for i in range(num_batches):

        single_batch = self.flip_flop()
        batch_list.append(single_batch)
        t2 = round((time.time()-t1)//1e6, 3)
        total +=t2
        # if i ==0:
        # print('time to generate' + 'batch'+str(i)+ 'is ', total)
        t1 = t2
        
      self.save_data(batch_list)
          # del(batch_list)
          # batch_list = []


      print('total time is %f', total)
      return batch_list
    def setup_optimizer(self, loss,rewiring_connectivity=None, temperature= None):

      if self.c_type == 'LIF' or self.c_type == 'ALIF':
        global_step = tf.Variable(0, dtype=tf.int32, trainable=False)
        learning_rate = tf.Variable(self.hps['lrate'], dtype=tf.float32, trainable=False)
        decay_learning_rate_op = tf.assign(learning_rate, learning_rate * self.hps['decay'])  # Op to decay learning rate

        loss = loss

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        l1 = 1e-01

        # if 0 < rewiring_connectivity:

        #     train_op = rewiring_optimizer_wrapper(optimizer, loss, learning_rate, l1, temperature,
        #                                             rewiring_connectivity,
        #                                             global_step = global_step,
        #                                             var_list=tf.trainable_variables())
        # else:
        train_op = optimizer.minimize(loss=loss, global_step=global_step)
    #   # else:
    #   self.global_step = tf.Variable(0, trainable=False)
    #   decay = tf.train.exponential_decay(self.hps['lrate'],self.global_step,1000,self.hps['decay'])
    # #   optimizer = tf.train.MomentumOptimizer(decay,self.hps['moment'],use_nesterov=False)
    #   # optimizer = tf.train.AdamOptimizer(learning_rate=self.hps['lrate'])
    #   # optimizer = tf.train.AdamOptimizer(learning_rate=decay,)
    #   # gradients, variables = zip(*optimizer.compute_gradients(loss,tf.trainable_variables()))
    #   # gradients = [
    #   #     None if gradient is None else tf.clip_by_norm(gradient, hps['norm'])
    #   #     for gradient in gradients]
    #   # train_op = optimizer.apply_gradients(zip(gradients, variables))
    #   train_op = optimizer.minimize(loss,global_step = self.global_step)

      return train_op, decay_learning_rate_op
      
    def save_data(self, array):
      dir_name = os.getcwd()+'/data_'+str(self.seed)+'/'
      if not os.path.exists(os.path.dirname(dir_name)):
        os.makedirs(os.path.dirname(dir_name))

      filename  = dir_name+'data.npy'   
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
          self.cell = tf.nn.rnn_cell.BasicRNNCell(self.neurons,reuse=tf.AUTO_REUSE, activation=tf.nn.tanh)
        else:
          self.cell = tf.nn.rnn_cell.BasicRNNCell(self.neurons,reuse=tf.AUTO_REUSE, activation=tf.nn.relu)

      if self.c_type == 'GRU':
        if self.activation == 'tanh':
          self.cell = tf.nn.rnn_cell.GRUCell(self.neurons,reuse=tf.AUTO_REUSE, activation=tf.nn.tanh)
        else:
          self.cell = tf.nn.rnn_cell.GRUCell(self.neurons,reuse=tf.AUTO_REUSE, activation=tf.nn.relu)

      if self.c_type == 'UGRNN':
        if self.activation == 'tanh':
          self.cell = tf.contrib.rnn.UGRNNCell(self.neurons,reuse=tf.AUTO_REUSE, activation=tf.nn.tanh)
        else:
          self.cell = tf.contrib.rnn.UGRNNCell(self.neurons,reuse=tf.AUTO_REUSE, activation=tf.nn.relu)

      if self.c_type == 'LSTM':
        if self.activation == 'tanh':
          self.cell = tf.nn.rnn_cell.BasicLSTMCell(self.neurons,reuse=tf.AUTO_REUSE,state_is_tuple=True, activation=tf.nn.tanh)
        else:
          self.cell = tf.nn.rnn_cell.BasicLSTMCell(self.neurons,reuse=tf.AUTO_REUSE,state_is_tuple=True, activation=tf.nn.relu)
      ##
      FLAGS = tf.app.flags.FLAGS

      beta= 1.8
      # to solve a task successfully we usually set tau_a to be close to the expected delay / memory length needed
      tau_a = 150
      tau_v = 20
      dt = 1
      thr= 0.01
      learning_rate = 1e-2
      n_regular = self.hps['n_regular']

      n_adaptive = self.hps['adaptive']
      lr_decay = 0.8
      reg = 1e-1
      rewiring_temperature = 0.
      proportion_excitatory= 0.8
      n_in = 3

      rewiring_connectivity =.12
      l1 = 1e-2
      dampening_factor = 0.3
      neuron_sign = True
      n_delay = 1
      n_ref = 0
      # Sign of the neurons
      if 0 < rewiring_connectivity and neuron_sign:
        n_excitatory_in = int(proportion_excitatory * n_in) + 1
        n_inhibitory_in = n_in - n_excitatory_in
        print('Inhibitory num ',n_inhibitory_in)
        in_neuron_sign = np.concatenate([-np.ones(n_inhibitory_in), np.ones(n_excitatory_in)])
        np.random.shuffle(in_neuron_sign)


        n_excitatory = int(proportion_excitatory * (n_regular + n_adaptive)) + 1
        n_inhibitory = n_regular + n_adaptive - n_excitatory
        rec_neuron_sign = np.concatenate([-np.ones(n_inhibitory), np.ones(n_excitatory)])
      else:
          if not (neuron_sign == False): print(
              'WARNING: Neuron sign is set to None without rewiring but sign is requested')
          in_neuron_sign = None
          rec_neuron_sign = None

      if self.c_type == 'LIF':
        
        self.cell = lsnn.LIF(self.bits,
                             self.neurons,
                             tau=tau_v, 
                             thr = thr, 
                             rewiring_connectivity=rewiring_connectivity,
                             in_neuron_sign = in_neuron_sign,
                             dampening_factor = dampening_factor,
                             rec_neuron_sign=rec_neuron_sign,
                             n_refractory = n_ref,
                             n_delay = n_delay,
                             )
      if self.c_type == 'ALIF':
        beta = np.concatenate([np.zeros(n_regular), np.ones(n_adaptive) * beta])

        self.cell = lsnn.ALIF(n_in=self.bits, 
                              n_rec=n_regular + n_adaptive, 
                              tau=tau_v, 
                              n_delay=n_delay,
                              n_refractory=n_ref, 
                              dt=1, tau_adaptation=tau_a, 
                              beta=beta, 
                              thr=thr,
                              rewiring_connectivity=rewiring_connectivity,
                              in_neuron_sign=in_neuron_sign, rec_neuron_sign=rec_neuron_sign,
                              dampening_factor=dampening_factor,
                              )
      init_state = self.cell.zero_state(self.batch_size, dtype=tf.float32)
      if self.c_type == 'LSTM':
        rnn_outputs = self.unroll_LSTM(self.cell,x,init_state)
        hiddens = rnn_outputs.h
      elif self.c_type == 'LIF':
          z = []
          v = []
          states = []
          state_inst = init_state
          for t_idx in range(self.time):

            z_inst, state_inst = self.cell(x[:,t_idx,:],state_inst)
            z.append(z_inst)
            v.append(state_inst.v)
            states.append(state_inst)
      elif self.c_type == 'ALIF':
            z = []
            v = []
            bias = []
            states = []
            state_inst = init_state
            for t_idx in range(self.time):

              out, state_inst = self.cell(x[:,t_idx,:],state_inst)
              z_,v_,b_ = out
              z.append(z_)
              v.append(v_)
              bias.append(b_)
              states.append(state_inst)

      else:
        rnn_outputs, final_state = tf.nn.dynamic_rnn(self.cell, x, initial_state=init_state,)

        hiddens = rnn_outputs
      """
      rnn_outputs gives out rnn hidden states ht which is of the size [batch_size, timestep, neurons]
    
      """

      scale = 1.0 / np.sqrt(self.neurons)
      W = np.multiply(scale,np.random.randn(self.neurons, self.bits))
      b = np.zeros(self.bits)
      # with tf.variable_scope('losses',reuse=tf.AUTO_REUSE):
      #     W = tf.get_variable('W', [self.neurons , self.bits])
      #     b = tf.get_variable('b', [self.bits], initializer=tf.constant_initializer(0.0))
      z = tf.stack(z,axis=1)
      v = tf.stack(v,axis=1)
      bias = tf.stack(bias,axis=1)
      
      # z = tf.convert_to_tensor(z)
      psp_decay = np.exp(-dt / tau_v)  # output layer psp decay, chose value between 15 and 30ms as for tau_v
      psp = exp_convolve(z, decay=psp_decay)
      # n_neurons =  self.neurons #z.get_shape()[2]
      # n_output_symbols = self.bits
      # # Define the readout weights
      # if 0 < rewiring_connectivity:
      #     w_out, w_out_sign, w_out_var, _ = weight_sampler(n_regular + n_adaptive, 
      #                                                      n_output_symbols,
      #                                                      rewiring_connectivity,
      #                                                      neuron_sign=rec_neuron_sign)
      # else:
      #     w_out = tf.get_variable(name='out_weight', shape=[n_neurons, n_output_symbols])
      # b_out = tf.get_variable(name='out_bias', shape=[n_output_symbols], initializer=tf.zeros_initializer())

      # # Define the loss function

      with tf.variable_scope('losses',reuse=tf.AUTO_REUSE):
          W = tf.Variable(W,dtype=tf.float32)
          b = tf.Variable(b,dtype=tf.float32)
    #   z = tf.convert_to_tensor(z)

      # out = einsum_bij_jk_to_bik(psp, w_out) + b_out
      if self.c_type == 'LIF' or 'ALIF':
        logits = tf.tensordot(z,W,axes=1) + b
        # logits = out
      else:
        logits = tf.tensordot(hiddens,W,axes=1) + b  
      vars_ = tf.trainable_variables() 


      y_as_list = tf.reshape(y, [-1, self.bits]) #shape is flattened_tensor x bits
      logits_as_list = tf.reshape(logits, [-1, self.bits])
   
      lossL2 = tf.nn.l2_loss(W)*self.l2_loss

      losses = tf.squared_difference(logits, y) + lossL2
      total_loss = tf.reduce_mean(losses)
      # self.global_step = tf.train.get_or_create_global_step()   
      train_step, decay = self.setup_optimizer(total_loss,rewiring_connectivity=rewiring_connectivity, temperature= rewiring_temperature)
      if self.c_type == 'LIF' or 'ALIF':
        return {'losses':total_loss, 'train_step':train_step, 
                'hiddens':z, 'voltage':v,'bias':bias, 'psp':psp,
                'X':x, 'Y':y, 'predict':logits, 'init_state':init_state , 
                'saver' : tf.train.Saver(),'cell':self.cell, 'weights':W,
                'init':init_state,'final_state':states, 'decay':decay}
      else:
        return {'losses':total_loss, 'train_step':train_step, 
                'hiddens':rnn_outputs, 
                'X':x, 'Y':y, 'predict':logits, 'init_state':init_state , 
                'saver' : tf.train.Saver(),'cell':self.cell, 'weights':W,
                'init':init_state,'final_state':final_state}
    



    def train_network(self, num_epochs, verbose=True, save=False):
      num_epochs = num_epochs
      self.model = self.setup_model()
      print('Model is setup')
      epoch = 0
      training_loss = 0
      hidden = []
      saver = tf.train.Saver()
      max_lr_epoch=10
      learning_rate=1.0
      decay = self.hps['lrate']
      orig_decay = 0.93
      prev_val = 0
      curr_val = 0
      path = os.getcwd()+'/'+ self.c_type+str(num_epochs)
      iteration = 0
      with tf.Session() as sess:
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())
        stop = False
        training_losses = []
        validation_loss = []
        for i in range(self.hps['epochs']):
          for i in range(num_epochs):
            data = self.flip_flop(p=self.p,plot= False)
            ground_truth = data['output']
            tr_losses, training_step_, outputs, predict= \
                        self.sess.run([self.model['losses'],
                                      self.model['train_step'],
                                      self.model['hiddens'],
                                      self.model['predict'],
                                      ],
                                      feed_dict={self.model['X']:data['inputs'], self.model['Y']:data['output'], 
                                      })
            # self.sess.run(self.global_step)
            if verbose and i%100==0:
              if i!=0 and i%self.decay_steps ==0:
                decay = self.sess.run(self.model['decay'])
                print("Average training loss for itr ", iteration, ": ", 
                      tr_losses, " dec: ",decay)
              else:
                print("Average training loss for itr ", iteration, ": ", 
                      tr_losses, " dec: ",decay)
            if i%10==0:
              self.seed = 900
              # prev_val = validation_loss[i-10:i]
              test_data = self.flip_flop(p=0.2)

              feed_dict = {self.model['X']:test_data['inputs'],self.model['Y']:test_data['output']}
              
              hiddens, outputs, losses= self.sess.run([self.model['hiddens'],self.model['predict'],self.model['losses']],feed_dict=feed_dict)
              validation_loss.append(losses)

              # if len(validation_loss)>5:
              #   curr_val = np.mean(validation_loss[-100:])

              # print("moving average", curr_val)
              # if(losses>curr_val and curr_val>0 and i>1000):
              #   break
              # else:
              #   prev_val = curr_val
              # print('validation loss', losses)
            if tr_losses<2.0:
              training_losses.append(tr_losses)
            
            if iteration == 0:
              hidden.append(outputs)
            iteration +=1
            if tr_losses<7e-02 and self.c_type in ['LIF', 'ALIF']:
              break
        hidden.append(outputs)

        if(save):
          saver.save(self.sess, path)
      plt.plot(np.arange(len(training_losses)),training_losses)
      plt.plot(np.arange(0,len(training_losses),10),validation_loss,)
      plt.xlabel('iterations')
      plt.ylabel('losses')
      plt.show()
      
      return {'losses':training_losses, 'hidden':hidden, 'predictions':predict, 
              'truth':ground_truth, }

    def lstm_hiddens(self, graph):

      n_hidden = self.neurons
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
      h = np.zeros([self.batch_size, self.time, self.neurons]) # hidden states: bxtxd
      c = np.zeros([self.batch_size, self.time, self.neurons]) # cell states: bxtxd
      for t in range(self.time):
          h[:,t,:] = ev_full_state_list[0][t].h
          c[:,t,:] = ev_full_state_list[0][t].c

      ev_LSTMCellState = tf.nn.rnn_cell.LSTMStateTuple(h=h, c=c)
      return(ev_LSTMCellState)

    def hps_array(self):
      arch  = ['Vanilla', 'UGRNN', 'GRU', 'LSTM']
      activ = ['tanh' , 'relu']
      units = [64, 128, 256 ]
      l2_norm = [1e-5, 1e-4, 1e-3, 1e-2]
      hps_dict = {'arch':arch,
                  'activ':activ,
                  'units':units,
                  'l2_norm':l2_norm}
    
      hps_array = []
    
      for i in arch:
        for j in activ:
          for k in units:
            for l in l2_norm:
              hps_array.append({'arch':i,
                                'activ':j,
                                'units':k,
                                'l2_norm':l})
      return hps_array  

    def reload_from_checkpoints(self, chkpt_,create_model= False):
        model = self.model
    
        # self.is_root = hvd.rank() == 0
        self.chkpt = tf.train.get_checkpoint_state(chkpt_)
        # print(tf.trainable_variables())
        if create_model:
          self.graph = self.setup_model()
        else:
          self.graph = model
        kernel_1 = tf.all_variables()[0]
        saver = tf.train.Saver()

        self.test_data = self.flip_flop(p=0.2)

        feed_dict = {self.graph['X']:np.zeros_like(self.test_data['inputs']),
                    self.graph['Y']:np.zeros_like(self.test_data['output'])}

        self.sess = tf.Session()
        saver.restore(self.sess, self.chkpt.model_checkpoint_path)

        kernel_2 = tf.all_variables()[0]
        
        # if self.is_root: print("successfully loaded from checkpoints")
        hiddens, outputs,v, init, kernel_2,kernel_1, final, bias, psp= self.sess.run([self.graph['hiddens'],
                                                                                 self.graph['predict']
                                                                                 ,self.graph['voltage'],
                                                                self.graph['init'],
                                                                kernel_2,kernel_1,
                                                                self.graph['final_state'],
                                                                self.graph['bias'],
                                                                self.graph['psp']],
                                                                feed_dict=feed_dict)
        fname = 'w_lstm.pkl'
        pickle.dump(kernel_2, open(fname, 'wb'))
      
        if(self.c_type=='LSTM'):

            return({'hiddens':self.lstm_hiddens(self.graph),
                    'predictions':outputs,'loss':losses,
                    'truth':self.test_data['output'],'inputs':self.test_data['inputs']})
        
        else:
        
            return({'states':final,'voltage':v,'bias':bias, 'psp': psp,
                    'hiddens':hiddens,'predictions':outputs,
                    'truth':self.test_data['output'],'inputs':self.test_data['inputs']})

    def run_validation(self, chkpt_, create_model=False):
      if create_model:
        model = self.setup_model()
      else:
        model = self.model
      self.seed = 900
      # self.is_root = hvd.rank() == 0
      # chkpt = tf.train.get_checkpoint_state(self.chkpt)
      # print(tf.trainable_variables())
      self.graph = model

      saver = tf.train.Saver()
      self.chkpt = tf.train.get_checkpoint_state(chkpt_)

      self.test_data = self.flip_flop(p=0.2)
      input  = np.zeros_like(self.test_data['inputs'])
      input[:,0:10,0] = 1
      input[:,0:10,1] = 1
      input[:,0:10,2] = 1
      
      input[:,50:60,0] = -1
      input[:,50:60,1] = 1
      input[:,50:60,2] = 1

      input[:,100:110,0] = 1
      input[:,100:110,1] = -1
      input[:,100:110,2] = 1

      input[:,150:160,0] = 1
      input[:,150:160,1] = 1
      input[:,150:160,2] = -1

      input[:,200:210,0] = -1
      input[:,200:210,1] = -1
      input[:,200:210,2] = 1

      input[:,250:260,0] = -1
      input[:,250:260,1] = 1
      input[:,250:260,2] = -1

      input[:,300:310,0] = 1
      input[:,300:310,1] = -1
      input[:,300:310,2] = -1

      input[:,350:360,0] = -1
      input[:,350:360,1] = -1
      input[:,350:360,2] = -1

      target = np.zeros_like(self.test_data['output'])
      feed_dict = {self.graph['X']:input, self.graph['Y']:target}

      self.sess = tf.Session()
      saver.restore(self.sess, self.chkpt.model_checkpoint_path)

      # if self.is_root: print("successfully loaded from checkpoints")
      hiddens, outputs, losses, init= self.sess.run([self.graph['hiddens'],
                                                     self.graph['predict'],
                                                     self.graph['losses'], 
                                                     self.graph['init']],feed_dict=feed_dict)
      fig,ax = plt.subplots(3)

      ax[0].plot(outputs[0,:,0])
      ax[0].plot(input[0,:,0])
      ax[0].plot(target[0,:,0])


      ax[1].plot(outputs[0,:,1])
      ax[1].plot(input[0,:,1])
      ax[1].plot(target[0,:,1])


      ax[2].plot(outputs[0,:,2])
      ax[2].plot(input[0,:,2])
      ax[2].plot(target[0,:,2])


    # plt.plot(hiddens[0,:,:10])
      plt.show()
      print(np.array(hiddens).shape)
      return np.array(hiddens)


    def fixed_points(self):
      n_bits = 3
      inputs = np.zeros([1,n_bits])
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

      # pathlist = os.listdir(os.getcwd()+'/trained')
      # lis = reload_from_chkpt(rnn_object, chkptpath, plot=True)
      lis = self.reload_from_checkpoints('/content', create_model=True)
      save_hidden_states([lis['hiddens'],np.array(lis['voltage']),np.array(lis['bias']),np.array(lis['psp']) ], '/content')
      hps = {'activ': 'tanh', 
          'arch' : 'LSTM',
          'l2_norm' : 1e-05,
          'units' : 128,
          'seed':400 }
      td = 20. 
      dt =  1.
      r = np.zeros((64,220))

      func = lambda x ,z: x*pylab.exp(-dt/td) + z/td 
      rs = np.zeros((64, 500, 220))
      for time_idx in range(500):
        r = func(r, lis['hiddens'][:,time_idx,:])
        rs[:,time_idx,:] = r
      self.fps = FixedPointSearch(
                  self.c_type,
                  lis['states'],
                  '/content', 
                  cell=self.graph['cell'],
                  sess = self.sess
                  )
      self.tol_q = 1e-12
      self.fps.rerun_q_outliers = False
      self.fps.compute_jacobians = False 
      self.fps.decompose_jacobians = False 
      self.fps.sample_states_lif(600,rs,lis['states'], 0.04)
      unique, all_fps = self.fps.find_fixed_points(inputs, save = True)

    def transition_graph(self, chkpt_):
        def save_transitions(trans, path):
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
        # self.chkpt = tf.train.get_checkpoint_state(chkpt_)
        # # print(tf.trainable_variables())
       
        # saver = tf.train.Saver()

        # self.sess = tf.Session()
        # saver.restore(self.sess, self.chkpt.model_checkpoint_path)
        fps = FixedPointStore(num_inits = 1,
                            num_states= 1,
                            num_inputs = 1)
        dict_d = fps.restore(os.getcwd()+'/fps_saver/fixedPoint_unique.p')
        fps.__dict__ = dict_d
        trans = np.zeros([fps.num_inits,fps.num_inits])
        sess = self.sess #tf.Session()
        zero_inpt = np.zeros(( 64, self.time ,  3))
        zero_inpt[:,0:10,0] = 1
        zero_inpt[:,0:10,1] = 1
        zero_inpt[:,0:10,2] = 1
        
        zero_inpt[:,50:60,0] = -1
        zero_inpt[:,50:60,1] = 1
        zero_inpt[:,50:60,2] = 1

        zero_inpt[:,100:110,0] = 1
        zero_inpt[:,100:110,1] = -1
        zero_inpt[:,100:110,2] = 1

        zero_inpt[:,150:160,0] = 1
        zero_inpt[:,150:160,1] = 1
        zero_inpt[:,150:160,2] = -1

        zero_inpt[:,200:210,0] = -1
        zero_inpt[:,200:210,1] = -1
        zero_inpt[:,200:210,2] = 1

        zero_inpt[:,250:260,0] = -1
        zero_inpt[:,250:260,1] = 1
        zero_inpt[:,250:260,2] = -1

        zero_inpt[:,300:310,0] = 1
        zero_inpt[:,300:310,1] = -1
        zero_inpt[:,300:310,2] = -1

        zero_inpt[:,350:360,0] = -1
        zero_inpt[:,350:360,1] = -1
        zero_inpt[:,350:360,2] = -1
        x = tf.placeholder(tf.float32, [64, self.time, 3], name='input_placeholder')
        # init = tf.placeholder(tf.float32, [fps.xstar.z.shape[0],fps.xstar.z.shape[1]], name='input_placeholder')
        
        a = slice(0,1)
        v = np.array(np.repeat(fps.xstar.v[a,:],64,axis=0))
        v = np.array(v + 1e-02*np.random.randn(v.shape[0],v.shape[1]),dtype= np.float32 )
        init_state = ALIFStateTuple(v = tf.convert_to_tensor(v),
                              z = tf.convert_to_tensor(np.repeat(fps.xstar.z[a,:],64,axis=0)),
                              b = tf.convert_to_tensor(np.repeat(fps.xstar.b[a,:],64,axis=0)),
                              i_future_buffer = tf.convert_to_tensor(np.repeat(fps.xstar.i_future_buffer[a,:],64,axis=0)),
                              z_buffer = tf.convert_to_tensor(np.repeat(fps.xstar.z_buffer[a,:],64,axis=0)))
        
        # h = []
        # h.append(init)
        # for i in range(1, rnn.time):
        #     h_x, state= rnn.cell(x[i-1,:,:],h[i-1])
        #     h.append(h_x)
        z = []
        v = []
        b = []
        states = []
        state_inst = init_state
        for t_idx in range(self.time):

          out, state_inst = self.graph['cell'](x[:,t_idx,:],state_inst)
          z_,v_,b_ = out
          z.append(z_)
          v.append(v_)
          b.append(b_)
          states.append(state_inst)
        z = tf.stack(z,axis=1)
        z = tf.convert_to_tensor(z)
        v = tf.stack(v,axis=1)
        v = tf.convert_to_tensor(v)
        bias = tf.stack(b,axis=1)
        bias = tf.convert_to_tensor(bias)

        hids, voltage, bias = sess.run([z,v,bias], feed_dict={x:zero_inpt})
        # plot_fps(fps,
        #     np.array(z),
        #     plot_batch_idx=range(20),
        #     plot_start_time=100,is_lif=True,plot_points=False)
        # plt.show()
        return hids, voltage, bias
    # rnn.reload_from_checkpoints('/content',create_model=True)
    # transition_graph(rnn)