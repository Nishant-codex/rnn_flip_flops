#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Name: Robert Kim
# Date: October 11, 2019
# Email: rkim@salk.edu
# Description: Implementation of the continuous rate RNN model

import os, sys
import numpy as np
import tensorflow as tf
import scipy.io
import pickle



np.random.seed(700)

# import horovod.tensorflow as hvd
# from hpc4neuro.errors import MpiInitError
# from hpc4neuro.distribution import DataDistributor
# import mpi4py

# Import utility functions
from utils import set_gpu
from utils import restricted_float
from utils import str2bool
'''
CONTINUOUS FIRING-RATE RNN CLASS
'''

class FR_RNN_dale:
    """
    Firing-rate RNN model for excitatory and inhibitory neurons
    Initialization of the firing-rate model with recurrent connections
    """
    def __init__(self, N, P_inh, P_rec, w_in, som_N, w_dist, gain, apply_dale, w_out):
        """
        Network initialization method
        N: number of units (neurons)
        P_inh: probability of a neuron being inhibitory
        P_rec: recurrent connection probability
        w_in: NxN weight matrix for the input stimuli
        som_N: number of SOM neurons (set to 0 for no SOM neurons)
        w_dist: recurrent weight distribution ('gaus' or 'gamma')
        apply_dale: apply Dale's principle ('True' or 'False')
        w_out: Nx1 readout weights

        Based on the probability (P_inh) provided above,
        the units in the network are classified into
        either excitatory or inhibitory. Next, the
        weight matrix is initialized based on the connectivity
        probability (P_rec) provided above.
        """
        self.N = N
        self.P_inh = P_inh
        self.P_rec = P_rec
        self.w_in = w_in
        self.som_N = som_N
        self.w_dist = w_dist
        self.gain = gain
        self.apply_dale = apply_dale
        self.w_out = w_out

        # Assign each unit as excitatory or inhibitory
        inh, exc, NI, NE, som_inh = self.assign_exc_inh()
        self.inh = inh
        self.som_inh = som_inh
        self.exc = exc
        self.NI = NI
        self.NE = NE

        # Initialize the weight matrix
        self.W, self.mask, self.som_mask = self.initialize_W()

    def assign_exc_inh(self):
        """
        Method to randomly assign units as excitatory or inhibitory (Dale's principle)

        Returns
            inh: bool array marking which units are inhibitory
            exc: bool array marking which units are excitatory
            NI: number of inhibitory units
            NE: number of excitatory units
            som_inh: indices of "inh" for SOM neurons
        """
        # Apply Dale's principle
        if self.apply_dale == True:
            inh = np.random.rand(self.N, 1) < self.P_inh
            exc = ~inh
            NI = len(np.where(inh == True)[0])
            NE = self.N - NI

        # Do NOT apply Dale's principle
        else:
            inh = np.random.rand(self.N, 1) < 0 # no separate inhibitory units
            exc = ~inh
            NI = len(np.where(inh == True)[0])
            NE = self.N - NI

        if self.som_N > 0:
            som_inh = np.where(inh==True)[0][:self.som_N]
        else:
            som_inh = 0

        return inh, exc, NI, NE, som_inh

    def initialize_W(self):
        """
        Method to generate and initialize the connectivity weight matrix, W
        The weights are drawn from either gaussian or gamma distribution.

        Returns
            w: NxN weights (all positive)
            mask: NxN matrix of 1's (excitatory units)
                  and -1's (for inhibitory units)
        NOTE: To compute the "full" weight matrix, simply
        multiply w and mask (i.e. w*mask)
        """
        # Weight matrix
        w = np.zeros((self.N, self.N), dtype = np.float32)
        idx = np.where(np.random.rand(self.N, self.N) < self.P_rec)
        if self.w_dist.lower() == 'gamma':
            w[idx[0], idx[1]] = np.random.gamma(2, 0.003, len(idx[0]))
        elif self.w_dist.lower() == 'gaus':
            w[idx[0], idx[1]] = np.random.normal(0, 1.0, len(idx[0]))
            w = w/np.sqrt(self.N*self.P_rec)*self.gain # scale by a gain to make it chaotic

        if self.apply_dale == True:
            w = np.abs(w)
        
        # Mask matrix
        mask = np.eye(self.N, dtype=np.float32)
        mask[np.where(self.inh==True)[0], np.where(self.inh==True)[0]] = -1

        # SOM mask matrix
        som_mask = np.ones((self.N, self.N), dtype=np.float32)
        if self.som_N > 0:
            for i in self.som_inh:
                som_mask[i, np.where(self.inh==True)[0]] = 0

        return w, mask, som_mask

    def load_net(self, model_dir):
        """
        Method to load pre-configured network settings
        """
        settings = scipy.io.loadmat(model_dir)
        self.N = settings['N'][0][0]
        self.som_N = settings['som_N'][0][0]
        self.inh = settings['inh']
        self.exc = settings['exc']
        self.inh = self.inh == 1
        self.exc = self.exc == 1
        self.NI = len(np.where(settings['inh'] == True)[0])
        self.NE = len(np.where(settings['exc'] == True)[0])
        self.mask = settings['m']
        self.som_mask = settings['som_m']
        self.W = settings['w']
        self.w_in = settings['w_in']
        self.b_out = settings['b_out']
        self.w_out = settings['w_out']

        return self
    
    def display(self):
        """
        Method to print the network setup
        """
        print('Network Settings')
        print('====================================')
        print('Number of Units: ', self.N)
        print('\t Number of Excitatory Units: ', self.NE)
        print('\t Number of Inhibitory Units: ', self.NI)
        print('Weight Matrix, W')
        full_w = self.W*self.mask
        zero_w = len(np.where(full_w == 0)[0])
        pos_w = len(np.where(full_w > 0)[0])
        neg_w = len(np.where(full_w < 0)[0])
        print('\t Zero Weights: %2.2f %%' % (zero_w/(self.N*self.N)*100))
        print('\t Positive Weights: %2.2f %%' % (pos_w/(self.N*self.N)*100))
        print('\t Negative Weights: %2.2f %%' % (neg_w/(self.N*self.N)*100))



def generate_flip_flop_trial(settings):


    bits = settings['bits']
    batch = settings['batches']
    trial_info = {'neural_input': np.zeros([settings['T'],batch, bits]),
                  'desired_output':np.zeros([settings['T'],batch, bits])}    



    unsigned_inp = settings['rng'].binomial(1,0.2,[settings['T']//10,batch, bits])
    unsigned_out = 2*settings['rng'].binomial(1,0.5,[settings['T']//10,batch, bits]) -1 


  
    inputs = unsigned_inp
    inputs = np.multiply(unsigned_inp,unsigned_out)
    inputs[0,:] = 1.0

    inputs = np.repeat(inputs,10,axis=0)

    trial_info['neural_input'] = inputs.T
    # trial_info['neural_input'] = 0.5*trial_info['neural_input']
    output = np.zeros_like(inputs)
    for trial_idx in range(batch):
        for bit_idx in range(bits):
            input_ = np.squeeze(inputs[:,trial_idx,bit_idx])
            t_flip = np.where(input_ != 0)
            for flip_idx in range(np.size(t_flip)):
                # Get the time of the next flip
                t_flip_i = t_flip[0][flip_idx]

                '''Set the output to the sign of the flip for the
                remainder of the trial. Future flips will overwrite future
                output'''
                output[t_flip_i:,trial_idx, bit_idx] = \
                    inputs[t_flip_i,trial_idx,  bit_idx]

    trial_info['desired_output'] = output
    # trial_info['desired_output'] = 0.5*trial_info['desired_output']

    return trial_info


'''
CONSTRUCT TF GRAPH FOR TRAINING

'''
def cell_rate(inputs, states):

    # if fr_rnn.apply_dale == True:
    # # Parametrize the weight matrix to enforce exc/inh synaptic currents

    def restore(path):
        file = open(path,'rb')
        restore_data = file.read()
        file.close()
        # print(type(pickle.loads(restore_data)))
        # print((self.__dict__))
        hid= pickle.loads(restore_data,encoding='latin1')
        return(hid)
    data = restore(os.getcwd()+'/flip.pkl')       

    x = tf.transpose(states)
    r = tf.sigmoid(x)
    taus = data['taus']

    w = tf.nn.relu(data['w'])
    w_in = data['w_in']

    # next_x is [N x 1]
    ww = tf.matmul(w, data['m'])
    ww = tf.multiply(ww, data['som_m'])

    # Pass the synaptic time constants thru the sigmoid function
    if len(taus) > 1:
        taus_sig = tf.sigmoid(taus_gaus)*(taus[1] - taus[0]) + taus[0]
    elif len(taus) == 1: # one scalar synaptic decay time-constant
        taus_sig = taus[0]


    a = tf.multiply((1 - 1/taus_sig), x)
    b = tf.multiply((1/taus_sig), ((tf.matmul(ww, r))\
            + tf.matmul(w_in, tf.transpose(inputs[:,:]))))

    next_x = a + b #+ tf.random_normal(tf.shape(b))/10
    # x.append(next_x)

    return tf.transpose(next_x), tf.transpose(next_x)

def find_fps(settings):
    def restore(path):
        file = open(path,'rb')
        restore_data = file.read()
        file.close()
        # print(type(pickle.loads(restore_data)))
        # print((self.__dict__))
        hid= pickle.loads(restore_data,encoding='latin1')
        return(hid)
    data = restore(os.getcwd()+'/flip.pkl')
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    x = data['x']

    n_bits = settings['bits']

    inputs = np.zeros([1,n_bits])

    fps = FixedPointSearch(
                        'rate',
                        np.array(x),
                        os.getcwd(),
                        cell = cell_rate,
                        sess = sess)
    fps.sample_states(1000,np.array(x),'rate',1.5)
    unique, all_fps = fps.find_fixed_points(inputs, save= True)
    return unique, all_fps





def construct_tf(fr_rnn, settings, training_params):
    """
    Method to construct a TF graph and return nodes with
    Dale's principle
    INPUT
        fr_rnn: firing-rate RNN class
        settings: dict containing the following keys
            T: duration of a single trial (in steps)
            stim_on: stimulus starting time (in steps)
            stim_dur: stimulus duration (in steps)
            delay: delay b/w two stimuli (in steps)
            taus: time-constants (in steps)
            DeltaT: sampling rate
        training_params: dictionary containing training parameters
            learning_rate: learning rate
    OUTPUT
        TF graph
    """

    # Task params
    T = settings['T']
    taus = settings['taus']
    DeltaT = settings['DeltaT']
    task = settings['task']

    # Training params
    learning_rate = training_params['learning_rate']

    # Excitatory units
    exc_idx_tf = tf.constant(np.where(fr_rnn.exc == True)[0], name='exc_idx')

    # Inhibitory units
    inh_idx_tf = tf.constant(np.where(fr_rnn.inh == True)[0], name='inh_idx')
    som_inh_idx_tf = tf.constant(fr_rnn.som_inh, name='som_inh_idx')

    if task == 'flip':
        stim = tf.placeholder(tf.float32, [settings['bits'],settings['batches'], T], name='u')

    # Target node
    z = tf.placeholder(tf.float32, [T,settings['batches'],settings['bits']], name='target')

    # Initialize the decay synaptic time-constants (gaussian random).
    # This vector will go through the sigmoid transfer function.
    if len(taus) > 1:
        taus_gaus = tf.Variable(tf.random_normal([fr_rnn.N, 1]), dtype=tf.float32, 
            name='taus_gaus', trainable=True)
    elif len(taus) == 1:
        taus_gaus = tf.Variable(tf.random_normal([fr_rnn.N, 1]), dtype=tf.float32, 
            name='taus_gaus', trainable=False)
        print('Synaptic decay time-constants will not get updated!')

    # Synaptic currents and firing-rates
    x = [] # synaptic currents
    r = [] # firing-rates
    x_t = []

    x.append(tf.random_normal([fr_rnn.N,settings['batches']], dtype=tf.float32)/100)

    x_t.append(tf.transpose(x[-1]))
    # Transfer function options
    if training_params['activation'] == 'sigmoid':
        r.append(tf.sigmoid(x[0]))
    elif training_params['activation'] == 'clipped_relu': 
        # r.append(tf.clip_by_value(tf.nn.relu(x[0]), 0, 20))
        r.append(tf.nn.relu(x[0]))
    elif training_params['activation'] == 'softplus':
        r.append(tf.clip_by_value(tf.nn.softplus(x[0]), 0, 20))

    # Initialize recurrent weight matrix, mask, input & output weight matrices
    w = tf.get_variable('w', initializer = fr_rnn.W, dtype=tf.float32, trainable=True)
    m = tf.get_variable('m', initializer = fr_rnn.mask, dtype=tf.float32, trainable=False)
    som_m = tf.get_variable('som_m', initializer = fr_rnn.som_mask, dtype=tf.float32,
            trainable=False)
    w_in = tf.get_variable('w_in', initializer = fr_rnn.w_in, dtype=tf.float32, trainable=True)
    w_out = tf.get_variable('w_out', initializer = fr_rnn.w_out, dtype=tf.float32, 
            trainable=True)

    b_out = tf.Variable(0, dtype=tf.float32, name='b_out', trainable=True)

    # Forward pass
    o = [] # output (i.e. weighted linear sum of rates, r)
    for t in range(1, T):
        if fr_rnn.apply_dale == True:
            # Parametrize the weight matrix to enforce exc/inh synaptic currents
            w = tf.nn.relu(w)

        # next_x is [N x 1]
        ww = tf.matmul(w, m)
        ww = tf.multiply(ww, som_m)

        # Pass the synaptic time constants thru the sigmoid function
        if len(taus) > 1:
            taus_sig = tf.sigmoid(taus_gaus)*(taus[1] - taus[0]) + taus[0]
        elif len(taus) == 1: # one scalar synaptic decay time-constant
            taus_sig = taus[0]

        next_x = tf.multiply((1 - DeltaT/taus_sig), x[t-1]) + \
                tf.multiply((DeltaT/taus_sig), ((tf.matmul(ww, r[t-1]))\
                + tf.matmul(w_in, tf.squeeze(stim[:,:, t-1])))) #+\
                # tf.random_normal(tf.shape(x[t-1]), dtype=tf.float32)/10
        x.append(next_x)
        x_t.append(tf.transpose(next_x))
        if training_params['activation'] == 'sigmoid':
            r.append(tf.sigmoid(next_x))
        elif training_params['activation'] == 'clipped_relu': 
            # r.append(tf.clip_by_value(tf.nn.relu(next_x), 0, 20))
            r.append(tf.nn.relu(next_x))
        elif training_params['activation'] == 'softplus':
            r.append(tf.clip_by_value(tf.nn.softplus(next_x), 0, 20))

        next_o = tf.matmul(w_out, r[t]) + b_out
        o.append(tf.transpose(tf.squeeze(next_o)))

    return stim, z, x_t, r, o, w, w_in, m, som_m, w_out, b_out, taus_gaus
    
'''
DEFINE LOSS AND OPTIMIZER



'''



class loss_op:
    def __init__(self, o, z, training_params, hvd):
        self.o  = o
        self.z = z
        self.global_step = tf.train.get_or_create_global_step()
        self.training_params = training_params 
        self.hvd = hvd
    def loss_op(self):
        """
        Method to define loss and optimizer for ONLY ONE target signal
        INPUT
            o: list of output values
            z: target values
            training_params: dictionary containing training parameters
                learning_rate: learning rate

        OUTPUT
            loss: loss function
            training_op: optimizer
        """
        # Loss function
        # print(z.shape)
        # print(tf.stack(tf.squeeze(o)))
        loss = tf.zeros(1)
        # print(o[0])
        loss_fn = self.training_params['loss_fn']
        # for i in range(0, len(o)):
        #     if loss_fn.lower() == 'l1':
        #         loss += tf.norm(o[i] - z[i])
        #     elif loss_fn.lower() == 'l2':
        #         loss += tf.reduce_sum(tf.squared_difference(o[i], z[i]))
        # if loss_fn.lower() == 'l2':
        #     loss = tf.sqrt(loss)
        self.loss = tf.reduce_mean(tf.squared_difference(self.o, self.z[:-1,:]))
        # Optimizer function
        with tf.name_scope('ADAM'):
            optimizer = tf.train.AdamOptimizer(learning_rate = self.training_params['learning_rate']*self.hvd.size()) 

            # # decay = tf.train.exponential_decay(self.training_params['learning_rate'], self.global_step, 128, 0.9)
            # # optimizer = tf.train.MomentumOptimizer(decay*hvd.size(), 0.9)
            # # optimizer = tf.train.AdamOptimizer(decay*hvd.size(),epsilon=1e-1)
            # optimizer = self.hvd.DistributedOptimizer( optimizer)
            # # gradients, variables = zip(*optimizer.compute_gradients(self.loss,tf.trainable_variables())) 
            # # gradients = [None if gradient is None else tf.clip_by_norm(gradient, 0.1) for gradient in gradients]
            # # self.training_op = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)


            self.training_op = optimizer.minimize(self.loss,global_step=self.global_step) 

        return self.loss, self.training_op

'''
EVALUATE THE TRAINED MODEL
NOTE: NEED TO BE UPDATED!!
'''
def eval_tf(model_dir, settings, u):
    """
    Method to evaluate a trained TF graph
    INPUT
        model_dir: full path to the saved model .mat file
        stim_params: dictionary containig the following keys
        u: 12xT stimulus matrix
            NOTE: There are 12 rows (one per dot pattern): 6 cues and 6 probes.
    OUTPUT
        o: 1xT output vector
    """
    T = settings['T']
    stim_on = settings['stim_on']
    stim_dur = settings['stim_dur']
    delay = settings['delay']
    DeltaT = settings['DeltaT']

    # Load the trained mat file
    var = scipy.io.loadmat(model_dir)

    # Get some additional params
    N = var['N'][0][0]
    exc_ind = [np.bool(i) for i in var['exc']]

    # Get the delays
    taus_gaus = var['taus_gaus']
    taus = var['taus'][0] # tau [min, max]
    taus_sig = (1/(1+np.exp(-taus_gaus))*(taus[1] - taus[0])) + taus[0] 

    # Synaptic currents and firing-rates
    x = np.zeros((N, T)) # synaptic currents
    r = np.zeros((N, T)) # firing-rates
    x[:, 0] = np.random.randn(N, )/100
    r[:, 0] = 1/(1 + np.exp(-x[:, 0]))
    # r[:, 0] = np.minimum(np.maximum(x[:, 0], 0), 1) #clipped relu
    # r[:, 0] = np.clip(np.minimum(np.maximum(x[:, 0], 0), 1), None, 10) #clipped relu
    # r[:, 0] = np.clip(np.log(np.exp(x[:, 0])+1), None, 10) # softplus
    # r[:, 0] = np.minimum(np.maximum(x[:, 0], 0), 6)/6 #clipped relu6


    # Output
    o = np.zeros((T, ))
    o_counter = 0

    # Recurrent weights and masks
    # w = var['w0'] #!!!!!!!!!!!!
    w = var['w']

    m = var['m']
    som_m = var['som_m']
    som_N = var['som_N'][0][0]

    # Identify excitatory/inhibitory neurons
    exc = var['exc']
    exc_ind = np.where(exc == 1)[0]
    inh = var['inh']
    inh_ind = np.where(inh == 1)[0]
    som_inh_ind = inh_ind[:som_N]

    for t in range(1, T):
        # next_x is [N x 1]
        ww = np.matmul(w, m)
        ww = np.multiply(ww, som_m)

        # next_x = (1 - DeltaT/tau)*x[:, t-1] + \
                # (DeltaT/tau)*(np.matmul(ww, r[:, t-1]) + \
                # np.matmul(var['w_in'], u[:, t-1])) + \
                # np.random.randn(N, )/10

        next_x = np.multiply((1 - DeltaT/taus_sig), np.expand_dims(x[:, t-1], 1)) + \
                np.multiply((DeltaT/taus_sig), ((np.matmul(ww, np.expand_dims(r[:, t-1], 1)))\
                + np.matmul(var['w_in'], np.expand_dims(u[:, t-1], 1)))) +\
                np.random.randn(N, 1)/10

        x[:, t] = np.squeeze(next_x)
        r[:, t] = 1/(1 + np.exp(-x[:, t]))
        # r[:, t] = np.minimum(np.maximum(x[:, t], 0), 1)
        # r[:, t] = np.clip(np.minimum(np.maximum(x[:, t], 0), 1), None, 10)
        # r[:, t] = np.clip(np.log(np.exp(x[:, t])+1), None, 10) # softplus
        # r[:, t] = np.minimum(np.maximum(x[:, t], 0), 6)/6


        wout = var['w_out']
        wout_exc = wout[0, exc_ind]
        wout_inh = wout[0, inh_ind]
        r_exc = r[exc_ind, :]
        r_inh = r[inh_ind, :]

        o[o_counter] = np.matmul(wout, r[:, t]) + var['b_out']
        # o[o_counter] = np.matmul(wout_exc, r[exc_ind, t]) + var['b_out'] # excitatory output
        # o[o_counter] = np.matmul(wout_inh, r[inh_ind, t]) + var['b_out'] # inhibitory output
        o_counter += 1
    return x, r, o

