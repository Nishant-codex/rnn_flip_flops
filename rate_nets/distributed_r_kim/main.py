#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Name: Robert Kim
# Date: Oct. 11, 2019
# Email: rkim@salk.edu
# Description: main script for training continuous-variable rate RNN models 
# For more info, refer to 
# Kim R., Li Y., & Sejnowski TJ. Simple Framework for Constructing Functional Spiking 
# Recurrent Neural Networks. Preprint at BioRxiv 
# https://www.biorxiv.org/content/10.1101/579706v2 (2019).

import os, sys
import time
import scipy.io
import numpy as np
import tensorflow as tf
import argparse
import datetime
import pickle

#For Distributed Training 
import horovod.tensorflow as hvd
from hpc4neuro.errors import MpiInitError
from hpc4neuro.distribution import DataDistributor
import mpi4py

# Import utility functions
from utils import set_gpu
from utils import restricted_float
from utils import str2bool

#import Fixedpoint Finder
from FixedPointStore import FixedPointStore
from FixedPointSearch import FixedPointSearch 
# Import the continuous rate model
from model import FR_RNN_dale
from model import find_fps
from model import cell_rate

# Import the tasks
# from model import generate_input_stim_xor
# from model import generate_target_continuous_xor

# from model import generate_input_stim_mante
# from model import generate_target_continuous_mante

# from model import generate_input_stim_go_nogo
# from model import generate_target_continuous_go_nogo
from model import generate_flip_flop_trial
from model import construct_tf
from model import loss_op

# Parse input arguments
parser = argparse.ArgumentParser(description='Training rate RNNs')
parser.add_argument('--gpu', required=False,
        default='0', help="Which gpu to use")
parser.add_argument("--gpu_frac", required=False,
        type=restricted_float, default=0.4,
        help="Fraction of GPU mem to use")
parser.add_argument("--n_trials", required=True,
        type=int, default=200, help="Number of epochs")
parser.add_argument("--mode", required=True,
        type=str, default='Train', help="Train or Eval")
parser.add_argument("--output_dir", required=True,
        type=str, help="Model output path")
parser.add_argument("--N", required=True,
        type=int, help="Number of neurons")
parser.add_argument("--gain", required=False,
        type=float, default = 1.5, help="Gain for the connectivity weight initialization")
parser.add_argument("--P_inh", required=False,
        type=restricted_float, default = 0.20,
        help="Proportion of inhibitory neurons")
parser.add_argument("--P_rec", required=False,
        type=restricted_float, default = 0.20,
        help="Connectivity probability")
parser.add_argument("--som_N", required=True,
        type=int, default = 0, help="Number of SST neurons")
parser.add_argument("--task", required=True,
        type=str, help="Task (XOR, sine, etc...)")
parser.add_argument("--act", required=True,
        type=str, default='sigmoid', help="Activation function (sigmoid, clipped_relu)")
parser.add_argument("--loss_fn", required=True,
        type=str, default='l2', help="Loss function (either L1 or L2)")
parser.add_argument("--apply_dale", required=True,
        type=str2bool, default='True', help="Apply Dale's principle?")
parser.add_argument("--decay_taus", required=True,
        nargs='+', type=float,
        help="Synaptic decay time-constants (in time-steps). If only one number is given, then all\
        time-constants set to that value (i.e. not trainable). Otherwise specify two numbers (min, max).")
args = parser.parse_args()

# Set up the output dir where the output model will be saved
out_dir = os.path.join(args.output_dir, 'models', args.task.lower())
if args.apply_dale == False:
    out_dir = os.path.join(out_dir, 'NoDale')
if len(args.decay_taus) > 1:
    out_dir = os.path.join(out_dir, 'P_rec_' + str(args.P_rec) + '_Taus_' + str(args.decay_taus[0]) + '_' + str(args.decay_taus[1]))
else:
    out_dir = os.path.join(out_dir, 'P_rec_' + str(args.P_rec) + '_Tau_' + str(args.decay_taus[0]))

try:
    if os.path.exists(out_dir) == False:
        os.makedirs(out_dir)
except:
    print('file_exists')
# Number of units/neurons
N = args.N
som_N = args.som_N; # number of SST neurons 

# Define task-specific parameters
# NOTE: Each time step is 5 ms

if args.task.lower() == 'flip':
    # Sensory integration task
    settings = {
            'T': 900, # trial duration (in steps)
            'stim_on': 50, # input stim onset (in steps)
            'stim_dur': 200, # input stim duration (in steps)
            'DeltaT': 1, # sampling rate
            'taus': args.decay_taus, # decay time-constants (in steps)
            'task': args.task.lower(), # task name
            'bits' : 3,
            'batches': 64,
            'rng': np.random.RandomState(400)
            }





'''
Initialize the input and output weight matrices
'''

if args.task.lower() == 'flip':
    w_in = np.float32(settings['rng'].randn(N,3))
    w_out = np.float32(settings['rng'].randn(3, N)/100)



'''
Initialize the continuous rate model
'''
P_inh = args.P_inh # inhibitory neuron proportion
P_rec = args.P_rec # initial connectivity probability (i.e. sparsity degree)
print('P_rec set to ' + str(P_rec))

w_dist = 'gaus' # recurrent weight distribution (Gaussian or Gamma)
net = FR_RNN_dale(N, P_inh, P_rec, w_in, som_N, w_dist, args.gain, args.apply_dale, w_out)


'''
Define the training parameters (learning rate, training termination criteria, etc...)
'''
training_params = {
        'learning_rate': 0.01, # learning rate
        'loss_threshold': 7, # loss threshold (when to stop training)
        'eval_freq': 6000, # how often to evaluate task perf
        'eval_tr': 100, # number of trials for eval
        'eval_amp_threh': 0.7, # amplitude threshold during response window
        'activation': args.act.lower(), # activation function
        'loss_fn': args.loss_fn.lower(), # loss function ('L1' or 'L2')
        'P_rec': 0.20
        }


hvd.init()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())




is_root = hvd.rank() == 0
if is_root: print('Intialized the network...')



'''
Construct the TF graph for training
'''
if args.mode.lower() == 'train':
    input_node, z, x, r, o, w, w_in, m, som_m, w_out, b_out, taus\
            = construct_tf(net, settings, training_params)
    if is_root: print('Constructed the TF graph...')

    # Loss function and optimizer
    loss_op = loss_op(o, z, training_params)
    loss_op.loss_op()

'''
Start the TF session and train the network

'''
hooks = [hvd.BroadcastGlobalVariablesHook(0),

tf.train.StopAtStepHook(last_step = args.n_trials//hvd.size()),

tf.train.LoggingTensorHook(tensors={'step': loss_op.global_step, 
                                  'loss': loss_op.loss},
                         every_n_iter=10) ]




# sess = tf.Session(config=tf.ConfigProto(gpu_options=set_gpu(args.gpu, args.gpu_frac)))
# init = tf.global_variables_initializer()

if args.mode.lower() == 'train':
    with tf.train.MonitoredTrainingSession(checkpoint_dir=out_dir,
                                           hooks=hooks,
                                           config=config,
                                           summary_dir = out_dir+'/tf_logs/') as sess:

        if is_root: print('Training started...')
        # init.run()
        training_success = False

        if args.task.lower() == 'flip':
            # Sensory integration task
            flip = generate_flip_flop_trial(settings)
            u= flip['neural_input']
            target = flip['desired_output']

            x0, r0, w0, w_in0, taus_gaus0 = \
                    sess.run([x, r, w, w_in, taus], feed_dict={input_node: u, z: target})

        # For storing all the loss vals
        losses = np.zeros((args.n_trials,))
        tr = 0
        while not sess.should_stop():
        # for tr in range(args.n_trials):
            start_time = time.time()

            # Generate a task-specific input signal
            if args.task.lower() == 'go-nogo':
                u, label = generate_input_stim_go_nogo(settings)
                target = generate_target_continuous_go_nogo(settings, label)
            elif args.task.lower() == 'xor':
                u, label = generate_input_stim_xor(settings)
                target = generate_target_continuous_xor(settings, label)
            elif args.task.lower() == 'mante':
                u, label = generate_input_stim_mante(settings)
                target = generate_target_continuous_mante(settings, label)
            elif args.task.lower() == 'flip':
                flip = generate_flip_flop_trial(settings)
                u= flip['neural_input']
                target = flip['desired_output']
            if is_root : print("Trial " + str(tr) )#+ ': ' + str(label))
            # sess.run(loss_op.global_step)
            # Train using backprop
            _, t_loss, t_w, t_o, t_w_out, t_x, t_r, t_m, t_som_m, t_w_in, t_b_out, t_taus_gaus = \
                    sess.run([loss_op.training_op, loss_op.loss, w, o, w_out, x, r, m, som_m, w_in, b_out, taus],
                    feed_dict={input_node: u, z: target})

            if is_root: print('Loss: ', t_loss)
            losses[tr] = t_loss
            tr +=1
            if t_loss<0.02:
                break


        elapsed_time = time.time() - start_time
        if is_root: print(elapsed_time)

        # Save the trained params in a .mat file
        var = {}
        var['x0'] = x0
        var['r0'] = r0
        var['w0'] = w0
        var['taus_gaus0'] = taus_gaus0
        var['w_in0'] = w_in0
        var['u'] = u
        var['o'] = t_o
        var['w'] = t_w
        var['x'] = t_x #np.array(t_x).reshape(settings['T'],32,N)
        var['target'] = target
        var['w_out'] = t_w_out
        var['r'] = np.array(t_r).reshape(settings['T'],32,N)
        var['m'] = t_m
        var['som_m'] = t_som_m
        var['N'] = N
        var['exc'] = net.exc
        var['inh'] = net.inh
        var['w_in'] = t_w_in
        var['b_out'] = t_b_out
        var['som_N'] = som_N
        var['losses'] = losses
        var['taus'] = settings['taus']
        # var['eval_perf_mean'] = eval_perf_mean
        # var['eval_loss_mean'] = eval_loss_mean
        # var['eval_os'] = eval_os
        # var['eval_labels'] = eval_labels
        var['taus_gaus'] = t_taus_gaus
        var['tr'] = tr
        var['activation'] = training_params['activation']
        fname_time = datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")
        # if len(settings['taus']) > 1:
        #     fname = 'Task_{}_N_{}_Taus_{}_{}_Act_{}_{}'.format(args.task.lower(), N, settings['taus'][0], 
        #             settings['taus'][1], training_params['activation'], fname_time)
        # elif len(settings['taus']) == 1:
        #     fname = 'Task_{}_N_{}_Tau_{}_Act_{}_{}'.format(args.task.lower(), N, settings['taus'][0], 
        #             training_params['activation'], fname_time)
        fname = str(args.task.lower())+'.pkl'
        
        if is_root: pickle.dump(var, open(os.path.join(out_dir, fname), 'wb'))
        # find_fps(settings)
        # scipy.io.savemat(os.path.join(out_dir, fname), var)


elif args.mode.lower() == 'fps':
    find_fps(settings)