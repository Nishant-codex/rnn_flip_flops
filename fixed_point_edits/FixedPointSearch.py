from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
#import os
import sys 

sys.path.insert(0,'/home/joshi/fixed_point_edits')
import os
import absl
from tensorflow.python.ops import parallel_for as pfor
from FixedPointStore import *
import tensorflow as tf

# import horovod.tensorflow as hvd

#import cProfile
# %tensorflow_version 1.x magic
#import matplotlib.pyplot as plt

import numpy.random as nrand

np.random.seed(0)
# import numpy as np
import time
from AdaptiveGradNormClip import AdaptiveGradNormClip
from AdaptiveLearningRate import AdaptiveLearningRate

class FixedPointSearch:

  def __init__(self, 
              ctype, 
              states,
              savepath, 
              cell=None,
              sess=None,
              max_iters = 5000,
              max_n_unique = np.inf,
              tol_q = 1e-12,
              tol_dq = 1e-20,
              adaptive_learning_rate_hps = {},
              grad_norm_clip_hps = {},
              adam_optimizer_hps = {'epsilon': 0.01},
              exclude_dis_outliers = True,
              outlier_distance_scale = 10.0,
              rerun_q_outliers = True,
              run_additional_iterations_on_outliers = True,
              outlier_q_scale = 10.0
              ):

    self.max_iters = max_iters 
    self.ctype = ctype
    self.dtype = np.float32
    self.tol_q = tol_q
    self.savepath = savepath
    self.tol_dq = tol_dq
    self.adaptive_learning_rate_hps = adaptive_learning_rate_hps
    self.grad_norm_clip_hps =grad_norm_clip_hps
    self.adam_optimizer_hps = adam_optimizer_hps 
    self.outlier_q_scale = outlier_q_scale
    self.outlier_distance_scale = outlier_distance_scale
    self.states = states
    self.bits = 3
    self.max_n_unique = max_n_unique
    self.rerun_q_outliers = rerun_q_outliers
    self.sampled_states = 0
    self.cell = cell
    self.is_root = False
    self.uniq_tol = 1e-3
    self.decompose_jacobians = True
    self.compute_jacobians = True
    self.sess = sess
    self.exclude_dis_outliers = exclude_dis_outliers
    self.run_additional_iterations_on_outliers = run_additional_iterations_on_outliers

  def convert_from_lstm_tuples(self, lstm):
    c = lstm.c
    h = lstm.h
    # print(c.shape)
    rank = len(lstm.c.shape)
    axis = rank -1 
    if(tf.is_numeric_tensor(c)):
      return tf.concat((c,h),axis=axis)
    else:
      return np.concatenate((c,h),axis=axis)

  def convert_to_lstm_tuples(self, lstm):

    array = lstm
    rank = len(array.shape)
    dim = array.shape[rank-1]
    if dim%2 ==0:
      conc_dim = dim//2
    else:
      raise ValueError("Dimentions are not even")

    if rank == 3:
      c = array[:,:,:conc_dim]
      h = array[:,:,conc_dim:]
    elif rank == 2:
      c = array[:,:conc_dim]
      h = array[:,conc_dim:]

    return tf.nn.rnn_cell.LSTMStateTuple(c=c,h=h)

  def build_vars(self, init_states):
    if self.ctype == 'LSTM':
      c_h_init = self.convert_from_lstm_tuples(init_states)
      x = tf.Variable(c_h_init,dtype=tf.float32)
      x_rnn_cell = self.convert_to_lstm_tuples(x)
    else:
      x = tf.Variable(init_states,dtype=tf.float32)
      x_rnn_cell = x
    return x,x_rnn_cell

  def maybe_convert(self, x_init):
    if self.ctype=='LSTM':
      return self.convert_from_lstm_tuples(x_init)
    else:
      return x_init
  
  def get_rnn(self, init_states, inputs):
    # print('inside get rnn')
    x, x_rnn = self.build_vars(init_states)
    inputs = tf.constant(inputs,dtype=tf.float32)
    # print('before cell')
    output, F_rnn = self.cell(inputs,x_rnn)
    # print('before cell')
    if self.ctype == 'LSTM':
      F = self.convert_from_lstm_tuples(F_rnn)
    else:
      F = F_rnn

    init = tf.variables_initializer(var_list=[x])
    self.sess.run(init)
    return x, F

  def compute_input_jacobians(self, fps):
    def grab_RNN_for_dFdu(initial_states, inputs):
      
      x, x_rnn = self.build_vars(initial_states)
      
      inputs = tf.Variable(inputs,dtype=tf.float32)
      
      output, F_rnn = self.cell(inputs,x_rnn)

      if self.ctype == 'LSTM':
        F = self.convert_from_lstm_tuples(F_rnn)
      else:
        F = F_rnn
      
      init = tf.variables_initializer(var_list = [x, inputs])
      self.sess.run(init)

      return inputs, F

    inputs_np = fps.inputs

    if self.ctype == 'LSTM':
      states_np = self.convert_to_lstm_tuples(fps.xstar)
    else:
      states_np = fps.xstar

    inputs, F_tf = grab_RNN_for_dFdu(states_np, inputs_np)

    try: 
      J_tf = pfor.batch_jacobian(F_tf, inputs)
    except absl.flags._exceptions.UnparsedFlagAccessError:
      J_tf = pfor.batch_jacobian(F_tf, inputs_tf, use_pfor=False)

    J_np = self.sess.run(J_tf)

    return J_np, J_tf

     
  def compute_recurrent_jacobians(self, fps):

    inputs = fps.inputs
    if self.ctype == 'LSTM':
      # print('line2')
    
      states_np = self.convert_to_lstm_tuples(fps.xstar)
      # print('line3')
    else:
      # print('line4')
    
      states_np = fps.xstar
    
    # print('line6')

    x_tf,F_tf = self.get_rnn(states_np,inputs)
    # print('line5')

    try: 
      if self.is_root:
        print('batch jacobians')
      J_tf = pfor.batch_jacobian(F_tf,x_tf)
    except absl.flags._exceptions.UnparsedFlagAccessError:
      J_tf = pfor.batch_jacobian(F_tf, x_tf, use_pfor=False)
    if self.is_root:
      print('running cells')
    J_np = self.sess.run(J_tf)
    if self.is_root:
      print('out of batch jacobians')
    return J_np, J_tf
  def _get_valid_mask(self, n_batch, n_time, valid_bxt =None):
      if valid_bxt is None:
          valid_bxt = np.ones((n_batch, n_time), dtype=np.bool)
      else:

          assert (valid_bxt.shape[0] == n_batch and
              valid_bxt.shape[1] == n_time),\
              ('valid_bxt.shape should be %s, but is %s'
               % ((n_batch, n_time), valid_bxt.shape))

          if not valid_bxt.dtype == np.bool:
              valid_bxt = valid_bxt.astype(np.bool)

      return valid_bxt
  def sample_states(self, init_size, state_matrix,c_type, noise):

    if c_type =='LSTM':
      matrix = self.convert_from_lstm_tuples(state_matrix)
    else:
      matrix = state_matrix
    
    [n_batch, n_time, n_states] = matrix.shape

    # valid_bxt = self._get_valid_mask(n_batch, n_time, valid_bxt = None)
    valid_idx = np.ones((n_batch, n_time), dtype=np.bool)

    (trial_idx, time_idx) = np.nonzero(valid_idx)

    # min_index = min(len(trial_idx),len(time_idx))
    max_sample_index = len(trial_idx)
    sample_indices = nrand.RandomState(200).randint(0, high = max_sample_index, size = init_size)
  
    states = np.zeros([init_size, n_states])

    for i in range(init_size):
      init_idx = trial_idx[i]
      t_idx = time_idx[i]
      states[i,:] = matrix[init_idx,t_idx,:]

    if noise>0.0:
      states = states + noise*np.random.randn(*states.shape)

    if c_type == 'LSTM':
      # print('this')
      self.sampled_states = self.convert_to_lstm_tuples(states)
    else:
      self.sampled_states = states

  def identify_distance_non_outliers(self, fps, initial_states, dist_thresh):
    if self.ctype == 'LSTM':
        initial_states = self.convert_from_lstm_tuples(initial_states)

    num_inits = initial_states.shape[0]
    n_fps = fps.num_inits

    # Centroid of initial_states, shape (n_states,)
    centroid = np.mean(initial_states, axis=0)

    # Distance of each initial state from the centroid, shape (n,)
    init_dists = np.linalg.norm(initial_states - centroid, axis=1)
    avg_init_dist = np.mean(init_dists)

    # Normalized distances of initial states to the centroid, shape: (n,)
    scaled_init_dists = np.true_divide(init_dists, avg_init_dist)

    # Distance of each FP from the initial_states centroid
    fps_dists = np.linalg.norm(fps.xstar - centroid, axis=1)

    # Normalized
    scaled_fps_dists = np.true_divide(fps_dists, avg_init_dist)

    init_non_outlier_idx = np.where(scaled_init_dists < dist_thresh)[0]
    n_init_non_outliers = init_non_outlier_idx.size
    if self.is_root:
      print('\t\tinitial_states: %d outliers detected (of %d).'
          % (num_inits - n_init_non_outliers, num_inits))

    fps_non_outlier_idx = np.where(scaled_fps_dists < dist_thresh)[0]
    n_fps_non_outliers = fps_non_outlier_idx.size
    if self.is_root:
      print('\t\tfixed points: %d outliers detected (of %d).'
          % (n_fps - n_fps_non_outliers, n_fps))

    return fps_non_outlier_idx


  def exclude_dis_outliers_(self, fps, initial_states):
    idx_keep = self.identify_distance_non_outliers(fps, initial_states, self.outlier_distance_scale)
    return fps[idx_keep]

  def identify_q_outliers(self, fps, q_thresh):

    return np.where(fps.qstar > q_thresh)[0]

  def _get_rnncell_compatible_states(self, states):

    if self.ctype == 'LSTM':
        return self.convert_to_lstm_tuples(states)
    else:
        return states

  def run_additional_iterations_on_outliers_(self, fps):

    def perform_outlier_optimization(fps, method):

      idx_outliers = self.identify_q_outliers(fps, outlier_min_q)
      n_outliers = len(idx_outliers)

      outlier_fps = fps[idx_outliers]
      n_prev_iters = outlier_fps.n_iters
      inputs = outlier_fps.inputs
      initial_states = self._get_rnncell_compatible_states(
          outlier_fps.xstar)

      if method == 'sequential':

          updated_outlier_fps = self.run_sequential_optimization(
              initial_states, inputs, q_prior=outlier_fps.qstar)
      elif method == 'joint':
          updated_outlier_fps = self.run_joint_optimization(initial_states, inputs)
      else:
          raise ValueError('Unsupported method: %s.' % method)

      updated_outlier_fps.n_iters += n_prev_iters
      fps[idx_outliers] = updated_outlier_fps

      return fps

    def outlier_update(fps):

      idx_outliers = self.identify_q_outliers(fps, outlier_min_q)
      n_outliers = len(idx_outliers)

      # self._print_if_verbose('\n\tDetected %d putative outliers '
      #                        '(q>%.2e).' % (n_outliers, outlier_min_q))

      return idx_outliers

    outlier_min_q = np.median(fps.qstar)*self.outlier_q_scale
    idx_outliers = outlier_update(fps)

    if len(idx_outliers) == 0:
        return fps


    fps = perform_outlier_optimization(fps, 'sequential')
    outlier_update(fps) # For print output only

    return fps


  def run_iteration_loops(self, states, inputs, init_array):

    x, F_cell = self.get_rnn(states, inputs)
    q = 0.5 * tf.reduce_sum(tf.square(F_cell - x ))

    q_scalar = tf.reduce_mean(q)
    grads = tf.gradients(q_scalar, [x])

    q_prev_tf = tf.placeholder(tf.float32, shape=list(q.shape), name='q_prev')

    # when (q-q_prev) is negative, optimization is making progress
    dq = tf.abs(q - q_prev_tf)
    hps={}

    # Optimizer
    adaptive_learning_rate = AdaptiveLearningRate(**self.adaptive_learning_rate_hps)
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    adaptive_grad_norm_clip = AdaptiveGradNormClip(**self.grad_norm_clip_hps)
    grad_norm_clip_val = tf.placeholder(tf.float32, name='grad_norm_clip_val')

    # Gradient clipping
    clipped_grads, grad_global_norm = tf.clip_by_global_norm(grads, grad_norm_clip_val)
    clipped_grad_global_norm = tf.global_norm(clipped_grads)
    clipped_grad_norm_diff = grad_global_norm - clipped_grad_global_norm
    grads_to_apply = clipped_grads

    # adam_hps = {'epsilon': 0.01}
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, **self.adam_optimizer_hps)
    # optimizer = hvd.DistributedOptimizer(optimizer)
    train = optimizer.apply_gradients(zip(grads_to_apply, [x]))

    # Initialize x and AdamOptimizer's auxiliary variables
    uninitialized_vars = optimizer.variables()
    init = tf.variables_initializer(var_list=uninitialized_vars)
    self.sess.run(init)

    ops_to_eval = [train,x, F_cell, q_scalar, q, dq, grad_global_norm]

    iter_count = 1
    t_start = time.time()
    q_prev = np.tile(np.nan, q.shape.as_list())
    rnn_cell_feed_dict = {}
    while True:
        # print('inside run iter loops')
        iter_learning_rate = adaptive_learning_rate()
        iter_clip_val = adaptive_grad_norm_clip()

        feed_dict = {learning_rate: iter_learning_rate,
                      grad_norm_clip_val: iter_clip_val,
                      q_prev_tf: q_prev}
        feed_dict.update(rnn_cell_feed_dict)

        (ev_train,
        ev_x,
        ev_F,
        ev_q_scalar,
        ev_q,
        ev_dq,
        ev_grad_norm) = self.sess.run(ops_to_eval, feed_dict)

        # print('doing iter count')
        if iter_count > 1 and \
            np.all(np.logical_or(
                ev_dq < self.tol_dq*iter_learning_rate,
                ev_q < self.tol_q)):
            if self.is_root:
              print('\tOptimization complete to desired tolerance.')
            break

        if iter_count + 1 > 5000:
          if self.is_root:
            print('\tMaximum iteration count reached. '
                                    'Terminating.')
          break

        q_prev = ev_q
        adaptive_learning_rate.update(ev_q_scalar)
        adaptive_grad_norm_clip.update(ev_grad_norm)
        iter_count += 1
    # print('outside the loop')
    iter_count = np.tile(iter_count, ev_q.shape)
    fixed_point = FixedPointStore(xstar = ev_x,
                                  inputs = inputs,
                                  dtype = self.dtype,
                                  alloc_zeros = False, 
                                  x_init = self.maybe_convert(states),
                                  F_xstar=ev_F, 
                                  qstar= ev_q,
                                  dq=ev_dq,
                                  n_iters = iter_count
                                  )
    return fixed_point

  def find_shape(self, states):
    if self.ctype == 'LSTM': 
      return (states.c.shape[0], states.c.shape[1]*2)
    else:
      return states.shape[0],states.shape[1]

  def return_index(self, states, index):
    if self.ctype=='LSTM':
      c= states.c[index]
      h = states.h[index]
      return tf.nn.rnn_cell.LSTMStateTuple(c=c,h=h)
    else: 
      return states[index]

  def run_joint_optimization(self, initial_states, inputs):

      n, _ = self.find_shape(initial_states)

      x, F = self.get_rnn(initial_states, inputs)

      # A shape [n,] TF Tensor of objectives (one per initial state) to be
      # combined in _run_optimization_loop.
      q = 0.5 * tf.reduce_sum(tf.square(F - x), axis=1)
      
      q_scalar = tf.reduce_mean(q)

      grads = tf.gradients(q_scalar, [x])

      q_prev_tf = tf.placeholder(tf.float32, 
                                shape=list(q.shape), 
                                name='q_prev')

      # when (q-q_prev) is negative, optimization is making progress
      dq = tf.abs(q - q_prev_tf)
      hps={}


      # Optimizer
      adaptive_learning_rate = AdaptiveLearningRate(**self.adaptive_learning_rate_hps)
      learning_rate = tf.placeholder(tf.float32, name='learning_rate')

      adaptive_grad_norm_clip = AdaptiveGradNormClip(**self.grad_norm_clip_hps)
      grad_norm_clip_val = tf.placeholder(tf.float32, name='grad_norm_clip_val')

      # Gradient clipping
      clipped_grads, grad_global_norm = tf.clip_by_global_norm(grads, grad_norm_clip_val)
      clipped_grad_global_norm = tf.global_norm(clipped_grads)
      clipped_grad_norm_diff = grad_global_norm - clipped_grad_global_norm
      grads_to_apply = clipped_grads

      # adam_hps = {'epsilon': 0.01}
      optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate , **self.adam_optimizer_hps)
      # optimizer = hvd.DistributedOptimizer(optimizer)#* hvd.size()
      train = optimizer.apply_gradients(zip(grads_to_apply, [x]))

      uninitialized_vars = optimizer.variables()
      init = tf.variables_initializer(var_list=uninitialized_vars)
      self.sess.run(init)

      ops_to_eval = [train,x, F, q_scalar, q, dq, grad_global_norm]

      iter_count = 1
      t_start = time.time()
      q_prev = np.tile(np.nan, q.shape.as_list())
      rnn_cell_feed_dict = {}
      while True:
          # print('inside run iter loops')
          iter_learning_rate = adaptive_learning_rate()
          iter_clip_val = adaptive_grad_norm_clip()

          feed_dict = {learning_rate: iter_learning_rate,
                        grad_norm_clip_val: iter_clip_val,
                        q_prev_tf: q_prev}
          feed_dict.update(rnn_cell_feed_dict)

          (ev_train,
          ev_x,
          ev_F,
          ev_q_scalar,
          ev_q,
          ev_dq,
          ev_grad_norm) = self.sess.run(ops_to_eval, feed_dict)

          # if self.super_verbose and \
          #     np.mod(iter_count, self.n_iters_per_print_update)==0:
          #     print_update(iter_count, ev_q, ev_dq, iter_learning_rate)
          # print('doing iter count')
          if iter_count > 1 and \
              np.all(np.logical_or(
                  ev_dq < self.tol_dq*iter_learning_rate,
                  ev_q < self.tol_q)):
              '''Here dq is scaled by the learning rate. Otherwise very
              small steps due to very small learning rates would spuriously
              indicate convergence. This scaling is roughly equivalent to
              measuring the gradient norm.'''
              if self.is_root:
                print('\tOptimization complete to desired tolerance.')
              break

          if iter_count + 1 > self.max_iters:
              if self.is_root:            
                print('\tMaximum iteration count reached. '
                                        'Terminating.')
              break

          q_prev = ev_q
          adaptive_learning_rate.update(ev_q_scalar)
          adaptive_grad_norm_clip.update(ev_grad_norm)
          iter_count += 1
      # print('outside the loop')
      iter_count = np.tile(iter_count, ev_q.shape)
      fixed_point = FixedPointStore(
          # num_states = init_array['num_states'],
          #                       num_inits = init_array['num_inits'], 
          #                       num_inputs = init_array['num_inputs'], 
                                xstar = ev_x,
                                alloc_zeros = False, 
                                dtype =self.dtype,
                                x_init = self.maybe_convert(initial_states),
                                inputs = inputs,
                                F_xstar=ev_F, 
                                qstar= ev_q,
                                dq=ev_dq,
                                n_iters = iter_count
                                )

      return fixed_point


  def run_sequential_optimization(self, states, inputs, q_prior = None):
    if self.is_root:
      print('running sequential optimization')
    num_inits, num_states = self.find_shape(states) 
    num_inputs = inputs.shape[1]

    fresh_start = q_prior is None
    if self.is_root:

      print('fresh_start ', fresh_start)
    fps = FixedPointStore(num_inits=num_inits, num_states=num_states, num_inputs=num_inputs, alloc_zeros=True)

    init_dict = {'num_inits':num_inits,'num_states':num_states,'num_inputs':num_inputs}
    
    for i in range(num_inits):
      index = slice(i, i+1)
      state_inst_i  = self.return_index(states, index)
      print(type(state_inst_i))
      input_inst_i  = inputs[index, :]
      if self.is_root : print('state number ',i)
      if fresh_start and i == 0 :
        if self.is_root:
          print('Starting to find the fixed points')


      elif fresh_start==False: 
        if self.is_root:
          print('running iterations over q again')
      
      fps[index] = self.run_iteration_loops(state_inst_i, input_inst_i, init_dict)

    return fps

  def find_fixed_points(self, inputs, save=False):
    
    # hvd.init()

    # self.is_root = hvd.rank() == 0
    if self.ctype == 'LSTM':
      n = (self.sampled_states.c.shape[0],self.sampled_states.c.shape[1]*2)[0]
      # _state = self.convert_from_lstm_tuples(self.sampled_states)
      _state = self.sampled_states
    else: 
      n = self.sampled_states.shape[0]
      _state = self.sampled_states
    # print('here')
      
    sample = inputs
    sample_inputs = np.tile(sample,[n,1])
    # sample_inputs = inputs
    # all_fps = self.run_sequential_optimization(_state, sample_inputs)
    if self.is_root:  
      print("running joint optimizer")
    all_fps = self.run_joint_optimization(_state, sample_inputs)

    if self.is_root:
      print('All FPS shape ', all_fps.num_inits)
    # print(all_fps.xstar.shape)
    if self.is_root:    
      print('Finding unique Fixedpoints')
    unique_fps = all_fps.get_unique()
    if self.is_root:
      print('Found unique Fixedpoints with size ',unique_fps.num_inits)

    if (self.exclude_dis_outliers):
  
      unique_fps = self.exclude_dis_outliers_(unique_fps,_state )
      if self.is_root:
        print('Distance outliers excluded, currently size',unique_fps.num_inits)
    
    if self.rerun_q_outliers:
      unique_fps = self.run_additional_iterations_on_outliers_(unique_fps)
      unique_fps = unique_fps.get_unique()
    
    if unique_fps.num_inits > self.max_n_unique:
    # self._print_if_verbose('\tRandomly selecting %d unique '
    #     'fixed points to keep.' % self.max_n_unique)
      max_n_unique = int(self.max_n_unique)
      idx_keep = self.rng.choice(unique_fps.n, max_n_unique, replace=False)
      unique_fps = unique_fps[idx_keep]

    #can select fixed maximum number of points since all are not needed
    if self.compute_jacobians:

      if (unique_fps.num_inits > 0) :
        if self.is_root:

          print('computing recurrent jacobians')
        
        dFdx, dFdx_tf = self.compute_recurrent_jacobians(unique_fps)
        unique_fps.J_xstar = dFdx
        if self.is_root:

          print('Compute input Jacobians')
        dFdu, dFdu_tf = self.compute_input_jacobians(unique_fps)
        unique_fps.dFdu = dFdu
      else:
        num_states = unique_fps.num_states
        num_inputs = unique_fps.num_inputs

        shape_dFdx = (0, num_states, num_states)
        shape_dFdu = (0, num_states, num_inputs)
        
        unique_fps.J_xstar = unique_fps._alloc_zeros(shape_dFdx)
        unique_fps.dFdu = unique_fps._alloc_zeros(shape_dFdu)

    if self.decompose_jacobians:
      if self.is_root:
        print('decomposing Jacobians')
      unique_fps.decompose_jacobians() 
      if self.is_root:
        print('decomposed Jacobians')
    
    if save == True and self.is_root:
      print('saving')
      all_fps.save(self.savepath, 'all')
      unique_fps.save(self.savepath, 'unique')          
    if self.is_root:
      print('coming out')
    return unique_fps, all_fps