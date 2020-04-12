# -*- coding: utf-8 -*-
"""flip_flop_lstm_states

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1V9SlynnIlbvdAEHV1-D28mH45Z1c6D37
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import sys 

sys.path.insert(0,'/home/joshi/fixed_point_edits')
import os
import absl
from tensorflow.python.ops import parallel_for as pfor

import tensorflow as tf

#import cProfile
# %tensorflow_version 1.x magic
#import matplotlib.pyplot as plt
import numpy.random as nrand
import pickle 
np.random.seed(400)
# import numpy as np
import time
from AdaptiveGradNormClip import AdaptiveGradNormClip
from AdaptiveLearningRate import AdaptiveLearningRate

class FixedPointStore:

      
  
  def __init__(self, 
               num_inits=None, 
               num_states=None,
               num_inputs=None,  
               xstar=None,
               x_init=None,
               inputs=None,
               F_xstar=None,
               qstar=None,
               eigval_J_xstar=None,
               eigvec_J_xstar=None,
               dq=None,
               dFdu = None,
               n_iters=None,
               tol_unique=1e-3,
               alloc_zeros = True,
               J_xstar = None
            ):

    if alloc_zeros == True:
      if num_inits is None:
          raise ValueError('n must be provided if '
                            'do_alloc_nan == True.')
      if num_states is None:
          raise ValueError('n_states must be provided if '
                            'do_alloc_nan == True.')
      if num_inputs is None:
          raise ValueError('n_inputs must be provided if '
                            'do_alloc_nan == True.')
      self.num_states = num_states
      self.num_inits = num_inits
      self.num_inputs = num_inputs
      
      self.tol_unique = tol_unique
      self.xstar = self._alloc_zeros([num_inits, num_states])
      self.x_init = self._alloc_zeros([num_inits, num_states])
      self.inputs = self._alloc_zeros([num_inits, num_inputs])
      self.F_xstar = self._alloc_zeros([num_inits, num_states])
      self.qstar = self._alloc_zeros((num_inits))
      self.dq = self._alloc_zeros((num_inits))
      self.dFdu = None
      self.n_iters = self._alloc_zeros((num_inits))
      self.J_xstar = self._alloc_zeros((num_inits, num_states, num_states))
      self.eigval_J_xstar = self._alloc_zeros((num_inits, num_states))
      self.eigvec_J_xstar = self._alloc_zeros((num_inits, num_states, num_states))
      self.has_decomposed_jacobians = False
      self._attrs = ['xstar','x_init','F_xstar','qstar','dq','n_iters',
                    #  'J_xstar',
                    #  'eigval_J_xstar',
                    #  'eigvec_J_xstar'
                    ]
      self.attrs_dict = {'xstar':self.xstar,
                        'x_init':self.x_init,
                        'F_xstar':self.F_xstar,
                        'qstar':self.qstar,
                        'dq':self.dq,
                        'n_iters':self.n_iters,
                        #  'J_xstar':self.J_xstar,
                        #  'eigval_J_xstar':self.eigval_J_xstar,
                        #  'eigvec_J_xstar':self.eigvec_J_xstar
                        }
    else:
      if xstar is not None:
        self.num_inits, self.num_states = xstar.shape
      elif x_init is not None:
        self.num_inits, self.num_states = x_init.shape
      elif F_xstar is not None:
        self.num_inits, self.num_states = F_xstar.shape
      elif J_xstar is not None:
        self.num_inits, self.num_states, _ = J_xstar.shape
      else:
        self.num_inits = None
        self.num_states = None

      if inputs is not None:
        self.num_inputs = inputs.shape[1]
        if self.num_inits is None:
            self.num_inits = inputs.shape[0]
      else:
        self.num_inputs = None

      self.tol_unique = tol_unique
      self.xstar = xstar
      self.x_init = x_init
      self.inputs = inputs
      self.F_xstar = F_xstar
      self.qstar = qstar
      self.dq = dq
      self.dFdu = None
      self.J_xstar = J_xstar
      self.eigvec_J_xstar = eigvec_J_xstar
      self.eigval_J_xstar = eigval_J_xstar
      self.n_iters = n_iters
      self.has_decomposed_jacobians = eigval_J_xstar is not None

      self.attrs_dict = {'xstar':self.xstar,
                       'x_init':self.x_init,
                       'F_xstar':self.F_xstar,
                       'J_xstar':self.J_xstar,
                       'eigvec_J_xstar':self.eigvec_J_xstar,
                       'eigval_J_xstar':self.eigval_J_xstar,
                       'qstar':self.qstar,
                       'dq':self.dq,
                       'n_iters':self.n_iters}

  def _alloc_zeros(self, shape):
    return np.zeros(shape)

  def get_attrs(self, attr , fps):
    try:
      return (self.attrs_dict[attr] , fps.attrs_dict[attr])
    except:
      pass

  def __setitem__(self, index, fps):
    if isinstance(index, int):
      # Force the indexing that follows to preserve numpy array ndim
      index = range(index, index+1)

      if self.xstar is not None:
          self.xstar[index] = fps.xstar

      if self.x_init is not None:
          self.x_init[index] = fps.x_init

      if self.inputs is not None:
          self.inputs[index] = fps.inputs

      if self.F_xstar is not None:
          self.F_xstar[index] = fps.F_xstar

      if self.qstar is not None:
          self.qstar[index] = fps.qstar

      if self.dq is not None:
          self.dq[index] = fps.dq

      if self.J_xstar is not None:
          self.J_xstar[index] = fps.J_xstar

      if self.has_decomposed_jacobians:
          self.eigval_J_xstar[index] = fps.eigval_J_xstar
          self.eigvec_J_xstar[index] = fps.eigvec_J_xstar

  def index_(self, obj, index):
    if obj is None:
      return None
    else:
      return obj[index]
      
  def get_unique(self):

    def unique(data, tol):
      d = int(np.round(np.max([0 -np.log10(tol)])))

      val,idx = np.unique(data.round(decimals=d), axis=0 ,return_index = True)
      return val,idx
    if self.xstar is not None:
      if self.inputs is not None:
        data_ = np.concatenate((self.xstar,self.inputs),axis=1)
        # print(data_.shape)
      else:
        data_ = self.xstar
    else:
      raise ValueError('cannot find unique fixed points, self.xstar is None')
    val, idx = unique(data_, self.tol_unique)
    return self[idx]
  
  def find(self, fp):

    result = np.array([], dtype=int)

    if isinstance(fp, FixedPointStore):
      if fp.num_states == self.num_states and fp.num_inputs == self.num_inputs:

          self_data = np.concatenate((self.xstar, self.inputs), axis=1)
          arg_data = np.concatenate((fp.xstar, fp.inputs), axis=1)

          elementwise_abs_diff = np.abs(self_data - arg_data)
          hits = np.all(elementwise_abs_diff <= self.tol_unique, axis=1)

          result = np.where(hits)[0]

    return result

  def __contains__(self, xstar):

      idx = self.find(xstar)

      return idx.size > 0


  def decompose_jacobians(self, do_batch=True, str_prefix=''):
    if self.has_decomposed_jacobians:
      print('%Jacobians have been decomposed, not repeating. '% str_prefix )
      return 
    
    num = self.num_inits
    num_states = self.num_states

    if do_batch:

      print('%Decomposing jacobians in a single batch.' % str_prefix)

      valid_J_idx = ~np.any(np.isnan(self.J_xstar), axis=(1,2))
  
      if np.all(valid_J_idx):
        e_vals_unsrt, e_vecs_unsrt = np.linalg.eig(self.J_xstar)
      else:
        e_vals_unsrt = self._alloc_zeros((num,num_states), dtype=np.complex64)

        e_vecs_unsrt = self._alloc_zeros((num,num_states,num_states),dtype=np.complex64)

        e_vals_unsrt[valid_J_idx],e_vecs_unsrt[valid_J_idx] = np.linalg.eig(self.J_xstar[valid_J_idx])

    else:
      print('%Decomposing jacobians one at a time.' %str_prefix)
      e_vals = []
      e_vecs = []
      for J in self.J_xstar:

        if np.any(np.isnan(J)):
          e_vals_i = self._alloc_zeros((num_states,))
          e_vecs_i = self._alloc_zeros((num_states, num_states))

        else:
          e_vals_i, e_vecs_i = np.linalg.eig(J)

        e_vals.append(np.expand_dims(e_vals_i, axis = 0))

        e_vecs.append(np.expand_dims(e_vecs_i, axis = 0))

      e_vals_unsrt = np.concatenate(e_vals,axis=0)
      e_vecs_unsrt = np.concatenate(e_vecs,axis=0)

    print('%sorting by Eigenvalues magnitude.' % str_prefix)

    sort_idx = np.argsort(np.abs(e_vals_unsrt))[:,::-1]

    self.eigval_J_xstar = self._alloc_zeros((num,num_states),dtype=np.complex64)

    self.eigvec_J_xstar = self._alloc_zeros((num,num_states),dtype=np.complex64)

    for k in range(num):
      sort_idx_k = sort_idx[k]
      self.eigval_J_xstar[k] = e_vals_unsrt[k][sort_idx_k]
    self.has_decomposed_jacobians = True


  def __getitem__(self, index):
    if isinstance(index, int):
    # Force the indexing that follows to preserve numpy array ndim
      index = range(index, index+1)
    xstar  = self.index_(self.xstar , index) 
    x_init = self.index_(self.x_init, index)
    inputs = self.index_(self.inputs, index)
    F_xstar= self.index_(self.F_xstar, index)
    qstar  = self.index_(self.qstar, index)
    dq = self.index_(self.dq, index)
    n_iters = self.index_(self.n_iters, index)
    tol = self.tol_unique
    J_xstar = self.index_(self.J_xstar, index)
    dFdu = self.dFdu
    if self.has_decomposed_jacobians:
      eigval_J_xstar = self._safe_index(self.eigval_J_xstar, index)
      eigvec_J_xstar = self._safe_index(self.eigvec_J_xstar, index)
    else:
      eigval_J_xstar = None
      eigvec_J_xstar = None
    indexed_fp = FixedPointStore(xstar=xstar,
                             alloc_zeros = False,
                             x_init = x_init,
                             inputs = inputs,
                             J_xstar= J_xstar,
                             dFdu = dFdu,
                             eigval_J_xstar = eigval_J_xstar,
                             eigvec_J_xstar = eigvec_J_xstar,
                             F_xstar = F_xstar,
                             qstar = qstar,
                             dq = dq,
                             n_iters = n_iters,
                             tol_unique = tol)
    return indexed_fp

  def save(self, string):
    dir_name = 'fps_saver'
    if not os.path.exists(os.path.dirname(dir_name)):
      os.makedirs(os.path.dirname(dir_name))
    
    filename  = 'fps_saver/fixedPoint_'+string+'.p'   
    f =  open(filename,'wb')
    pickle.dump(self.__dict__,f)
    f.close()

  def restore(self, path):
    file = open(path,'rb')
    restore_data = file.read()
    file.close()
    # print(type(pickle.loads(restore_data)))
    # print(type(self.__dict__))
    self.__dict__ = pickle.loads(restore_data,encoding='latin1')