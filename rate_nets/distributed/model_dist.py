# %tensorflow_version 1.x
import tensorflow as tf
import numpy as np
import stimulus
import analysis
import pickle
import seaborn as sns
import time
import scipy.io as sio

import horovod.tensorflow as hvd
from hpc4neuro.errors import MpiInitError
from hpc4neuro.distribution import DataDistributor
import mpi4py

from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.util import nest
from tensorflow.python.ops import rnn
from tensorflow.python.ops.rnn_cell_impl import RNNCell
from FixedPointStore import FixedPointStore
from FixedPointSearch import FixedPointSearch
from parameters import par
import os, sys
np.random.seed(400)
# Ignore "use compiled version of TensorFlow" errors
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class LeakyRNNCell(RNNCell):
    """The most basic RNN cell.

    Args:
        num_units: int, The number of units in the RNN cell.
        activation: Nonlinearity to use.    Default: `tanh`.
        reuse: (optional) Python boolean describing whether to reuse variables
         in an existing scope.    If not `True`, and the existing scope already has
         the given variables, an error is raised.
        name: String, the name of the layer. Layers with the same name will
            share weights, but to avoid mistakes we require reuse=True in such
            cases.
    """

    def __init__(self,
                 num_units,
                 n_output,
                 alpha,
                 sigma_rec=0,
                 activation='softplus',
                 w_rec_init='diag',
                 rng=None,
                 reuse=None,
                 name=None):
        super(LeakyRNNCell, self).__init__(_reuse=reuse, name=name)

        # Inputs must be 2-dimensional.
        # self.input_spec = base_layer.InputSpec(ndim=2)

        self._num_units = num_units
        self._w_rec_init = w_rec_init
        self._reuse = reuse

        if activation == 'softplus':
            self._activation = tf.nn.softplus
            self._w_in_start = 1.0
            self._w_rec_start = 0.5
        elif activation == 'tanh':
            self._activation = tf.tanh
            self._w_in_start = 1.0
            self._w_rec_start = 1.0
        elif activation == 'relu':
            self._activation = tf.nn.relu
            self._w_in_start = 1.0
            self._w_rec_start = 0.5
        elif activation == 'power':
            self._activation = lambda x: tf.square(tf.nn.relu(x))
            self._w_in_start = 1.0
            self._w_rec_start = 0.01
        elif activation == 'retanh':
            self._activation = lambda x: tf.tanh(tf.nn.relu(x))
            self._w_in_start = 1.0
            self._w_rec_start = 0.5
        elif activation == 'sigmoid':
            self._activation = tf.nn.sigmoid
            self._w_in_start = 1.0
            self._w_rec_start = 0.5        
        else:
            raise ValueError('Unknown activation')
        self._alpha = alpha
        self._sigma = np.sqrt(2 / alpha) * sigma_rec
        if rng is None:
            self.rng = np.random.RandomState()
        else:
            self.rng = rng

        # Generate initialization matrix
        n_hidden = self._num_units
        w_in0 = (self.rng.randn(n_output, n_hidden) /
                 np.sqrt(n_output) * self._w_in_start)

        if self._w_rec_init == 'diag':
            w_rec0 = self._w_rec_start*np.eye(n_hidden)
        elif self._w_rec_init == 'randortho':
            w_rec0 = self._w_rec_start*tools.gen_ortho_matrix(n_hidden,
                                                              rng=self.rng)
        elif self._w_rec_init == 'randgauss':
            w_rec0 = (self._w_rec_start *
                      self.rng.randn(n_hidden, n_hidden)/np.sqrt(n_hidden))
        elif self._w_rec_init == 'exc_inh':

            w_rec0 = par['w_rnn0']
   

            

        matrix0 = np.concatenate((w_in0, w_rec0), axis=0)

        self.w_rnn0 = matrix0
        self._initializer = tf.constant_initializer(matrix0, dtype=tf.float32)

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def initialize(dims, connection_prob, shape=0.05, scale=1.0 ):
        w = np.random.gamma(shape, scale, size=dims)
        w *= (np.random.rand(*dims) < connection_prob)

        return np.float32(w)
    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError(
                "Expected inputs.shape[-1] to be known, saw shape: %s"
                                             % inputs_shape)

        input_depth = inputs_shape[1].value
        self._kernel = self.add_variable(
                'kernel',
                shape=[input_depth + self._num_units, self._num_units],
                initializer=self._initializer)
        # self._bias = self.add_variable(
        #         'bias',
        #         shape=[self._num_units],
        #         initializer=init_ops.zeros_initializer(dtype=self.dtype))

        self.built = True

    def call(self, inputs, state):
        """Most basic RNN: output = new_state = act(W * input + U * state + B)."""

        # gate_inputs = math_ops.matmul(
        #     array_ops.concat([inputs, state], 1), self._kernel)
        inputs = math_ops.matmul(inputs, self._kernel[:par['n_output'],:])
        rate = self._activation(state)
        if self._w_rec_init == 'exc_inh':
          rnn = math_ops.matmul(rate,par['w_rnn_mask']*(par['EI_matrix']@ tf.math.abs(self._kernel[par['n_output']:,:])))
        else:
          rnn = math_ops.matmul(state,self._kernel[par['n_output']:,:] )
        # rnn = math_ops.matmul(state, tf.nn.relu(self._kernel[par['n_output']:,:]))
        gate_inputs = math_ops.add(inputs,rnn)
        # gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)

        noise = tf.random_normal(tf.shape(state), mean=0, stddev=self._sigma)
        gate_inputs = gate_inputs + noise

        output = gate_inputs

        output = (1-self._alpha) * state + self._alpha * output

        return output, output
class Model:

    def __init__(self, input_data, target_data, mask):

        # Load the input activity, the target data, and the training mask for this batch of trials

        self.input_data = input_data
        self.cell = None
        self.target_data = target_data

        self.global_step = tf.train.get_or_create_global_step()

        self.initialize_weights()
        self.saver = tf.train.Saver()


        self.run_dyn_rnn()
        self.optimize_dyn()        
        

    def initialize_weights(self):
        # Initialize all weights. biases, and initial values
        self.var_dict = {}
        # all keys in par with a suffix of '0' are initial values of trainable variables
        a = ['w_out0','b_out0']
        with tf.variable_scope('losses',reuse=tf.AUTO_REUSE):
          for k, v in par.items():
            if k in a:
                name = k[:-1]
                self.var_dict[name] = tf.Variable(v, name=name, dtype=tf.float32)

    def run_dyn_rnn(self):
        # Main model loop

        self.h = []
        self.y = []
        self.r = []
        r = np.random.RandomState(400)
        self.cell = LeakyRNNCell(num_units = par['n_hidden'],
                            n_output= par['num_motion_tuned'],
                            alpha = par['alpha_neuron'],
                            sigma_rec = par['noise_rnn_sd'],
                            activation = par['activation'],
                            w_rec_init = par['inital'],
                            reuse=tf.AUTO_REUSE,
                            rng=r)
        
        self.h, states = rnn.dynamic_rnn(self.cell, self.input_data, dtype=tf.float32, time_major=True)
        self.r = self.cell._activation(self.h)        
        self.y = tf.tensordot(self.r,self.var_dict['w_out'],axes=1) + self.var_dict['b_out']
        self.h = tf.stack(self.h)
        self.r = tf.stack(self.r)
        self.y = tf.stack(self.y)

    def optimize_dyn(self):
        self.perf_loss = tf.reduce_mean(tf.squared_difference(self.y, self.target_data))

        # L2 penalty term on hidden state activity to encourage low spike rate solutions
        n = 2 if par['spike_regularization'] == 'L2' else 1
        self.spike_loss = tf.reduce_mean(self.r**n) 
        # self.weight_loss = tf.reduce_mean(tf.nn.relu(self.w_rnn)**n)
        var_list = tf.trainable_variables()
        self.loss = self.perf_loss + par['spike_cost']*self.spike_loss 
                    # + par['weight_cost']*self.weight_loss

        # opt = tf.train.AdamOptimizer(learning_rate = par['learning_rate'])

        decay = tf.train.exponential_decay(par['learning_rate'],self.global_step,128,0.9)
        opt = tf.train.MomentumOptimizer(decay*hvd.size(),0.9,use_nesterov=False)
        opt = hvd.DistributedOptimizer(opt)
        grads_and_vars = opt.compute_gradients(self.loss,var_list)
        capped_gvs = []

        for grad, var in grads_and_vars:
          if 'w_out' in var.op.name:
              grad *= par['w_out_mask']

          capped_gvs.append((tf.clip_by_norm(grad, par['clip_max_grad_val']), var))
        self.train_op = opt.apply_gradients(capped_gvs, global_step=self.global_step)

        # decay = tf.train.exponential_decay(par['learning_rate'], self.global_step, 128, 0.9)
        # optimizer = tf.train.MomentumOptimizer(decay, 0.9)
        # # optimizer = tf.train.AdamOptimizer(decay*hvd.size(),epsilon=1e-1)
        # optimizer = hvd.DistributedOptimizer( optimizer)
        # gradients, variables = zip(*optimizer.compute_gradients(self.loss,tf.trainable_variables()))
        # gradients = [None if gradient is None else tf.clip_by_norm(gradient, par['clip_max_grad_val']) for gradient in gradients]
        # self.train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)

    def save(self):
        """Save the model."""

        sess = tf.get_default_session()
        path = os.getcwd()+'/model'
        os.mkdir(path)
        save_path = os.path.join(path, 'model.ckpt')
        self.saver.save(sess, save_path)
        print("Model saved in file: %s" % save_path)

        
def main(gpu_id = None):
    tf.reset_default_graph()
    

    hvd.init()
    from tensorflow.core.protobuf import rewriter_config_pb2
    config = tf.ConfigProto()
    # off = rewriter_config_pb2.RewriterConfig.OFF
    # config.graph_options.rewrite_options.memory_optimization  = off
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())


    is_root = hvd.rank() == 0
    # if gpu_id is not None:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    # Print key parameters

    # Reset TensorFlow before running anything
    #Create the stimulus class to generate trial paramaters and input activity
    stim = stimulus.Stimulus()

    # Define all placeholder
    m = tf.placeholder(tf.float32, [par['num_time_steps'], par['batch_size']], 'mask')
    x = tf.placeholder(tf.float32, [par['num_time_steps'], par['batch_size'],  par['n_output']], 'input')
    t = tf.placeholder(tf.float32, [par['num_time_steps'], par['batch_size'],  par['n_output']],'target')
    

    if is_root:
        print_important_params()
    model = Model(x, t, m)

    num_steps = par['num_iterations']//hvd.size()


    if is_root:
        print('Total steps: ',num_steps)


    hooks = [hvd.BroadcastGlobalVariablesHook(0),

    tf.train.StopAtStepHook(last_step = num_steps),

    tf.train.LoggingTensorHook(tensors={'step': model.global_step, 
                                      'loss': model.loss},
                             every_n_iter=10) ]



    dir_name = os.getcwd()+'/leaky_rnn'

    try:
        if not os.path.exists(os.path.dirname(dir_name)):
            os.makedirs(os.path.dirname(dir_name))
        path = dir_name 
    except: 
        path = dir_name 


    # bcst = hvd.BroadcastGlobalVariablesHook(0)
    checkpoint_dir = path if hvd.rank() == 0 else None

    with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir,
                                           hooks=hooks,
                                           config=config,
                                           summary_dir = path+'/tf_logs/') as sess:


    # with tf.Session(config=config) as sess:

        # device = '/cpu:0' if gpu_id is None else '/gpu:0'
        # with tf.device(device):


        # sess.run(tf.global_variables_initializer())


        # keep track of the model performance across training
        model_performance = {'accuracy': [], 'loss': [], 'perf_loss': [], 'spike_loss': [], \
            'weight_loss': [], 'iteration': []}

        loss_ = []
        i = 0
        while i!=num_steps:
            # generate batch of batch_train_size
            trial_info = stim.generate_trial(set_rule = None)
            # print(trial_info['neural_input'].shape)
            # stim.plot_neural_input(trial_info)
            
            # Run the model
            _, loss, perf_loss, spike_loss, y, h ,r, kernel, w_out= \
                sess.run([model.train_op, model.loss, model.perf_loss, model.spike_loss, \
                 model.y, model.h, model.r,model.cell._kernel,model.var_dict['w_out']], \
                {x: trial_info['neural_input'], t: trial_info['desired_output'], m: trial_info['train_mask']})

            if is_root:
                accuracy = analysis.get_perf(trial_info['desired_output'], y, trial_info['train_mask'])

            if is_root:
                model_performance = append_model_performance(model_performance, accuracy, loss, perf_loss, spike_loss, i)

            # Save the network model and output model performance to screen
            if i%par['iters_between_outputs']==0:
                if is_root:
                    # stim.plot_neural_input(trial_info)

                    print_results(i, perf_loss, spike_loss, r, accuracy)
            
            loss_.append(perf_loss)
            # if perf_loss<0.004:
            #   break
            i +=1
            # sess.run(model.global_step)
        # Save model and results
        if is_root:
            # kernel = sess.run(model.cell._kernel)
            # w_out = sess.run(model.var_dict['w_out'])
            save_results(model_performance, w_out, kernel, h, trial_info)        
        path = os.getcwd()
        # model.save()
        #plt.plot(loss_)
        #plt.show()
        # unique, all_fps = fixed_points(h,model.cell,sess)
    
    reload_from_checkpoints(dir_name, model,  {'x':x,'m':m,'t':t})
    
    return(trial_info['desired_output'], y,trial_info['neural_input'],h)


def fixed_points(hid,cell,sess, fps_dir):
  n_bits = par['n_output']
  inputs = np.zeros([1,n_bits])
  def save_hidden_states(hid):

      # print(path) 
      dir_name = os.getcwd()+'/fps_saver/' 
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

  fps = FixedPointSearch(
              'rate',
              hid,
              fps_dir, 
              cell=cell,
              sess = sess
              )
  fps.sample_states(1000,hid,'rate',0.5)
  # fps.rerun_q_outliers = False
  unique, all_fps = fps.find_fixed_points(inputs, save = True)
  return unique, all_fps 



def reload_from_checkpoints(chkpt, model,  dict_t ):

   #  Uncomment to run a saved network
    # Need to close the session manually 
    hvd.init()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())

    # m_t = tf.placeholder(tf.float32, [par['num_time_steps'], par['batch_size']], 'mask_t')
    # x_t = tf.placeholder(tf.float32, [par['num_time_steps'], par['batch_size'],  par['n_output']], 'input_t')
    # t_t = tf.placeholder(tf.float32, [par['num_time_steps'], par['batch_size'],  par['n_output']],'target_t')


    # model_t = Model(x_t, t_t, m_t)

    model = model

    is_root = hvd.rank() == 0
    chkpt_ = tf.train.get_checkpoint_state(chkpt)

    saver = tf.train.Saver()

    stim = stimulus.Stimulus()
    # trial_info = stim.generate_trial(set_rule = None)

    test_data = stim.generate_trial(set_rule = None)
    
    feed_dict = {dict_t['x']: test_data['neural_input'], dict_t['t']: test_data['desired_output'], dict_t['m']: test_data['train_mask']}

    sess = tf.Session(config = config)

    sess.run(tf.global_variables_initializer())

    saver.restore(sess, chkpt_.model_checkpoint_path)

    if is_root: print("successfully loaded from checkpoints")

    h = sess.run( model.h,feed_dict=feed_dict)


    u,a = fixed_points(h,model.cell,sess,chkpt)
    transition_graph(model.cell, sess)

def save_results(model_performance, weights, kernel ,h, trial_info, save_fn = None):

    results = {'weights': weights, 'parameters': par}
    w_out = weights
    w_rnn = kernel[par['n_input']:,:]
    w_in = kernel[:par['n_input'],:]
    for k,v in model_performance.items():
        results[k] = v
    if save_fn is None:
        fn = par['save_dir'] + par['save_fn']
    else:
        fn = par['save_dir'] + save_fn
    if not os.path.exists(par['save_dir']):
      os.makedirs(par['save_dir'])
    fn = par['save_dir']+par['save_fn']
    fn_in = par['save_dir']+'w_in.pkl'
    fn_out = par['save_dir']+'w_out.pkl'
    fn_rnn = par['save_dir']+'weights.mat'
    pickle.dump(results, open(fn, 'wb'))
    sio.savemat(fn_rnn,{'w_in':w_in,'w_out':w_out,'w_rnn':w_rnn,'hid':h,'stim':trial_info})
    # sio.savemat(fn_in)
    # pickle.dump(w_in, open(fn_in, 'wb'))
    # pickle.dump(w_out, open(fn_out, 'wb'))
    # pickle.dump(w_rnn, open(fn_rnn, 'wb'))
    print('Model results saved in ',fn)
    
def transition_graph(cell, sess, plot=True, save=True):

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
  fps = FixedPointStore(num_inits = 1,
                        num_states= 1,
                        num_inputs = 1)
  dict_d = fps.restore(os.getcwd()+'/leaky_rnn/fps_saver/fixedPoint_unique.p')
  fps.__dict__ = dict_d
  trans = np.zeros([fps.num_inits,fps.num_inits])
  stim = stimulus.Stimulus()
  trial_info = stim.generate_trial(set_rule = None)

  for y in range(20):
    fixed_points = fps.xstar
    fps_w_noise = fps.xstar+1e-01*np.random.randn(fixed_points.shape[0],fixed_points.shape[1])
    init_state = tf.convert_to_tensor(fps_w_noise, dtype=tf.float32)
    x0 = np.zeros_like(trial_info['neural_input'])
    x = tf.placeholder(tf.float32, [par['num_time_steps'], fps.num_inits,  par['n_input']], name='input_placeholder')
    
    r = np.random.RandomState(400)
    # cell = LeakyRNNCell(num_units = par['n_hidden'],
    #                     n_input= par['num_motion_tuned'],
    #                     alpha = par['alpha_neuron'],
    #                     sigma_rec = par['noise_rnn_sd'],
    #                     activation = 'relu',
    #                     w_rec_init = par['inital'],
    #                     rng=r)
    h, states = rnn.dynamic_rnn(cell, x, dtype=tf.float32, time_major=True,initial_state=init_state)
    zero_inpt = np.zeros(( par['num_time_steps'], fps.num_inits,  par['n_input']))
    hids = sess.run(h, feed_dict={x:zero_inpt})
    for l in range(fps.num_inits):
      for j in range(fps.num_inits):
        index = slice(j,j+1)
        inits_ = fixed_points[index]
        if np.linalg.norm(inits_-hids[-1,l,:])<0.01:
          trans[l,j]+=1

  if plot:
    sns.heatmap(trans,cmap='rainbow')

  if save:
    save_transitions(trans,os.getcwd()+'/leaky_rnn/fps_saver')

def append_model_performance(model_performance, accuracy, loss, perf_loss, spike_loss, iteration):

    model_performance['accuracy'].append(accuracy)
    model_performance['loss'].append(loss)
    model_performance['perf_loss'].append(perf_loss)
    model_performance['spike_loss'].append(spike_loss)
    # model_performance['weight_loss'].append(weight_loss)
    model_performance['iteration'].append(iteration)

    return model_performance


def print_results(iter_num, perf_loss, spike_loss,  h, accuracy):

    print(par['trial_type'] + ' Iter. {:4d}'.format(iter_num) + ' | Accuracy {:0.4f}'.format(accuracy) +
      ' | Perf loss {:0.4f}'.format(perf_loss) + ' | Spike loss {:0.4f}'.format(spike_loss) +
      ' | Mean activity {:0.4f}'.format(np.mean(h)))

def print_important_params():

    important_params = ['num_iterations', 'learning_rate', 'noise_rnn_sd', 'noise_in_sd','spike_cost',\
        'spike_regularization', 'weight_cost','test_cost_multiplier', 'trial_type','balance_EI', 'dt',\
        'delay_time', 'connection_prob','synapse_config','tau_slow','tau_fast']
    for k in important_params:
        print(k, ': ', par[k])
