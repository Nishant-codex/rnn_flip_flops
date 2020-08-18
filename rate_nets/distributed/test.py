# %tensorflow_version 1.x

import tensorflow as tf
import numpy as np
import stimulus
import analysis
import pickle
import time
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
                 n_input,
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
        w_in0 = (self.rng.randn(n_input, n_hidden) /
                 np.sqrt(n_input) * self._w_in_start)

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
        self._bias = self.add_variable(
                'bias',
                shape=[self._num_units],
                initializer=init_ops.zeros_initializer(dtype=self.dtype))

        self.built = True

    def call(self, inputs, state):
        """Most basic RNN: output = new_state = act(W * input + U * state + B)."""

        # gate_inputs = math_ops.matmul(
        #     array_ops.concat([inputs, state], 1), self._kernel)
        
        inputs = math_ops.matmul(inputs, self._kernel[:par['n_input'],:])

        if self._w_rec_init == 'exc_inh':
            print('exc_inh')
            rnn = tf.matmul(state, tf.constant(par['EI_matrix'])@ tf.nn.relu(self._kernel[par['n_input']:,:])) 
          
        else:
            rnn = math_ops.matmul(state, self._kernel[par['n_input']:,:] )
        # rnn = math_ops.matmul(state, tf.nn.relu(self._kernel[par['n_input']:,:]))
        gate_inputs = math_ops.add(inputs,rnn)
        gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)

        noise = tf.random_normal(tf.shape(state), mean=0, stddev=self._sigma)
        gate_inputs = gate_inputs + noise

        output = self._activation(gate_inputs)

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
        self.saver = None


        self.run_dyn_rnn()
        self.optimize_dyn()        
        

    def initialize_weights(self):
        # Initialize all weights. biases, and initial values
        self.var_dict = {}

        # all keys in par with a suffix of '0' are initial values of trainable variables
        a = ['w_out0','b_out0']
        for k, v in par.items():

            if k in a:
                name = k[:-1]
                self.var_dict[name] = tf.Variable(par[k], name)

    def run_dyn_rnn(self):
        # Main model loop

        self.h = []
        self.y = []
        r = np.random.RandomState(400)
        self.cell = LeakyRNNCell(num_units = par['n_hidden'],
                            n_input= par['num_motion_tuned'],
                            alpha = par['alpha_neuron'],
                            sigma_rec = par['noise_rnn_sd'],
                            activation = 'relu',
                            w_rec_init = par['inital'],
                            rng=r)
        
        self.h, states = rnn.dynamic_rnn(self.cell, self.input_data, dtype=tf.float32, time_major=True)
        self.y = tf.tensordot(self.h,self.var_dict['w_out'],axes=1) + self.var_dict['b_out']
        self.h = tf.stack(self.h)
        self.y = tf.stack(self.y)

    def optimize_dyn(self):
        self.perf_loss = tf.reduce_mean(tf.squared_difference(self.y, self.target_data))

        # L2 penalty term on hidden state activity to encourage low spike rate solutions
        n = 2 if par['spike_regularization'] == 'L2' else 1
        self.spike_loss = tf.reduce_mean(self.h**n) 
        # self.weight_loss = tf.reduce_mean(tf.nn.relu(self.w_rnn)**n)
        var_list = tf.trainable_variables()
        self.loss = self.perf_loss + par['spike_cost']*self.spike_loss 
                    # + par['weight_cost']*self.weight_loss

        # opt = tf.train.AdamOptimizer(learning_rate = par['learning_rate'])
        decay = tf.train.exponential_decay(par['learning_rate'],self.global_step,1,0.5)
        opt = tf.train.MomentumOptimizer(par['learning_rate'],0.9,use_nesterov=False)

        grads_and_vars = opt.compute_gradients(self.loss,var_list)
        capped_gvs = []
        


        for grad, var in grads_and_vars:
          if 'w_out' in var.op.name:
              grad *= par['w_out_mask']

          capped_gvs.append((tf.clip_by_norm(grad, par['clip_max_grad_val']), var))
        self.train_op = opt.apply_gradients(capped_gvs)

    def save(self):
        """Save the model."""
        sess = tf.get_default_session()
        save_path = os.path.join(os.getcwd(), 'model.ckpt')
        self.saver.save(sess, save_path)
        print("Model saved in file: %s" % save_path)
def main(gpu_id = None):

    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    # Print key parameters
    print_important_params()

    # Reset TensorFlow before running anything
    tf.reset_default_graph()

    #Create the stimulus class to generate trial paramaters and input activity
    stim = stimulus.Stimulus()

    # Define all placeholder
    m = tf.placeholder(tf.float32, [par['num_time_steps'], par['batch_size']], 'mask')
    x = tf.placeholder(tf.float32, [par['num_time_steps'], par['batch_size'],  par['n_input']], 'input')
    t = tf.placeholder(tf.float32, [par['num_time_steps'], par['batch_size'],  par['n_output']],'target')

    # enter "config=tf.ConfigProto(log_device_placement=True)" inside Session to check whether CPU/GPU in use
    with tf.Session(config=tf.ConfigProto()) as sess:

        device = '/cpu:0' if gpu_id is None else '/gpu:0'
        with tf.device(device):
            model = Model(x, t, m)

        sess.run(tf.global_variables_initializer())

        # keep track of the model performance across training
        model_performance = {'accuracy': [], 'loss': [], 'perf_loss': [], 'spike_loss': [], \
            'weight_loss': [], 'iteration': []}


        for i in range(par['num_iterations']):

            # generate batch of batch_train_size
            trial_info = stim.generate_trial(set_rule = None)
            # print(trial_info['neural_input'].shape)
            # stim.plot_neural_input(trial_info)
            
            # Run the model
            _, loss, perf_loss, spike_loss, y, h = \
                sess.run([model.train_op, model.loss, model.perf_loss, model.spike_loss, \
                 model.y, model.h], \
                {x: trial_info['neural_input'], t: trial_info['desired_output'], m: trial_info['train_mask']})

            accuracy = analysis.get_perf(trial_info['desired_output'], y, trial_info['train_mask'])

            model_performance = append_model_performance(model_performance, accuracy, loss, perf_loss, spike_loss, i)

            # Save the network model and output model performance to screen
            if i%par['iters_between_outputs']==0:
                # stim.plot_neural_input(trial_info)
                print_results(i, perf_loss, spike_loss, h, accuracy)
            if perf_loss<0.004:
              break
        # Save model and results
        weights = sess.run(model.var_dict)
        path = os.getcwd()
        # unique, all_fps = fixed_points(h,model.cell,sess)
        return(trial_info['desired_output'], y,trial_info['neural_input'],h)
        save_results(model_performance, weights)

def fixed_points(hid,cell,sess):
  n_bits = 3
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
              '/content', 
              cell=cell,
              sess = sess
              )
  fps.sample_states(600,hid,'rate',0.9)
  fps.rerun_q_outliers = False
  unique, all_fps = fps.find_fixed_points(inputs, save = True)
  return unique, all_fps 
def save_results(model_performance, weights,  save_fn = None):

    results = {'weights': weights, 'parameters': par}
    for k,v in model_performance.items():
        results[k] = v
    if save_fn is None:
        fn = par['save_dir'] + par['save_fn']
    else:
        fn = par['save_dir'] + save_fn
    pickle.dump(results, open(fn, 'wb'))
    print('Model results saved in ',fn)


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
import numpy as np
from parameters import *
import model
import model_leaky
import sys
import matplotlib.pyplot as plt
# %matplotlib inline

def try_model(gpu_id):
    try:
        # Run model
        h,y,t,x= model.main(gpu_id)
    except KeyboardInterrupt:
        quit('Quit by KeyboardInterrupt')
    return h,y,t,x
try:
    gpu_id = sys.argv[1]
    print('Selecting GPU ', gpu_id)
except:
    gpu_id = None


update_parameters({ 'simulation_reps'       : 0,
                    'exc_inh_prop'          : .8,
                    'spike_cost'            : 1e-02,
                    'batch_size'            : 4,
                    'dt'                    : 5,
                    'dist'                  : 'gamma',
                    'inital'                : 'randgauss',
                    'shape'                 : 1.,
                    'scale'                 : 0.1,
                    'n_hidden'              : 150,
                    'learning_rate'         : 1e-02,
                    'membrane_time_constant': 55,
                    'noise_rnn_sd'          : 0.0,
                    'noise_in_sd'           : 0.0,
                    'num_iterations'        : 1,
                    'spike_regularization'  : 'L2',
                    'synaptic_config'       : None,
                    'test_cost_multiplier'  : 1.,
                    'balance_EI'            : True,
                    'connection_prob'       : 0.8,
                    'delay_time'            : 100,
                    'p_flip'                : 0.02,
                    'input_len'             : 10,
                    'savedir'               : './savedir/'})

task_list = ['Flip']

update_parameters({'trial_type': task_list[0]})
h,y,t,x= main(None)
