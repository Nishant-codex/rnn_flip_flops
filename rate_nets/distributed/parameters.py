import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt

print("--> Loading parameters...")

"""
Independent parameters
"""

par = {
    # Setup parameters
    'save_dir'              : os.getcwd()+'/savedir/',
    'save_fn'               : 'model_results.pkl',

    # Network configuration
    'synapse_config'        : 'full', # full is half facilitating, half depressing. See line 295 for all options
    'exc_inh_prop'          : 0.5,    # excitatory/inhibitory ratio, set to 1 so that units are neither exc or inh
    'balance_EI'            : True,
    'connection_prob'       : 1.,
    'dist'                  : 'gamma',
    # Network shape
    'num_motion_tuned'      : 24,
    'num_fix_tuned'         : 0,
    'num_rule_tuned'        : 0,
    'n_hidden'              : 100,
    'n_output'              : 3,
    'shape'                 : 1.,
    'scale'                 : 0.003,
    # Timings and rates
    'dt'                    : 10,
    'learning_rate'         : 2e-2,
    'membrane_time_constant': 100,

    # Input and noise
    'input_mean'            : 0.0,
    'noise_in_sd'           : 0.1,
    'noise_rnn_sd'          : 0.5,
    'input_len'             : 10,
    'p_flip'                : 0.02,
    # Synaptic plasticity specs
    'tau_fast'              : 200,
    'tau_slow'              : 1500,
    'U_stf'                 : 0.15,
    'U_std'                 : 0.45,
    # Loss parameters
    'spike_regularization'  : 'L2', # 'L1' or 'L2'
    'spike_cost'            : 2e-2,
    'weight_cost'           : 0.,
    'clip_max_grad_val'     : 0.1,
    'initial'               : 'exc_inh',

    # Training specs
    'batch_size'            : 1024,
    'num_iterations'        : 2000,
    'iters_between_outputs' : 100,

    # Task specs
    'trial_type'            : 'Flip', # allowable types: DMS, DMRS45, DMRS90, DMRS180, DMC, DMS+DMRS, ABBA, ABCA, dualDMS
    'rotation_match'        : 0,  # angular difference between matching sample and test
    'dead_time'             : 0,
    'fix_time'              : 500,
    'sample_time'           : 500,
    'delay_time'            : 1000,
    'test_time'             : 500,
    'variable_delay_max'    : 300,
    'mask_duration'         : 50,  # duration of training mask after test onset
    'catch_trial_pct'       : 0.0,
    'num_receptive_fields'  : 1,
    'num_rules'             : 1, # this will be two for the DMS+DMRS task
    'test_cost_multiplier'  : 1.,
    'rule_cue_multiplier'   : 1.,
    'var_delay'             : False,
    'rng'                   : np.random.RandomState(400),
    'activation'            : 'relu'
}



def update_parameters(updates):
    """ Takes a list of strings and values for updating parameters in the parameter dictionary
        Example: updates = [(key, val), (key, val)] """

    print('Updating parameters...')
    for key, val in updates.items():
        par[key] = val
        #print('Updating ', key)

    update_trial_params()
    update_dependencies()

def update_trial_params():
    """ Update all the trial parameters given trial_type """

    par['num_rules'] = 1
    par['num_receptive_fields'] = 1
    #par['num_rule_tuned'] = 0



    if par['trial_type'] == 'Flip':
        par['num_motion_tuned'] = par['n_output']


    else:
        print(par['trial_type'], ' not a recognized trial type')
        quit()


    par['trial_length'] = par['dead_time']+par['fix_time']+par['sample_time']+par['delay_time']+par['test_time']
    # Length of each trial in time steps
    par['num_time_steps'] = par['trial_length']//par['dt']


def update_dependencies():
    """ Updates all parameter dependencies """

    # Number of input neurons
    par['n_input'] = par['num_motion_tuned'] 
    # General network shape
    par['shape'] = (par['n_input'], par['n_hidden'], par['n_output'])
    # The time step in seconds
    par['dt_sec'] = par['dt']/1000
    # Length of each trial in ms

   
    # If exc_inh_prop is < 1, then neurons can be either excitatory or
    # inihibitory; if exc_inh_prop = 1, then the weights projecting from
    # a single neuron can be a mixture of excitatory or inhibitory
    if par['exc_inh_prop'] < 1:
        par['EI'] = True
    else:
        par['EI']  = False

    par['num_exc_units'] = int(np.round(par['n_hidden']*par['exc_inh_prop']))
    par['num_inh_units'] = par['n_hidden'] - par['num_exc_units']

    par['EI_list'] = np.ones(par['n_hidden'], dtype=np.float32)
    par['EI_list'][-par['num_inh_units']:] = -1.

    par['ind_inh'] = np.where(par['EI_list']==-1)[0]

    par['EI_matrix'] = np.diag(par['EI_list'])

    # Membrane time constant of RNN neurons
    par['alpha_neuron'] = np.float32(par['dt'])/par['membrane_time_constant']
    # The standard deviation of the Gaussian noise added to each RNN neuron
    # at each time step
    par['noise_rnn'] = np.sqrt(2*par['alpha_neuron'])*par['noise_rnn_sd']
    par['noise_in'] = np.sqrt(2/par['alpha_neuron'])*par['noise_in_sd'] # since term will be multiplied by par['alpha_neuron']

    # initial neural activity
    par['h0'] = 0.1*np.zeros((1, par['n_hidden']), dtype=np.float32)
    #par['h0'] = 0.1*np.ones((par['batch_size'], par['n_hidden']), dtype=np.float32)

    # initial input weights
    par['w_in0'] = initialize([par['n_input'], par['n_hidden']], 1., shape=0.2, scale=1.)


    # Initialize starting recurrent weights
    # If excitatory/inhibitory neurons desired, initializes with random matrix with
    #   zeroes on the diagonal
    # If not, initializes with a diagonal matrix

    if par['EI']:
        # par['w_rnn0'] = initialize([par['n_hidden'], par['n_hidden']], par['connection_prob'], shape=0.005, scale=1.)
        par['w_rnn0'] = np.float32(par['rng'].uniform(0.001,0.01,(par['n_hidden'], par['n_hidden'])))
        par['w_rnn_mask'] = (par['rng'].rand(*par['w_rnn0'].shape) < par['connection_prob'])
        par['w_rnn0'] *= par['w_rnn_mask']

    else:
        par['w_rnn0'] = np.float32(0.54*np.eye(par['n_hidden']))


    # initial recurrent biases
    par['b_rnn0'] = np.zeros((1, par['n_hidden']), dtype=np.float32)

    # Effective synaptic weights are stronger when no short-term synaptic plasticity
    # is used, so the strength of the recurrent weights is reduced to compensate
    if par['synapse_config'] is None:
        par['w_rnn0'] = par['w_rnn0']/3.

    # initial output weights and biases
    par['w_out0'] = initialize([par['n_hidden'], par['n_output']], 1.)
    # par['w_out0'] = np.float32(np.random.randn(par['n_hidden'], par['n_output']))/100

    par['b_out0'] = np.zeros((1, par['n_output']), dtype=np.float32)

    # for EI networks, masks will prevent self-connections, and inh to output connections
    par['w_rnn_mask'] = np.ones_like(par['w_rnn0'])
    par['w_out_mask'] = np.ones_like(par['w_out0'])
    par['w_in_mask'] = np.ones_like(par['w_in0'])

    # if par['EI']:
    #     par['w_rnn_mask'] = np.ones((par['n_hidden'], par['n_hidden']), dtype=np.float32) - np.eye(par['n_hidden'])
    #     par['w_out_mask'][par['ind_inh'], :] = 0

    # par['w_rnn0'] *= par['w_rnn_mask']
    # par['w_out0'] *= par['w_out_mask']
    
    # inh = np.random.rand(par['n_hidden'], 1) < (1-par['exc_inh_prop'])
    # exc = ~inh

    # mask = np.eye(par['n_hidden'], dtype=np.float32)
    # mask[np.where(inh==True)[0], np.where(inh==True)[0]] = -1
    # par['EI_matrix'] = mask

def initialize(dims, connection_prob, shape=0.1, scale=1.0 ):
    w = par['rng'].gamma(shape, scale, size=dims)
    w *= (par['rng'].rand(*dims) < connection_prob)

    return np.float32(w)

update_trial_params()
update_dependencies()

print("--> Parameters successfully loaded.\n")
