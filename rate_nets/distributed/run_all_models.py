
import numpy as np
from parameters import *
# import model
import model_dist
import sys
import matplotlib.pyplot as plt
# %matplotlib inline

# def try_model(gpu_id):
#     try:
#         # Run model
#         h,y,t,x= model_dist.main(gpu_id)
#     except KeyboardInterrupt:
#         quit('Quit by KeyboardInterrupt')
#     return h,y,t,x
# try:
#     gpu_id = sys.argv[1]
#     print('Selecting GPU ', gpu_id)
# except:
#     gpu_id = None


update_parameters({ 'simulation_reps'       : 0,
                    'exc_inh_prop'          : .8,
                    'spike_cost'            : 1e-05,
                    'batch_size'            : 128,
                    'dt'                    : 5,
                    'dist'                  : 'gamma',
                    'inital'                : 'exc_inh',
                    'shape'                 : 1.,
                    'scale'                 : 0.1,
                    'n_hidden'              : 250,
                    'learning_rate'         : 1e-02,
                    'membrane_time_constant': 50,
                    'noise_rnn_sd'          : 0.0,
                    'noise_in_sd'           : 0.0,
                    'num_iterations'        : 20001,
                    'clip_max_grad_val'     : .1,
                    'spike_regularization'  : 'L2',
                    'synaptic_config'       : None,
                    'test_cost_multiplier'  : 1.,
                    'balance_EI'            : False,
                    'connection_prob'       : .8,
                    'delay_time'            : 3000,
                    'p_flip'                : 0.02,
                    'input_len'             : 10,
                    'activation'            : 'relu',    
                    'savedir'               : './savedir/'})

task_list = ['Flip']

update_parameters({'trial_type': task_list[0]})
h,y,t,x= model_dist.main(None)
