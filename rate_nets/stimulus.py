import numpy as np
import matplotlib.pyplot as plt
from parameters import *


class Stimulus:

    def __init__(self):

        # generate tuning functions
        pass

    def generate_trial(self, test_mode = False, set_rule = None):

        if par['trial_type'] == 'Flip':
            trial_info = self.generate_flip_flop_trial()
        # input activity needs to be non-negative
        # trial_info['neural_input'] = np.maximum(0., trial_info['neural_input'])

        return trial_info


    def generate_flip_flop_trial(self):
        np.random.seed(400)

        trial_info = {'neural_input': np.zeros([par['num_time_steps'],par['batch_size'],par['n_output']]),
                      'desired_output':np.zeros([par['num_time_steps'],par['batch_size'],par['n_output']]),
                      'train_mask': np.ones((par['num_time_steps'], par['batch_size'],),dtype=np.float32)}    



        unsigned_inp = np.random.binomial(1,par['p_flip'],[par['num_time_steps'],par['batch_size'],par['n_output']])
        unsigned_out = 2*np.random.binomial(1,0.5,[par['num_time_steps'],par['batch_size'],par['n_output']]) -1 


        inputs = np.multiply(unsigned_inp,unsigned_out)
        inputs[0,:,:] = 1.0
        # plt.plot(inputs[:,0,0])
        for j in range(inputs.shape[1]):
          for k in range(inputs.shape[2]):
            
            pos = np.where(inputs[:,j,k]== 1)
            neg = np.where(inputs[:,j,k]== -1)
            for i in pos[0]:
              inputs[i:i+par['input_len'],j,k] =1 
            for i in neg[0]:
              inputs[i:i+par['input_len'],j,k] =-1 

        trial_info['neural_input'] = inputs
        # trial_info['neural_input'] = 0.5*trial_info['neural_input']
        output = np.zeros_like(inputs)
        for trial_idx in range(par['batch_size']):
            for bit_idx in range(par['n_output']):
                input_ = np.squeeze(inputs[:,trial_idx,bit_idx])
                t_flip = np.where(input_ != 0)
                for flip_idx in range(np.size(t_flip)):
                    # Get the time of the next flip
                    t_flip_i = t_flip[0][flip_idx]

                    '''Set the output to the sign of the flip for the
                    remainder of the trial. Future flips will overwrite future
                    output'''
                    output[t_flip_i:,trial_idx, bit_idx] = \
                        inputs[t_flip_i, trial_idx, bit_idx]

        trial_info['desired_output'] = output
        # trial_info['desired_output'] = 0.5*trial_info['desired_output']

        return trial_info





    def plot_neural_input(self, trial_info):

        # print(trial_info['desired_output'][ :, 0, :].T)
        f = plt.figure(figsize=(8,4))
        ax = f.add_subplot(1, 1, 1)
        t = np.arange(0,400+500+2000,par['dt'])
        t -= 900
        t0,t1,t2,t3 = np.where(t==-500), np.where(t==0),np.where(t==500),np.where(t==1500)
        #im = ax.imshow(trial_info['neural_input'][:,0,:].T, aspect='auto', interpolation='none')
        im = ax.imshow(trial_info['neural_input'][:,:,0], aspect='auto', interpolation='none')
        plt.imshow(trial_info['desired_output'][:, :, 0], aspect='auto')
        ax.set_xticks([t0[0], t1[0], t2[0], t3[0]])
        ax.set_xticklabels([-500,0,500,1500])
        ax.set_yticks([0, 9, 18, 27])
        ax.set_yticklabels([0,90,180,270])
        f.colorbar(im,orientation='vertical')
        ax.spines['right'].set_visible(True)
        ax.spines['top'].set_visible(True)
        ax.set_ylabel('Motion direction')
        ax.set_xlabel('Time relative to sample onset (ms)')
        ax.set_title('Motion input')
        plt.show()
        plt.savefig('stimulus.pdf', format='pdf')
