import numpy as np
import pylab
import os
from scipy.io import loadmat
import pickle 
np.random.seed(300)

import numpy as np

def generate_flip_flop_trial(par):

    # par['p_flip'] = 0.1
    # par['input_len'] = 10
    trial_info = {'neural_input': np.zeros([par['num_time_steps'],par['batch_size'],par['n_output']]),
                  'desired_output':np.zeros([par['num_time_steps'],par['batch_size'],par['n_output']]),
                  'train_mask': np.ones((par['num_time_steps'], par['batch_size'],),dtype=np.float32)}    



    unsigned_inp = par['rng'].binomial(1,par['p_flip'],[par['num_time_steps']//par['input_len'],par['batch_size'],par['n_output']])
    unsigned_out = 2*par['rng'].binomial(1,0.5,[par['num_time_steps']//par['input_len'],par['batch_size'],par['n_output']]) -1 


   
    inputs = unsigned_inp
    inputs = np.multiply(unsigned_inp,unsigned_out)
    inputs[0,:,:] = 1.0

    inputs = np.repeat(inputs,par['input_len'],axis=0)

    # inputs = inputs[:par['num_time_steps'],:,:]
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
def restore(path):
    file = open(path,'rb')
    restore_data = file.read()
    file.close()
    hid= pickle.loads(restore_data,encoding='latin1')
    return(hid)
par = restore(os.getcwd()+'/model_results.pkl')['parameters']

u = generate_flip_flop_trial(par)

data=loadmat(os.getcwd()+'/weights.mat')
# print(data['w_out'])


def LIF_net(data,par,u,hyp): #data['opt_scaling_factor'][0][0]

  inputs = u #generate_flip_flop_trial(par)
  N = par['n_hidden'] 
  w0 =  par['w_rnn0']               #data['w0']
  m =   par['EI_matrix']            #data['m']
  som_m = par['w_rnn_mask']         #data['som_m']
  w = data['w_rnn']
  w_in = data['w_in'] 
  u_ = u['neural_input'][:,0,:]


  taus = 50
  w_out = data['w_out']#[0][0][0]

 
  dt= par['dt']                  #5.00000000000000e-05
  use_initial_weights = False

  batch_size = u_.shape[1]
  w =  m@w 
  # w = w  

  #Scale the connectivity weights by the optimal scaling factor 
  
  W = w/hyp['scaling_factor'];

  #Inhibitory and excitatory neurons
  inh_ind = par['ind_inh']                     
  exc_ind = np.where(par['EI_list']==1)[0]     

  # Input stimulus

  ext_stim = u_ @ w_in

  #------------------------------------------------------
  # LIF network parameters
  #------------------------------------------------------

  downsample= hyp['downsample']
  dt = hyp['dt']*downsample     # sampling rate
  T = u_.shape[0]*dt        # trial duration (in sec)

  nt =  round(T/dt) #par['num_time_steps']          # total number of points in a trial
  tref = hyp['tref'] # 0.005#0.002              # refractory time constant (in sec)
  tm =  hyp['tm'] #.01         # membrane time constant (in sec)
  vreset = hyp['vreset'] #-45 # voltage reset (in mV)
  vpeak = hyp['vpeak'] # 20   # voltage peak (in mV) for linear LIF

  # Synaptic decay time constants (in sec) for the double-exponential
  # synpatic filter
  # tr: rising time constant
  # td: decay time constants
  # td0: initial decay time constants (before optimization)

  td = taus*hyp['tx']/1000 
  td0 = td
  tr = hyp['tr']

  #Synaptic parameters
  IPSC = np.zeros(int(N))       # post synaptic current storage variable
  h = np.zeros(int(N))          # storage variable for filtered firing rates
  r =np.zeros(int(N))           # second storage variable for filtered rates
  hr =np.zeros(int(N))          # third variable for filtered rates
  JD = IPSC                    # storage variable required for each spike time
  tspike = np.zeros((200*int(nt),2))  # storage variable for spike times
  ns = 0                       # number of spikes, counts during simulation
  #TODA add randomness again
  v_val = hyp['v_val'] #100
  v = vreset + np.random.rand(int(N))*(v_val- vreset) # initialize voltage with random distribtuions
  
  v_ = v                                       # v_ is the voltage at previous time steps
  v0 = v                                       # store the initial voltage values

  # Record REC (membrane voltage), Is (input currents), 
  # spk (spike raster), rs (firing rates) from all the units
  REC = np.zeros(( int(nt),int(N)))  # membrane voltage (in mV) values
  Is = np.zeros(( int(nt),int(N)))  # input currents from the ext_stim
  IPSCs = np.zeros(( int(nt),int(N))) # IPSC over time
  spk = np.zeros(( int(nt),int(N)))   # spikes
  rs = np.zeros(( int(nt),int(N)))    # firing rates
  hs = np.zeros(( int(nt),int(N)))    # filtered firing rates

  # used to set the refractory times
  tlast = np.zeros((int(N))) 

  # Constant bias current to be added to ensure the baseline membrane voltage
  # is around the rheobase
  if hyp['type'] == 'LIF':
    BIAS = hyp['BIAS'] # for linear LIF
  else:
    BIAS = 0; # for quadratic LIF


  #------------------------------------------------------
  # Start the simulation
  #------------------------------------------------------

  for i  in range(0,int(nt)):#range(0,int(nt)):

      IPSCs[i,:] = IPSC # record the IPSC over time (comment out if not used to save time)

     
      I = IPSC + BIAS           # synaptic current

      # Apply external input stim if there is any
      I = I + (ext_stim[ i,:])

      Is[i,:] = ext_stim[ i,:]
    
      # LIF voltage equation with refractory period

      if hyp['type'] == 'LIF':
        dv = np.multiply(np.array((dt*i>tlast + tref)),np.array((-v+I)/tm)) # linear LIF
      else:   
        dv = (dt*i>tlast + tref)*(v**2+I)/tm # quadratic LIF
      

      
      v = v+ dt*np.array(dv) #+ np.random.randn(*np.array(dv).shape)/10

      # find the neurons that have fired
      index = np.where(v>=vpeak)[0]  

      # store spike times, and get the weight matrix column sum of spikers
      if len(index)>0:
        JD  = np.sum(W[:,index],1) #compute the increase in current due to spiking

        # tspike[ns:ns+len(index),:] = np.array([index, 0*index+dt*(i)]).T
        ns = ns + len(index)   #total number of psikes so far
      
      # used to set the refractory period of LIF neuron
      tlast = tlast + (dt*(i) -tlast)*(v>=vpeak)
      
      # if the rise time is 0, then use the single synaptic filter,
      # otherwise (i.e. rise time is positive) use the double filter
      if tr == 0:
        
        IPSC = IPSC*pylab.exp(-dt/td)+JD*(len(index)>0)/(td)
        r = r*pylab.exp(-dt/td) + (v>=vpeak)/td
        rs[i ,: ] = r

      else:

        IPSC = IPSC*pylab.exp(-dt/td) + h*dt
        h = h*pylab.exp(-dt/tr) + JD*(len(index)>0)/(tr*td)  #Integrate the current
        hs[i,:] = h

        r = r*pylab.exp(-dt/td) + hr*dt

        hr = hr*pylab.exp(-dt/tr) + (v>=vpeak)/(tr*td)

        rs[i,:] = r
      
      # record the spikes
      spk[i,:] = v >= vpeak
  
      v = v + (v_val - v)*(v>=vpeak)

      # record the membrane voltage tracings from all the units
      REC[i,:] = v 

      #reset with spike time interpolant implemented.
      v = v + (vreset - v)*(v>=vpeak) 
      time = range(1,int(nt))
      
      # Plot the population response

      out = rs@(w_out/hyp['scaling_factor'])

  # Compute average firing rate for each population (excitatory/inhibitory)
  inh_fr = np.zeros( len(inh_ind))
  for i in range(len(inh_fr)): 
      inh_fr[i] = len(np.where(spk[inh_ind[i], :]>0))/T

  exc_fr = np.zeros(len(exc_ind))
  for i in range(len(exc_fr)): 
      exc_fr[i] = len(np.where(spk[exc_ind[i], :]>0))/T

  all_fr = np.zeros( len(inh_ind))
  for i in range(len(inh_fr)): 
      inh_fr[i] = len(np.where(spk[inh_ind[i], :]>0))/T

  all_fr = np.zeros(int(N))
  for i in range(int(N)): 
      all_fr[i] = len(np.where(spk[i, 10:]>0))/T



  return W, REC, spk, rs, all_fr, out, inputs

import matplotlib.pyplot as plt
hyp = {'downsample':1 ,
       'dt' : 5e-05 ,
       'vreset' : -65 ,
       'vpeak' : 30 ,
       'v_val' : 1 ,
       'tref' : 0.003 ,
       'tx' : 1,
       'tm':0.009 ,
       'BIAS' : 0,
       'scaling_factor': 70 ,
       'tr' : 0.000001 ,
       'type' : 'QIF'
       }
bit = 1
W, REC, spk, rs, all_fr, out, inp = LIF_net(data, par, u ,hyp)
plt.plot(out[:,bit])
plt.plot(inp['desired_output'][:,0,bit])
# plt.plot(inp['neural_input'][:,0,1])
