import numpy as np
import pylab
from scipy.io import loadmat
import pickle 

def restore(path):
    file = open(path,'rb')
    restore_data = file.read()
    file.close()
    hid= pickle.loads(restore_data,encoding='latin1')
    return(hid)

data=loadmat('/home/joshi/Downloads/weights_mat/weights.mat')

net_dict = restore('/home/joshi/Downloads/weights_mat/model_results.pkl') 
par = net_dict['parameters']
def generate_flip_flop_trial(par):
    np.random.seed(400)

    trial_info = {'neural_input': np.zeros([par['num_time_steps'],par['batch_size'],par['n_output']]),
                  'desired_output':np.zeros([par['num_time_steps'],par['batch_size'],par['n_output']]),
                  'train_mask': np.ones((par['num_time_steps'], par['batch_size'],),dtype=np.float32)}    



    unsigned_inp = par['rng'].binomial(1,par['p_flip'],[par['num_time_steps'],par['batch_size'],par['n_output']])
    unsigned_out = 2*par['rng'].binomial(1,0.5,[par['num_time_steps'],par['batch_size'],par['n_output']]) -1 


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

inputs = generate_flip_flop_trial(par)
N = par['n_hidden']               #float(data['N'][0][0])
w0 =  par['w_rnn0']               #data['w0']
m =   par['EI_matrix']            #data['m']
som_m = par['w_rnn_mask']         #data['som_m']
w = data['w_rnn']
w_in = data['w_in'].flatten()
scaling_factor = 20 #data['opt_scaling_factor'][0][0]
# inh = data['inh'].flatten()
# exc = data['exc'].flatten()
#u = data['u'][0]
# u = np.zeros (201) # input stim
u = inputs
#u[30:50] = 1;
# taus = data['taus'][0]
taus = 50
# taus_gaus = data['taus_gaus'].flatten()
# taus_gaus0 = data['taus_gaus0'].flatten()
w_out = data['w_out']
downsample=1
dt= par['dt']                  #5.00000000000000e-05
use_initial_weights = False

# Shuffle nonzero weights
# if use_initial_weights == True:
#     w = np.dot(w0,m)*som_m
# else:
#     w = np.dot(w,m)*som_m
w = w
print(w)
#Scale the connectivity weights by the optimal scaling factor 
W = w/scaling_factor;

#Inhibitory and excitatory neurons
inh_ind = par['inh_ind']                     #np.where(inh!=0)
exc_ind = np.where(par['EI_list']==-1)[0]     #np.where(exc!=0)

# Input stimulus
u=u[::downsample]
ext_stim = np.matrix(w_in).transpose()*np.matrix(u)

#------------------------------------------------------
# LIF network parameters
#------------------------------------------------------

dt = 0.00005*downsample    # sampling rate
T = (len(u)-1)*dt*100          # trial duration (in sec)
nt = round(T/dt)           # total number of points in a trial
tref = 0.002               # refractory time constant (in sec)
tm = 0.010                 # membrane time constant (in sec)
vreset = -65               # voltage reset (in mV)
vpeak = -40                # voltage peak (in mV) for linear LIF

# Synaptic decay time constants (in sec) for the double-exponential
# synpatic filter
# tr: rising time constant
# td: decay time constants
# td0: initial decay time constants (before optimization)
if len(taus) > 1:
    td = (1./(1+pylab.exp(-taus_gaus))*(taus[1] - taus[0])+taus[0])*5/1000
    td0 = (1./(1+pylab.exp(-taus_gaus0))*(taus[1] - taus[0])+taus[0])*5/1000
    tr = 0.002
else:
    td = taus*5/1000 
    td0 = td
    tr = 0.002

#Synaptic parameters
IPSC = np.zeros(int(N))       # post synaptic current storage variable
h = np.zeros(int(N))          # storage variable for filtered firing rates
r =np.zeros(int(N))           # second storage variable for filtered rates
hr =np.zeros(int(N))          # third variable for filtered rates
JD = IPSC                    # storage variable required for each spike time
tspike = np.zeros((200*int(nt),2))  # storage variable for spike times
ns = 0                       # number of spikes, counts during simulation
#TODA add randomness again
v = vreset + np.random.rand(int(N),1)*(30-vreset) # initialize voltage with random distribtuions
v_ = v                                       # v_ is the voltage at previous time steps
v0 = v                                       # store the initial voltage values

# Record REC (membrane voltage), Is (input currents), 
# spk (spike raster), rs (firing rates) from all the units
REC = np.zeros((int(nt),int(N)))  # membrane voltage (in mV) values
Is = np.zeros((int(N), int(nt)))  # input currents from the ext_stim
IPSCs = np.zeros((int(N), int(nt))) # IPSC over time
spk = np.zeros((int(N), int(nt)))   # spikes
rs = np.zeros((int(N), int(nt)))    # firing rates
hs = np.zeros((int(N), int(nt)))    # filtered firing rates

# used to set the refractory times
tlast = np.zeros(int(N)) 

# Constant bias current to be added to ensure the baseline membrane voltage
# is around the rheobase
BIAS = vpeak # for linear LIF
#BIAS = 0; # for quadratic LIF


#------------------------------------------------------
# Start the simulation
#------------------------------------------------------


for i  in range(0,int(nt)):#range(0,int(nt)):
    print(i)
    
    v=v.flatten()
    IPSCs[:, i] = IPSC # record the IPSC over time (comment out if not used to save time)

    
    I = IPSC.flatten() + BIAS           # synaptic current

    # Apply external input stim if there is any
    I = I + (ext_stim[:, int(np.round((i+1)/100.))]).flatten()
    Is[:, i] = ext_stim[:, int(np.round((i+1)/100.))].flatten()

    # LIF voltage equation with refractory period
    dv = np.multiply((dt*i>tlast + tref),(-v+I)/tm) # linear LIF
    #dv = (dt*i>tlast + tref)*(v.^2+I)/tm; # quadratic LIF
    

    
    v = v+dt*np.array(dv).flatten() + np.random.randn(int(N), 1).flatten()/10.

    # Artificial stimulation/inhibition
    # if strcmpi(stims.mode, 'exc')
    #   if i >= stims.dur(1) & i < stims.dur(2)
    #     if rand < 0.50
    #       v(stims.units) = v(stims.units) + 0.5;
    #     end
    #   end
    # elseif strcmpi(stims.mode, 'inh')
    #   if i >= stims.dur(1) & i < stims.dur(2)
    #     if rand < 0.50
    #       v(stims.units) = v(stims.units) - 0.5;
    #     end
    #   end
    # end

    # find the neurons that have fired
    index = pylab.find(v>=vpeak)  

    # store spike times, and get the weight matrix column sum of spikers
    if len(index)>0:
      JD = JD = np.sum(W[:,index],1) #compute the increase in current due to spiking

      tspike[ns:ns+len(index),:] = np.array([index, 0*index+dt*(i+1)]).transpose()
      ns = ns + len(index)   #total number of psikes so far
    
    # used to set the refractory period of LIF neuron
    tlast = tlast + (dt*(i+1) -tlast)*(v>=vpeak)
    
    # if the rise time is 0, then use the single synaptic filter,
    # otherwise (i.e. rise time is positive) use the double filter
    if tr == 0:
        IPSC = IPSC*pylab.exp(-dt/td)+JD*(len(index)>0)/(td)
        r = r*pylab.exp(-dt/td) + (v>=vpeak)/td
        rs[:, i] = r

    else:
        IPSC = IPSC*pylab.exp(-dt/td) + h*dt
        h = h*pylab.exp(-dt/tr) + JD*(len(index)>0)/(tr*td)  #Integrate the current
        hs[:, i] = h

        r = r*pylab.exp(-dt/td) + hr*dt

        hr = hr*pylab.exp(-dt/tr) + (v>=vpeak)/(tr*td)

        rs[:, i] = r
    # record the spikes
    spk[:, i] = v >= vpeak
 
    v = v + (30 - v)*(v>=vpeak)

    # record the membrane voltage tracings from all the units
    REC[i,:] = v 

    #reset with spike time interpolant implemented.
    v = v + (vreset - v)*(v>=vpeak) 
    time = range(1,int(nt))
    
    # Plot the population response
    out = np.array((np.matrix(w_out/scaling_factor)*rs))[0]

# Compute average firing rate for each population (excitatory/inhibitory)
inh_fr = np.zeros( len(inh_ind[0]))
for i in range(len(inh_fr)): 
    inh_fr[i] = len(pylab.find(spk[inh_ind[0][i], :]>0))/T

exc_fr = np.zeros(len(exc_ind[0]))
for i in range(len(exc_fr)): 
    exc_fr[i] = len(pylab.find(spk[exc_ind[0][i], :]>0))/T

all_fr = np.zeros( len(inh_ind[0]))
for i in range(len(inh_fr)): 
    inh_fr[i] = len(pylab.find(spk[inh_ind[0][i], :]>0))/T

all_fr = np.zeros(int(N))
for i in range(int(N)): 
    all_fr[i] = len(pylab.find(spk[i, 10:]>0))/T

pylab.ion()
pylab.figure()
t = np.arange(dt,T+dt,dt)
pylab.plot(t,out,'k')
pylab.show()

    # return W, REC, spk, rs, all_fr, out


