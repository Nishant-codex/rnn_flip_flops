This is the repository for studying Universality and Individuality across recurrent networks. This works was presented as a master thesis for KTH Royal institute of Technology. 

# RNN Dynamics for Flip Flop task
This repository contains the code for studying RNNs which are trained for 3 bits flip_flop tasks. The method of finding the fixed points is based on the work by Golub et. al. 

```
Golub and Sussillo (2018), "FixedPointFinder: A Tensorflow toolbox for identifying and 
characterizing fixedpoints in recurrent neural networks," Journal of Open Source Software, 
3(31), 1003, https://doi.org/10.21105/joss.01003.
```


The work is extended to the rate networks and LIF networks. Two different approaches were tried. First is based on the work by Kim et. al.
```
Kim, Robert, Yinghao Li, and Terrence J. Sejnowski. "Simple framework for constructing 
functional spiking recurrent neural networks." Proceedings of the national academy of 
sciences 116.45 (2019): 22811-22820.
```
In this we trained rate networks on 3-bit Flip-FLop task and transfered the weights on to an LIF model. 

The second approach is based on the work by [Bellec et. al](https://github.com/IGITUGraz/LSNN-official). In this approach a descrete LIF model was trained on the 3 bit FLip-Flop and the state trajectories were observed.  
## Usage
### Training on Cluster 
Since we compared 102 networks in total, we trained these networks in a distributed fashion and compare their fixed point structure and representations with each other. 
To run the discrete networks over the cluster use 'fixed_point_edits/runfiles_edits.py'. Please uncomment the main() portion of the script. 

To run the Rate networks over the cluster please run 'rate_nets/distributed_r_kim/main.py'

To run discrete LIF network please run the script present in 'LIF_Flip_Flop/LIF_setup.py'

### Using notebooks to train single networks
The notebooks folder containes examples for each network type. This can be run using google colab. 

# Sequential MNIST task
Discrete RNNs are trained on Sequential mnist task. Two types of MNIST tasks were used:  

1. We provided each image as a sequence of pixels with a definite batch size, batch_sizex784x1
2. We used the dataset provided by Jong et.al. Please find more about the task and training method [here](https://edwin-de-jong.github.io/blog/isl/incremental-sequence-learning.html). 

The releveant code example can be found in the notebooks folder. 
