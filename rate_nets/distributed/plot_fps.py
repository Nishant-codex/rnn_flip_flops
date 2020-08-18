from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from matplotlib.collections import LineCollection
from FixedPointStore_dist import FixedPointStore
from FixedPointSearch_dist import FixedPointSearch
# %matplotlib inline
from scipy.io import loadmat
from plot_utils import plot_fps
from sklearn import manifold
from sklearn.metrics import euclidean_distances
from sklearn.decomposition import PCA
import os
def restore(path):
	file = open(path,'rb')
	restore_data = file.read()
	file.close()
	hid= pickle.loads(restore_data,encoding='latin1')
	return(hid)
hid = loadmat(os.getcwd()+'/savedir/weights.mat')['hid']

# hid = restore(os.getcwd()+'/fps_saver_crnn/fps_saver/hid.p')
fps = FixedPointStore(num_inits = 1,
				num_states= 1,
				num_inputs = 1)
dict_d = fps.restore(os.getcwd()+'/leaky_rnn/fps_saver/fixedPoint_unique.p')
fps.__dict__ = dict_d
print(fps.num_inits)
plot_fps(fps,
    # hid,
    plot_batch_idx=range(20),
    plot_start_time= 10)
# plt.show()