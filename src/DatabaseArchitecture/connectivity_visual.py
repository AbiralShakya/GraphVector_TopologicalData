import numpy as np
img_array = np.load('/Users/abiralshakya/Documents/Research/GraphVectorTopological/kspace_topology_graphs/SG_002/connectivity_matrix.npy')
print(img_array)
from matplotlib import pyplot as plt

plt.imshow(img_array, cmap='gray')
plt.show()