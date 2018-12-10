import numpy as np
import matplotlib.pyplot as plt
import pywt
import pylab

from nt_toolbox.general import *
from nt_toolbox.signal import *
from nt_toolbox.perform_wavelet_transf import *

"""
    Load an image and solve the inverse problem
"""

f0 = load_image("DFB_artificial_dataset/im0_blurry.bmp")

plt.figure()
wavelets = 'sym5'
g,args = pywt.dwt2(f0, wavelets)
print(g)
print(np.shape(g))

plt.subplot(221)
imageplot(g)
plt.subplot(222)
imageplot(args[0])
plt.subplot(223)
imageplot(args[1])
plt.subplot(224)
imageplot(args[2])
 

print(f0)
plt.figure()
Jmin = 6
a = perform_wavelet_transf(f0, Jmin, +1);
plot_wavelet(a, Jmin)
plt.show()
