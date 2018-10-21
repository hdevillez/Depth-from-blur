import numpy as np
import matplotlib.pyplot as plt

from nt_toolbox.general import *
from nt_toolbox.signal import *
from nt_toolbox.perform_wavelet_transf import *

"""
    Load an image and solve the inverse problem
"""
f0 = load_image("DFB_artificial_dataset/im0_blurry.bmp")
sigma = 3.

#show blurred image
plt.figure(figsize = (12,12))
plt.subplot(211)
imageplot(f0, 'Image')

#wavelet transform and inverse wavelet transform
Jmax = np.log2(f0.shape[0])-1
Jmin = (Jmax-3)
Psi = lambda a: perform_wavelet_transf(a, Jmin, -1, ti=0)
PsiS = lambda f: perform_wavelet_transf(f, Jmin, +1, ti=0)
#the linear blur
Phi = lambda x: gaussian_blur(x, sigma)

#define the soft threshold function applied in the wavelet domain for the optimization
SoftThresh = lambda x,T: np.multiply(x,np.maximum(0, 1-np.divide(T,np.maximum(np.abs(x),1e-10))))
SoftThreshPsi = lambda f,T: Psi(SoftThresh(PsiS(f), T))

#optimization
lmbd = 1e-3/len(f0)
tau = 1.5
niter = 50
fSpars = f0
for iter in range(niter):
    fSpars = np.add( fSpars, tau*Phi(f0-Phi(fSpars)))
    fSpars = SoftThreshPsi( fSpars, lmbd*tau )

#show result
plt.subplot(212)
imageplot(fSpars, 'Image Deconvoluted')
plt.show()
