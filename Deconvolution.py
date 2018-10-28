import numpy as np
import matplotlib.pyplot as plt
import pywt

from nt_toolbox.general import *
from nt_toolbox.signal import *

"""
    Load an image and solve the inverse problem
"""
f0 = load_image("DFB_artificial_dataset/im0_blurry.bmp")
sigma = 3.

#show blurred image
plt.figure(figsize = (9,9))
plt.subplot(211)
imageplot(f0, 'Image')

#wavelet transform and inverse wavelet transform
wavelets = 'sym5'
g,args = pywt.dwt2(f0, wavelets)
PsiS = lambda f: pywt.dwt2(f,wavelets)[0]
Psi = lambda a: pywt.idwt2((a,args), wavelets)

#the linear blur
Phi = lambda x: gaussian_blur(x, sigma)

#define the soft threshold function applied in the wavelet domain for the optimization
SoftThresh = lambda x,T: np.multiply(x,np.maximum(0, 1-np.divide(T,np.maximum(np.abs(x),1e-10))))
SoftThreshPsi = lambda f,T: Psi(SoftThresh(PsiS(f), T))

#optimization
lmbd = 1e-3/len(f0)
tau = 1.
niter = 50
fSpars = f0
for iter in range(niter):
    #gradient step
    fSpars = np.add( fSpars, tau*Phi(f0-Phi(fSpars)))
    #soft-threshold step
    fSpars = SoftThreshPsi( fSpars, lmbd*tau )

#show result
plt.subplot(212)
imageplot(fSpars, 'Image Deconvoluted')
plt.show()
