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

def cost(f0, fSpars, Phi):
   
    ## HIGHLY FOIREUX
    """
    # define wavelet transform function
    PsiS = lambda f: perform_wavelet_transf(f,0,+1, ti=0)
   
    n = np.shape(f0)[0]
    J = int(np.log2(n)-1)+1
    
    u = [4^(-J), 4**(-np.floor(np.arange(J+2/3,1,-1/3)))];
    a = PsiS(fSpars)
    print(np.shape(a))
    print(u)
    print(np.shape(np.arrays(u)))
    Ja = sum([a[i][j] * u[i] for i in range(np.shape(a)[0])] for j in range(np.shape(a)[0]))

    #evaluate the solution with J; L1/L2 and ^2 penalties
    C = Ja + np.sum(np.ravel((f0-Phi(fSpars))**2))/2
    return C
    """
    return 0

def deconvolution(f, kernel, scale, options):
    
    # define wavelet transform and inverse wavelet transform functions
    PsiS = lambda f: perform_wavelet_transf(f,3, +1, ti=1)
    Psi  = lambda a:  perform_wavelet_transf(a,3, -1,ti=1)

    #define the soft threshold function applied in the wavelet domain for the optimization
    SoftThresh = lambda x,T: np.multiply(x,np.maximum(0, 1-np.divide(T,np.maximum(np.abs(x),1e-10))))
    SoftThreshPsi = lambda f,T: Psi(SoftThresh(PsiS(f), T))

    #the linear blur operation
    Phi = lambda x: kernel(x, scale)

    #optimization
    lmbd = 0.01
    tau = 1.5
    niter = 20
    a = PsiS(f0)
    for iter in range(niter):
        if(options['verbose']): print(cost(f0, a, Phi))
        
        #gradient step
        a = np.add(a, tau*PsiS(Phi(f0-Phi(Psi(a)))))
        #soft-threshold step
        a = SoftThresh(a, lmbd*tau )
    
    return Psi(a)

def deconvolution_unknown_scale(f, kernel, options):
    scales = np.linspace(1,5,5)

    J = np.zeros(scales.shape)
    
    for i, scale in enumerate(scales):
        #test ith scale
        fSpars = deconvolution(f, kernel, scale, options) 
        J[i] = cost(f0, fSpars, lambda x : kernel(x, scale)) 


        if(options['verbose']): print('Cost for scale %f : %f' % (scale, J[i]))

    #show J results
    if(options['verbose']):
        plt.figure()
        plt.plot(scales,J)
        plt.title('J-Sigma')
        plt.show()

    #optimal scale
    scale = scales[np.argmin(J)] 
    if(options['verbose']): print('Optimal scale : %f' % scalse)

    fSpars, _ = deconvolution(f, kernel, scale, options) 

    return fSpars

f0 = load_image("DFB_artificial_dataset/im1_blurry.bmp")

options = {}
options['verbose'] = True

#fSpars = deconvolution_unknown_scale(f0, gaussian_blur, options)
scale = 5 
fSpars = deconvolution(f0, gaussian_blur, scale, options)

#show blurred image
plt.figure(figsize = (9,9))
plt.subplot(211)
imageplot(f0, 'Image')

#show result
plt.subplot(212)
imageplot(fSpars, 'Image Deconvoluted')
plt.show()
