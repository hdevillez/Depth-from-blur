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

def SSD(x,y):
    """ Sum of square differences """
    return np.sum((x-y)**2)

def cost(f0, fSpars, Phi):
	#Blind Deconvolution Using a Normalized Sparsity Measure, Krishnan, Tay, Fergus
	#L1/L2
    Grad = lambda f: (np.abs(f[:,0:-1]-f[:,1:]), np.abs(f[0:-1,:]-f[1:,:]))
    G = Grad(fSpars)#gradient of the image solution along x and y
    L1 = np.sum(np.abs(G[0]))+np.sum(np.abs(G[1]))
    L2 = np.sqrt(np.sum(G[0]**2)+np.sum(G[1]**2))

    return L1/L2

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
        #if(options['verbose']): print(cost(f0, Psi(a), Phi))

        #gradient step
        a = np.add(a, tau*PsiS(Phi(f0-Phi(Psi(a)))))
        #soft-threshold step
        a = SoftThresh(a, lmbd*tau )

    return Psi(a)

def deconvolution_unknown_scale(f, kernel, options):
    scales = np.linspace(1,10,10)

    J = np.zeros(scales.shape)

    for i, scale in enumerate(scales):
        #test ith scale
        fSpars = deconvolution(f, kernel, scale, options)
        J[i] = cost(f0, fSpars, lambda x : kernel(x, scale))


        if(options['verbose']): print('Cost for scale %f : %f' % (scale, J[i]))

    #show J results
    if(options['verbose']):
        plt.figure()
        plt.plot(scales,J,'o')
        plt.title('J-Sigma')
        plt.show()

    #optimal scale
    scale = scales[np.argmin(J)]
    if(options['verbose']): print('Optimal scale : %f' % scale)

    fSpars = deconvolution(f, kernel, scale, options)

    return fSpars

def circular_blur(f,radius):
    n = max(f.shape);
    t = np.concatenate( (np.arange(0,n/2+1), np.arange(-n/2,-1)) )
    [Y,X] = np.meshgrid(t,t)
    k = (X**2+Y**2)<=radius**2
    k = k/np.sum(k)
    return np.real( pylab.ifft2(pylab.fft2(f) * pylab.fft2(k)) )


f0 = load_image("DFB_artificial_dataset/im2_blurry.bmp")

options = {}
options['verbose'] = True

#fSpars = deconvolution_unknown_scale(f0, circular_blur, options)

scale = 4
fSpars = deconvolution(f0, circular_blur, scale, options)

if True:
    radius = 6
    ftrue = load_image("DFB_artificial_dataset/im2_original.bmp")
    fSpars2 = deconvolution(f0, circular_blur, radius, options)
    blindSSD = SSD(fSpars,ftrue)
    nonBlindSSD = SSD(fSpars2,ftrue)
    errorRatio = nonBlindSSD/blindSSD
    print("Error ratio : " + str(errorRatio))


#show blurred image
plt.figure(figsize = (9,9))
plt.subplot(211)
imageplot(f0, 'Image')

#show result
plt.subplot(212)
imageplot(fSpars, 'Image Deconvoluted')
plt.show()
