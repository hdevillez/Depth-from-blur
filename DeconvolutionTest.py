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

def L1L2(fSpars):
	#Blind Deconvolution Using a Normalized Sparsity Measure by Krishnan, Tay and Fergus
	#L1/L2
    Grad = lambda f: (np.abs(f[:,0:-1]-f[:,1:]), np.abs(f[0:-1,:]-f[1:,:]))
    G = Grad(fSpars)#gradient of the image solution along x and y
    L1 = np.sum(np.abs(G[0]))+np.sum(np.abs(G[1]))
    L2 = np.sqrt(np.sum(G[0]**2)+np.sum(G[1]**2))

    return L1/L2

def correlation_transform(f,radius):
    # use it to compute an approximate measure of local correlation
    n = max(f.shape);
    t = np.concatenate( (np.arange(0,n/2+1), np.arange(-n/2,-1)) )
    [Y,X] = np.meshgrid(t,t)
    k = (X**2+Y**2)<=radius**2
    C = np.sum(k)
    k = k/np.sum(k)
    #imageplot(k)
    #plt.show()
    return np.real( pylab.ifft2(pylab.fft2(f) * pylab.fft2(k)) ) - f/C

def whiteness(r):
    # measure the whiteness in the residual r, lower is better (more white)
    r = (r-np.mean(r))/np.sqrt(np.var(r))
    return np.abs(np.sum(np.multiply(r, correlation_transform(r, 10))))

def deconvolution(f0, kernel, scale, options, lmbd=0.01):

    # define wavelet transform and inverse wavelet transform functions
    PsiS = lambda f: perform_wavelet_transf(f,3, +1, ti=1)
    Psi  = lambda a:  perform_wavelet_transf(a,3, -1,ti=1)

    #define the soft threshold function applied in the wavelet domain for the optimization
    SoftThresh = lambda x,T: np.multiply(x,np.maximum(0, 1-np.divide(T,np.maximum(np.abs(x),1e-10))))
    SoftThreshPsi = lambda f,T: Psi(SoftThresh(PsiS(f), T))

    #the linear blur operation
    Phi = lambda x: kernel(x, scale)

    #optimization
    #lmbd = 0.01
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

def deconvolution_adaptativeLambda(f, kernel, scale, options):
    """
        make a search on the lambda value to get a residual as decorrelated as possible
        Parameter estimation for blind and nonblind deconvolution using residual whiteness
    """
    lmbds = np.exp(np.linspace(-6,-3,10))
    W = np. zeros(lmbds.shape)
    for i,lmbd in enumerate(lmbds):
        fSpars = deconvolution(f, kernel, scale, options, lmbd=lmbd)
        W[i] = whiteness(kernel(fSpars, scale)-f0)
        print("Lambda : "+str(lmbd)+"\tWhiteness : "+str(W[i]))

    lmbd = lmbds[np.argmin(W)]
    if(options['verbose']): print("Choosen lambda : "+str(lmbd))

    fSpars = deconvolution(f, kernel, scale, options, lmbd=lmbd)

    return fSpars

def deconvolution_unknown_scale(f, kernel, options):
    scales = np.linspace(1,10,10)

    J = np.zeros(scales.shape)

    for i, scale in enumerate(scales):
        #test ith scale
        fSpars = deconvolution(f, kernel, scale, options)
        J[i] = L1L2(fSpars)


        if(options['verbose']): print('Cost for scale %f : %f' % (scale, J[i]))

    #show J results
    if(options['verbose']):
        plt.figure()
        plt.plot(scales,J,'o')
        plt.title('$L_1/L_2$ cost')
        plt.xlabel('$\sigma$')
        plt.ylabel('$L_1/L_2( I(\sigma) )$')
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

def BlurredLaplacian(f,r):
    n = max(f.shape);
    t = np.concatenate( (np.arange(0,n/2+1), np.arange(-n/2,-1)) )
    [Y,X] = np.meshgrid(t,t)
    k = np.zeros(Y.shape)
    k[0,0] = -4; k[0,-1] = 1; k[-1,0] = 1; k[0,1] = 1; k[1,0] = 1;
    k = gaussian_blur(k,r)
    k = k/np.sum(np.abs(k))
    return np.real( pylab.ifft2(pylab.fft2(f) * pylab.fft2(k)) )


f0 = load_image("DFB_artificial_dataset/im8_blurry.bmp")

L1L2tab = np.zeros(15)
for i in range(15):
    fL = BlurredLaplacian(f0,i+1)
    print(L1L2(fL))
    L1L2tab[i] = L1L2(fL)
    #print(np.var(fL))

print("Best L1/L2 : " + str(np.argmin(L1L2tab)+1))
'''
imageplot(fL)
plt.show()
'''

options = {}
options['verbose'] = True


#fSpars = deconvolution_unknown_scale(f0, gaussian_blur, options)

scale = np.argmin(L1L2tab)+1
#fSpars = deconvolution_adaptativeLambda(f0, gaussian_blur, scale, options)
fSpars = deconvolution(f0, gaussian_blur, scale, options)

if False:
    radius = 6
    ftrue = load_image("DFB_artificial_dataset/im2_original.bmp")
    fSpars2 = deconvolution(f0, circular_blur, radius, options)
    blindSSD = SSD(fSpars,ftrue)
    nonBlindSSD = SSD(fSpars2,ftrue)
    errorRatio = nonBlindSSD/blindSSD
    print("Error ratio : " + str(errorRatio))


#show blurred image
plt.figure(figsize=(9,5))
plt.subplot(121)
imageplot(f0, 'Image')

#show result
plt.subplot(122)
imageplot(fSpars, 'Image Deconvoluted')
plt.show()
