import numpy as np
import matplotlib.pyplot as plt
import pywt
import pylab
from sklearn.cluster import KMeans

from nt_toolbox.general import *
from nt_toolbox.signal import *
from nt_toolbox.perform_wavelet_transf import *
"""
    Load an image and solve the inverse problem in the MultiScales setting
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

def BlurredLaplacian(f,r):
    k = np.zeros(f.shape)
    k[0,0] = -4; k[0,-1] = 1; k[-1,0] = 1; k[0,1] = 1; k[1,0] = 1;
    k = gaussian_blur(k,r)
    return np.real( pylab.ifft2(pylab.fft2(f) * pylab.fft2(k)) )

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
    tau = 1
    niter = 30
    a = PsiS(f0)
    for iter in range(niter):
        #gradient step
        a = np.add(a, tau*PsiS(Phi(f0-Phi(Psi(a)))))
        #soft-threshold step
        a = SoftThresh(a, lmbd*tau )

        if iter%5==0:
            print("Iteration : "+str(iter)+"\tCost : "+
                str((np.sum((f0-Phi(Psi(a)))**2) + lmbd*np.sum(np.abs(a)))*1e-4) )

    return Psi(a)

def deconvolution_3Planes(f0, kernel, C, scales, options, lmbd=0.01):
    """
        Apply three deconvolution operations for the three different scale and
        reconstruct a full result given the partionning in C
    """
    print("First scale optimization")
    fs0 = deconvolution(f0, kernel, scales[0], options, lmbd)
    print("Second scale optimization")
    fs1 = deconvolution(f0, kernel, scales[1], options, lmbd)
    print("Third scale optimization")
    fs2 = deconvolution(f0, kernel, scales[2], options, lmbd)

    fs = np.zeros(f0.shape)
    fs[C==0] = fs0[C==0]
    fs[C==1] = fs0[C==1]
    fs[C==2] = fs0[C==2]

    return fs

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
    scales = np.linspace(0.1,15,50)

    J = np.zeros(scales.shape)
    for i,scale in enumerate(scales):
        #test ith scale
        fL = BlurredLaplacian(f0, scale)
        J[i] = L1L2(fL)

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


f0 = load_image("DFB_artificial_dataset/im12_blurry.bmp")

options = {}
options['verbose'] = True

#local scale estimation
#construct padded image f0_ for border estimation
rSize = 40
fM = np.mean(f0)
f0_ = np.vstack((fM*np.ones((rSize,f0.shape[1])), f0, fM*np.ones((rSize,f0.shape[1]))))
f0_ = np.hstack((fM*np.ones((f0.shape[0]+2*rSize,rSize)), f0_, fM*np.ones((f0.shape[0]+2*rSize,rSize))))

#compute local scale
D = 32#space between sampled points
scales = np.array(list(range(11)))
S = np.zeros((int(f0.shape[0]/D)+1,int(f0.shape[1]/D)+1,scales.shape[0]))
for i in range(S.shape[0]):
    print("Sampling image : " + ('%.1f' % (100*i*D/f0.shape[0]))+ "%")
    for j in range(S.shape[1]):
        #point position
        A = rSize+i*D
        B = rSize+j*D
        #window
        fi = f0_[A-rSize:A+rSize, B-rSize:B+rSize]
        for k,scale in enumerate(scales):
            #cost measure for kth scale
            S[i,j,k] = L1L2(BlurredLaplacian(fi, scale))

#choose scale with a point wise minimum
C = np.zeros(S[:,:,0].shape)
for i in range(S.shape[0]):
    for j in range(S.shape[1]):
        C[i,j] = np.argmin(S[i,j,:])

#histogram of the best scales
plt.figure()
plt.title("Estimated scales distribution on the image")
plt.hist([scales[int(i)] for i in np.ravel(C)])
plt.xlabel('$\sigma$')
plt.ylabel('number of samples with estimated scale $\sigma$')

#clustering to find the 3 best scales
kmeans = KMeans(n_clusters=3, random_state=0).fit(
    np.array([scales[int(i)] for i in np.ravel(C)]).reshape(-1,1))
#reassignation to the cluster center
scales = [a[0] for a in kmeans.cluster_centers_]
indSort = np.argsort(scales)
scales = np.sort(scales)
print("Chosen scales : "+str(scales))
labels = np.array([indSort[i] for i in kmeans.labels_])
C = labels.reshape(C.shape)

#extrapolation of the sampled results
C_ = C
C = np.zeros((f0.shape[0],f0.shape[1]))
for i in range(f0.shape[0]):
    for j in range(f0.shape[1]):
        C[i,j] = C_[int(np.floor(i/D)),int(np.floor(j/D))]

#show partitionning
fb0 = np.zeros(f0.shape); fb0[C==0] = f0[C==0]
fb1 = np.zeros(f0.shape); fb1[C==1] = f0[C==1]
fb2 = np.zeros(f0.shape); fb2[C==2] = f0[C==2]
plt.figure(figsize=(9,5))
plt.plot()
plt.title("Partition of the image onto the different scales")
plt.subplot(131)
imageplot(fb0)
plt.title("$\sigma$="+('%.1f' % scales[0]))
plt.subplot(132)
imageplot(fb1)
plt.title("$\sigma$="+('%.1f' % scales[1]))
plt.subplot(133)
imageplot(fb2)
plt.title("$\sigma$="+('%.1f' % scales[2]))
plt.show()

#solve the problems at the differents scales
fSpars = deconvolution_3Planes(f0, gaussian_blur, C, scales, options)

#show blurred image
plt.figure(figsize=(9,5))
plt.subplot(121)
imageplot(f0, 'Image')

#show result
plt.subplot(122)
imageplot(fSpars, 'Image Deblurred')
plt.show()
