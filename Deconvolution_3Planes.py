import numpy as np
import matplotlib.pyplot as plt
import pywt
import pylab

from nt_toolbox.general import *
from nt_toolbox.signal import *
from nt_toolbox.perform_wavelet_transf import *

import cvxpy as cp
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

    if scale==0:
        return f0

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
    niter = 50
    a = PsiS(f0)
    for iter in range(niter):
        #if(options['verbose']): print(cost(f0, Psi(a), Phi))

        #gradient step
        a = np.add(a, tau*PsiS(Phi(f0-Phi(Psi(a)))))
        #soft-threshold step
        a = SoftThresh(a, lmbd*tau )

    return Psi(a)

def deconvolution_3Planes(f0, kernel, C, scales, options, lmbd=0.01):
    fs0 = deconvolution(f0, kernel, scales[0], options, lmbd)
    fs1 = deconvolution(f0, kernel, scales[1], options, lmbd)
    fs2 = deconvolution(f0, kernel, scales[2], options, lmbd)

    fs = np.zeros(f0.shape)
    fs[C==0] = fs0[C==0]
    fs[C==1] = fs0[C==1]
    fs[C==2] = fs0[C==2]

    return fs

def deconvolution_3PlanesDecoupled(f0, kernel, C, scales, options, lmbd=0.01):

    # define wavelet transform and inverse wavelet transform functions
    PsiS = lambda f: perform_wavelet_transf(f,3, +1, ti=1)
    Psi  = lambda a:  perform_wavelet_transf(a,3, -1,ti=1)

    #define the soft threshold function applied in the wavelet domain for the optimization
    SoftThresh = lambda x,T: np.multiply(x,np.maximum(0, 1-np.divide(T,np.maximum(np.abs(x),1e-10))))
    SoftThreshPsi = lambda f,T: Psi(SoftThresh(PsiS(f), T))

    #the linear blur operation
    Phi0 = lambda x: kernel(x, scales[0])
    Phi1 = lambda x: kernel(x, scales[1])
    Phi2 = lambda x: kernel(x, scales[2])

    #optimization
    tau = 1.5
    niter = 20
    print(0)
    fc0 = np.mean(f0)*np.ones(f0.shape); fc0[C==0] = f0[C==0]
    a0 = PsiS(fc0)
    for iter in range(niter):
        #gradient step
        a0 = np.add(a0, tau*PsiS(Phi0(( (fc0-Phi0(Psi(a0))) ))))
        #soft-threshold step
        a0 = SoftThresh(a0, lmbd*tau )

    print(1)
    fc1 = np.mean(f0)*np.ones(f0.shape); fc1[C==1] = f0[C==1]
    a1 = PsiS(fc1)
    for iter in range(niter):
        #gradient step
        a1 = np.add(a1, tau*PsiS(Phi1(( (fc1-Phi1(Psi(a1))) ))))
        #soft-threshold step
        a1 = SoftThresh(a1, lmbd*tau )

    print(2)
    fc2 = np.mean(f0)*np.ones(f0.shape); fc2[C==2] = f0[C==2]
    a2 = PsiS(fc2)
    for iter in range(niter):
        #gradient step
        a2 = np.add(a2, tau*PsiS(Phi2(( (fc2-Phi2(Psi(a2))) ))))
        #soft-threshold step
        a2 = SoftThresh(a2, lmbd*tau )

    return Psi(a0)+Psi(a1)+Psi(a2)

def deconvolution_3PlanesFull(f0, kernel, C, scales, options, lmbd=0.01):

    # define wavelet transform and inverse wavelet transform functions
    PsiS = lambda f: perform_wavelet_transf(f,3, +1, ti=1)
    Psi  = lambda a:  perform_wavelet_transf(a,3, -1,ti=1)

    #define the soft threshold function applied in the wavelet domain for the optimization
    SoftThresh = lambda x,T: np.multiply(x,np.maximum(0, 1-np.divide(T,np.maximum(np.abs(x),1e-10))))
    SoftThreshPsi = lambda f,T: Psi(SoftThresh(PsiS(f), T))

    #the linear blur operation
    Phi0 = lambda x: kernel(x, scales[0])
    Phi1 = lambda x: kernel(x, scales[1])
    Phi2 = lambda x: kernel(x, scales[2])

    #optimization
    #lmbd = 0.01
    tau = 1.5
    niter = 20
    a = PsiS(f0)
    for iter in range(niter):
        #if(options['verbose']): print(cost(f0, Psi(a), Phi))
        print(iter)

        #gradient step
        I = Psi(a)
        I0 = np.zeros(I.shape); I0[C==0] = I[C==0]
        I1 = np.zeros(I.shape); I1[C==1] = I[C==1]
        I2 = np.zeros(I.shape); I2[C==2] = I[C==2]
        DI = f0- (Phi0(I0)+Phi1(I1)+Phi2(I2))
        DIb = np.zeros(DI.shape)
        DIb[C==0] = Phi0(DI)[C==0]
        DIb[C==1] = Phi1(DI)[C==1]
        DIb[C==2] = Phi2(DI)[C==2]
        a = np.add(a, tau*PsiS(DIb))
        #soft-threshold step
        a = SoftThresh(a, lmbd*tau )

        I = Psi(a)
        I0 = np.zeros(I.shape); I0[C==0] = I[C==0]
        I1 = np.zeros(I.shape); I1[C==1] = I[C==1]
        I2 = np.zeros(I.shape); I2[C==2] = I[C==2]
        Ib = (Phi0(I0)+Phi1(I1)+Phi2(I2))

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

def deconvolution_unknown_scale(f0, kernel, options):
    scales = np.array([0,1,4])

    J = np.zeros(scales.shape)

    '''
    #first version
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
    '''

    for i,scale in enumerate(scales):
        fL = BlurredLaplacian(f0, scale)
        '''if np.abs(scale-1.012)<0.01 or np.abs(scale-2.8)<0.1:
            imageplot(fL)
            plt.show()'''
        J[i] = L1L2(fL)

    #optimal scale
    scale = scales[np.argmin(J)]
    if(options['verbose']): print('Optimal scale : %f' % scale)

    #fSpars = deconvolution(f, kernel, scale, options)

    return 0,scale

def circular_blur(f,radius):
    radius = radius*1.3
    n = max(f.shape);
    t = np.concatenate( (np.arange(0,n/2+1), np.arange(-n/2,-1)) )
    [Y,X] = np.meshgrid(t,t)
    k = (X**2+Y**2)<=radius**2
    k = k/np.sum(k)
    return np.real( pylab.ifft2(pylab.fft2(f) * pylab.fft2(k)) )


f0 = load_image("DFB_artificial_dataset/im9_blurry.bmp")

options = {}
options['verbose'] = True

_,S = deconvolution_unknown_scale(f0, gaussian_blur, options)
print(S)

rSize = 40

fM = np.mean(f0)
f0_ = np.vstack((fM*np.ones((rSize,f0.shape[1])), f0, fM*np.ones((rSize,f0.shape[1]))))
f0_ = np.hstack((fM*np.ones((f0.shape[0]+2*rSize,rSize)), f0_, fM*np.ones((f0.shape[0]+2*rSize,rSize))))
#imageplot(f0_)
#plt.show()

D = 16
S = np.zeros((int(f0.shape[0]/D)+1,int(f0.shape[1]/D)+1,3))
scales = np.array([1,3,7])
for i in range(S.shape[0]):#range(rSize, f0.shape[0]-rSize,16):
    print(i)
    for j in range(S.shape[1]):#range(rSize, f0.shape[1]-rSize,16):
        A = rSize+i*D
        B = rSize+j*D
        fi = f0_[A-rSize:A+rSize, B-rSize:B+rSize]
        for k,scale in enumerate(scales):
            S[i,j,k] = L1L2(BlurredLaplacian(fi, scale))

print(S.shape)
#S = S[rSize:-rSize,rSize:-rSize,:]
#print(S.shape)
#S = S[list(range(0,512,16)),list(range(0,512,16)),:]
'''
imageplot(S[:,:,0])
plt.show()
imageplot(S[:,:,1])
plt.show()
imageplot(S[:,:,2])
plt.show()
'''

S_ = S
S = np.zeros((f0.shape[0],f0.shape[1],3))
for i in range(f0.shape[0]):
    for j in range(f0.shape[1]):
        #print(str(i)+" "+str(j))
        #print(str(int(np.floor(i/D)))+" "+str(int(np.floor(j/D))))
        S[i,j,:] = S_[int(np.floor(i/D)),int(np.floor(j/D)),:]
        '''+S_[int(np.floor(i/D)),int(np.ceil(j/D)),:]
            +S_[int(np.ceil(i/D)),int(np.floor(j/D)),:]
            +S_[int(np.ceil(i/D)),int(np.ceil(j/D)),:])/4 '''

#simple placement point-wise minimum
C = np.zeros(S[:,:,0].shape)
for i in range(S.shape[0]):
    for j in range(S.shape[1]):
        C[i,j] = np.argmin(S[i,j,:])


fb0 = np.zeros(f0.shape); fb0[C==0] = f0[C==0]
fb1 = np.zeros(f0.shape); fb1[C==1] = f0[C==1]
fb2 = np.zeros(f0.shape); fb2[C==2] = f0[C==2]
imageplot(fb0)
plt.show()
imageplot(fb1)
plt.show()
imageplot(fb2)
plt.show()


fSpars = deconvolution_3Planes(f0, gaussian_blur, C, scales, options)

#show blurred image
plt.figure(figsize=(9,5))
plt.subplot(121)
imageplot(f0, 'Image')

#show result
plt.subplot(122)
imageplot(fSpars, 'Image Deconvoluted')
plt.show()

'''
#integer programming problem assignement
x0 = cp.Variable(S[:,:,0].shape,'x0',boolean=True)
x1 = cp.Variable(S[:,:,1].shape,boolean=True)
x2 = cp.Variable(S[:,:,2].shape,boolean=True)

constraints = [x0+x1+x2<=1]

problem = cp.Problem( cp.Minimize(cp.sum(x0*S[:,:,0])+cp.sum(x1*S[:,:,1])+cp.sum(x2*S[:,:,2])), constraints)
#sol = problem.solve()

cp.glpk.ilp()
print(problem.solve(solver='GLPK_MI')['x0'])
'''

'''
NSample = 1000
S = np.zeros(NSample)
RSize = 150
for i in range(NSample):
    A = np.random.randint(0,f0.shape[0]-RSize)
    B = np.random.randint(0,f0.shape[1]-RSize)
    fi = f0[A:A+RSize,B:B+RSize]
    #imageplot(fi)
    #plt.show()
    _,S[i] = deconvolution_unknown_scale(fi, gaussian_blur, options)

plt.hist(S)
plt.show()
print(np.sort(S)[int(9.*S.shape[0]/10.)])
'''



'''
fSpars = deconvolution_unknown_scale(f0, gaussian_blur, options)

#scale = 4
#fSpars = deconvolution_adaptativeLambda(f0, gaussian_blur, scale, options)

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
'''
