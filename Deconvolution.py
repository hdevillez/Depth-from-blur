import numpy as np
import matplotlib.pyplot as plt
import pywt
import pylab

from nt_toolbox.general import *
from nt_toolbox.signal import *

"""
    Load an image and solve the inverse problem
"""
f0 = load_image("DFB_artificial_dataset/im0_blurry.bmp")

#wavelet transform and inverse wavelet transform
wavelets = 'sym5'
g,args = pywt.dwt2(f0, wavelets)
PsiS = lambda f: pywt.dwt2(f,wavelets)[0]
Psi = lambda a: pywt.idwt2((a,args), wavelets)

#define the soft threshold function applied in the wavelet domain for the optimization
SoftThresh = lambda x,T: np.multiply(x,np.maximum(0, 1-np.divide(T,np.maximum(np.abs(x),1e-10))))
SoftThreshPsi = lambda f,T: Psi(SoftThresh(PsiS(f), T))

#gradient extraction
Grad = lambda f: (np.abs(f[:,0:end-1]-f[:,1:end]), np.abs(f[0:end-1,:]-f[1:end,:]))

sigmas = np.linspace(1,5,5)
J = np.zeros(sigmas.shape)
for i in range(len(sigmas)):
	#test ith sigma
	print(i+1)
	sigma = sigmas[i]
	
	#the linear blur operation
	Phi = lambda x: gaussian_blur(x, sigma)

	#optimization
	lmbd = 1e-3/len(f0)
	tau = 1.
	niter = 20
	fSpars = f0
	for iter in range(niter):
		#gradient step
		fSpars = np.add( fSpars, tau*Phi(f0-Phi(fSpars)))
		#soft-threshold step
		fSpars = SoftThreshPsi( fSpars, lmbd*tau )
	
	#evaluate the solution with J; L1/L2 and ^2 penalties
	fW = PsiS(fSpars)
	L1 = np.sum(fW)
	L2 = np.sqrt(np.sum(fW**2))
	J[i] += L1/L2
	#J[i] += np.sum(np.abs(np.ravel(PsiS(fSpars))))
	J[i] += np.sum(np.ravel((f0-Phi(fSpars))**2))/5
	
	print(J[i])

#show J results
plt.figure()
plt.plot(sigmas,J)
plt.title('J-Sigma')

#optimal sigma
sigma = sigmas[np.argmin(J)] 
print(sigma)

#evaluation of the solution for this sigma
#the linear blur operation
Phi = lambda x: gaussian_blur(x, sigma)

#optimization
lmbd = 1e-3/len(f0)
tau = 1.
niter = 30
fSpars = f0
for iter in range(niter):
	#gradient step
	fSpars = np.add( fSpars, tau*Phi(f0-Phi(fSpars)))
	#soft-threshold step
	fSpars = SoftThreshPsi( fSpars, lmbd*tau )

#show blurred image
plt.figure(figsize = (9,9))
plt.subplot(211)
imageplot(f0, 'Image')
#show result
plt.subplot(212)
imageplot(fSpars, 'Image Deconvoluted')
plt.show()
