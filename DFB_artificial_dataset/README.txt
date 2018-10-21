This folder contains several (artificially created!) blurred images,
 with or without the original image that goes with them,
 with one or several regions of different blurs (depths),
 with or without noise,
 with a known or unkown PSF.
 They are approximatively ordered by "deconvolution difficulty".
 ####################################################################

 Considered kernels: (x,y) are pixel coordinates, c is a normalization constant
 - Gaussian kernel with sigma parameter :
   h = c * exp(-(x^2+y^2)/(2*sigma^2))
 - Circular kernel with radius parameter :
   h = c * (1 if x^2 + y^2 =< radius^2; 0 otherwise)

 ####################################################################

 - im0 : single scene blurred by convolution with Gaussian kernel, sigma = 3, with almost no noise corruption.
 - im1 = single scene blurred by convolution with Gaussian kernel, sigma = 5, with noise corruption (as all the next images).
 - im2 = single scene blurred by convolution with circular kernel, radius = 6.
 - im3 = single scene blurred by convolution with circular kernel, radius = 13.
 - im4 = single scene blurred by convolution with Gaussian kernel, sigma = 11.
 - im5 = single scene blurred by convolution with Gaussian kernel, sigma unknown.
 - im6 = single scene blurred by convolution with Gaussian kernel, sigma unknown.
 - im7 = single scene blurred by convolution with circular kernel, radius unknown.
