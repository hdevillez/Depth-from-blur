# -LELEC2885-Depth-from-blur-project

-to run Deconvolution.py, the pywavelets toolbox is needed
conda install pywavelets

-Deconvolution.py is the main file for single scale deblurring
-Deconvolution_MultiScale is the main file for multi scale deblurring

It find the optimum kernel size by minimizing J = L1(Gradient of the input) / L2(Gradient of the input)
where the input is the convolution of the blurred image with the laplacian operator and a blurring operator with a given scale.

The minimization is done with an enumerative search over the kernel size.

It then re-solve the inverse problem for the optimal kernel size.

Inverse problems are solved using the soft-threshold method.
