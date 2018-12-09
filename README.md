# -LELEC2885-Depth-from-blur-project

-to run Deconvolution.py, the pywavelets toolbox is needed
conda install pywavelets

-Deconvolution.py is the main file

It find the optimum kernel size by minimizing J = L1(wavelet coefficients) / L2(wavelet coefficients)
The minimization is done with an enumerative search over the kernel size, each time solving the inverse problem.

It then re-solve the inverse problem for the optimal kernel size.

Inverse problems are solved using the soft-threshold method.


install cvxopt with glpk support in conda
conda install -c conda-forge glpk=4.60
conda install -c conda-forge cvxopt=1.1.7
