# A Two-stage Spatial Prediction Modeling Approach Based on Graph Neural Networks and Neural Processes
Li-Li Bao, Chun-Xia Zhang, Jiang-She Zhang

This code implemented by Li-Li Bao for Location Embedding Graph Neural Network-Residual Neural Processes (LEGNN-RNP) for Spatial Data is based on PyTorch 1.11.0 Python 3.8. GPU is NVIDIA GeForce RTX 3080.

LEGNN: epoch=300, learning rate=0.001, hidden number=128,  MSE loss. The adjacency matrix A is defined by the k-nearest-neighbor approach.

RNP: epoch=100, learning rate=0.001, hidden number=128,  Maximizing the evidence lower bound (ELBO) of the conditional log-likelihood. 

## The main contributions of this paper:

1. We propose a two-stage spatial prediction modeling approach. Initially, an attention-based location embedding GNN is employed to capture spatial dependencies for spatial data. Subsequently, residual neural processes are employed to estimate the regression residuals' distribution. The final prediction outcome is the sum of the predictions from these two models.
2. Our approach is formally akin to RK, substituting regression models with a location embedding GNN and OK with a neural embedding-based Gaussian Processes (GPs) model in RK interpolation. Similar to Kriging, our model can quantify forecast uncertainty.
3. Experimental results showcase that our two-stage model outperforms existing methods, achieving state-of-the-art results in spatial prediction tasks.
