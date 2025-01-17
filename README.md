# Many-Body Message Passing Neural Network (MPNN)

The code is accompanying paper published at [link](https://openreview.net/forum?id=aR7R8Odhdx).

This repository contains the experiment codes for the ICML2024 workshop paper "A General Formulation of Many-Body MPNN and Its Mixing Power". We open-source our `ManyBodyMPNNConv` implementation and accompany it with experimental notebooks. For experiment reproduction, run the notebooks ideally with GPUs and CUDA installed in the environment. 

[This notebook](./ManybodyMPNN_SyntheticZINC_OSQ_Playground.ipynb) contains what was not extensively mentioned in the paper, about evaluating the Jacobians of the node pairs from spine graphs from the last layer with respect to the first layer.
