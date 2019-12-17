# dl_graph_connectomes
DL course project

#### Desciption:
This is the final project for ESE 546 Principles of Deep Learning at the University of Pennsylvania, a graduate-level deep learning course focusing on Optimization, Neural Architecture, Learning Theory, MCMC, and Variational Inference. We did our final project on the topic of  Graph Convolutional Networks (GCNs) applied to Neural Connectomes Analysis using novel normalization layers. Our project is based on the previous works of Khosla et al. and Zhang et al., respectively the fundamental works in applying GCNs to the connectome analysis and extracting embeddings for graph structure inference.


#### Goal:
We proposed to address the issue of low test-retest reliability due to noise and the effect of high-dimensionality by using two embedded normalization layers. These layers are effective in helping with convergence and stability of the GCN during training. We have one layer for the nodes and one layer for the embeddings, described in detail in the paper.

#### Dataset:
The dataset we are using is the Austism Brain Imaging Data Exchange (ABIDE). There are 774 subjects, 379 Autism Spectrum Disorder and 355 Normal Controls.

#### Baseline:
(additional baseline methods still running by Chenyang)

#### Result: 
The accuracy of GCN-NE achieved is 68.7%, slightly better than 3D-CNN (68.6%) and significantly better than Ridge Regression (66.7%) and SVD methods (65.3%). We had some issues with the convergence of the basic GCN network and we used variations GCN-B, GCN-E for comparison. The GCN-NE method can achieve better generalizability and stability compared with batch normalization.
