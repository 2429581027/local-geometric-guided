Geometry Guided Deep Surface Normal Estimation
---
Created by [Jie Zhang], [Jun-Jie Cao], [Hai-Rui Zhu], [Dong-Ming Yan], and [Xiu-Ping Liu].


### Introduction
This is the code for Geometry Guided Deep Surface Normal Estimation.
This code is based on the architecture of DeepFit.
It allows to train, test and evaluate our weight prediction models for normal estimation.
We provide the code for train a model or use a pretrained model on your own data.

Please follow the installation instructions below.

Abstract:

We propose a geometry-guided neural network architecture for robust and detail-preserving surface normal estimation for unstructured point clouds. Previous deep normal estimators usually estimate the normal directly from the neighbors of a query point, which lead to poor performance. The proposed network is composed of a weight learning sub-network (WL-Net) and a lightweight normal learning sub-network (NL-Net). WL-Net first predicates point-wise weights for generating an optimized point set (OPS) from the input. Then, NL-Net estimates a more accurate normal from the OPS especially when the local geometry is complex. To boost the weight learning ability of the WL-Net, we introduce two geometric guidance in the network. First, we design a weight guidance using the deviations between the neighbor points and the ground truth tangent plane of the query point. This deviation guidance offers a “ground truth” for weights corresponding to some reliable inliers and outliers determined by the tangent plane. Second, we integrate the normals of multiple scales into the input. Its performance and robustness are further improved without relying on multi-branch networks, which are employed in previous multi-scale normal estimators. Thus our method is more efficient. Qualitative and quantitative evaluations demonstrate the advantages of our approach over the state-of-the-art methods, in terms of estimation accuracy, model size and inference time. Code is available at https://github.com/2429581027/local-geometric-guided.

### Citation
If you find our work useful in your research, please cite our paper:

 [Preprint](https://arxiv.org/abs/2003.10826):

    @article{jie2020,
      title={Geometry Guided Deep Surface Normal Estimation},
      author={Jie Zhang, Jun-Jie Caob, Hai-Rui Zhu, Dong-Ming Yan, Xiu-Ping Liu},
      journal={Computer-Aided Design},
      year={2021}
    }

### Instructions

##### 1. Requirements

Install [PyTorch](https://pytorch.org/).

The code was tested with Python 3.7.3, torch 1.4.0, torchvision 0.5.0, CUDA 10.1.243, and cuDNN 7605 on Ubuntu 18.04.
For a full list of requirements see `requirements.txt`.

##### 3.Train your own model:
Run `get_data.py` to download PCPNet data.

Alternatively, Download the PCPNet data from this [link](http://geometry.cs.ucl.ac.uk/projects/2018/pcpnet/pclouds.zip) and place it in  `../data/pcpnet/` directory.

For convenience, multi-scale least-squares fitting results are saved in .features files. We will release them soon.

To train a model run `train_n_est_pcp.py`.

To train, test and evaluate, you can run individual train, test and evaluation.

### License
See LICENSE file.
