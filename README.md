***DeepFit***: 3D Surface Fitting via Neural Network Weighted Least Squares (ECCV 2020 Oral)
---
Created by [Yizhak Ben-Shabat (Itzik)](http://www.itzikbs.com) and [Stephen Gould](http://users.cecs.anu.edu.au/~sgould/) from [ANU](https://www.anu.edu.au/) and [ACRV](https://www.roboticvision.org/) .

<div align="center">
  <a href="https://www.itzikbs.com/" target="blank">
    <img src="doc/ybenshabat.jpg" alt="Yizhak Ben-Shabat (Itzik)">
  </a>
  <a href="https://cecs.anu.edu.au/people/stephen-gould/" target="blank">
    <img src="doc/sgould.jpg" alt="Stephen Gould">
  </a>
</div>

### Introduction
![DeepFit_pipeline](doc/DeepFit_Pipeline.png)
This is the code for unstructured 3D point cloud surface fitting using DeepFit.
It allows to train, test and evaluate our weight prediction models for weighted least squares in the context of normal estimation and principal curvature estimation.
We provide the code for train a model or use a pretrained model on your own data.

Please follow the installation instructions below.

A short YouTube video providing a brief overview of the methods is coming soon.

Abstract:

We propose a surface fitting method for unstructured 3D point clouds. This method, called DeepFit, incorporates a neural network to learn point-wise weights for weighted least squares polynomial surface fitting. The learned weights act as a soft selection for the neighborhood of surface points thus avoiding the scale selection required of previous methods. To train the network we propose a novel surface consistency loss that improves point weight estimation. The method enables extracting normal vectors and other geometrical properties, such as principal curvatures, the latter were not presented as ground truth during training. We achieve state-of-the-art results on a benchmark normal and curvature estimation dataset, demonstrate robustness to noise, outliers and density variations, and show its application on noise removal.

Our [short video](https://www.youtube.com/watch?v=jwZDU6hVUzA&t=9s) (2 minutes) and [extended video](https://www.youtube.com/watch?v=PrlFen2BuaU) (7.5 minutes) presentation are available on YouTUbe. 

### Citation
If you find our work useful in your research, please cite our paper:

 [Preprint](https://arxiv.org/abs/2003.10826):

    @article{ben2020deepfit,
      title={DeepFit: 3D Surface Fitting via Neural Network Weighted Least Squares},
      author={Ben-Shabat, Yizhak and Gould, Stephen},
      journal={arXiv preprint arXiv:2003.10826},
      year={2020}
    }

### Instructions

##### 1. Requirements

Install [PyTorch](https://pytorch.org/).

The code was tested with Python 3.7.3, torch 1.4.0, torchvision 0.5.0, CUDA 10.1.243, and cuDNN 7605 on Ubuntu 18.04.
For a full list of requirements see `requirements.txt`.

#####  2. Estimate normal vectors for your data:

To test DeepFit on your own data. Run the `compute_normals.py` in the `./tutorial` directory.
It allows you to specify the input file path (`.xyz` file), output path for the estimated normals, jet order (1-4), and a mode (use pretrained DeepFit or our pytorch implementation of the classic jet fitting).

To help you get started, we provide a step by step tutorial `./tutorial/DeepFit_tutorial.ipynb` with extended explenations, interactive visualizations and example files.

 ##### 3.Reproduce the results in the paper:
Run `get_data.py` to download PCPNet data.

Alternatively, Download the PCPNet data from this [link](http://geometry.cs.ucl.ac.uk/projects/2018/pcpnet/pclouds.zip) and place it in  `./data/pcpnet/` directory.

To test the model and output all normal estimations for the dataset run `test_n_est.py`. This will export the normal estimations for each file in the provided file list as a `.normals` file.  

To evaluate the results and output a report run `evaluate.py`

To get all of the method's outputs exported (`beta, weights, normals, curvatures`) run `test_c_est.py`.

To evaluate curvature estimation performance run `evaluate_curvatures.py` (after exporting the results).

##### 4.Train your own model:
To train a model run `train_n_est.py`.

To train, test and evaluate run `run_DeepFit_single_experiment.py`.
Alternatively you can run individual train, test and evaluation.

#### Visualization
Click on the link for details on [how to visialize normal vectors on 3D point clouds](http://www.itzikbs.com/how-to-visualize-normal-vectors-on-3d-point-clouds).

For a quick visualization of a single 3D point cloud with the normal vector overlay run the `visualize_normals.m` script provided MATLAB code in `./MATLAB`.

For visualizing all of the PCPNet dataset results and exporting images use `export_visualizations.m`.

 ### License
See LICENSE file.
