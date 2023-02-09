<!--
Copyright (C) 2023 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: AGPL-3.0-or-later
-->

# FoldingNet

## Features

This source code package contains the Python/Caffe implementation of our FoldingNet based on [Caffe](http://github.com/BVLC/caffe) for unsupervised deep learning on point clouds (training/testing/visualization).

## Installation

The codes were written and tested under **Ubuntu 14.04** with **CUDA 8.0**, **cuDNN 7.0**, **Python 2.7**, with the following packages:
- gdown>=3.4.6 ([pip install gdown](https://github.com/wkentaro/gdown))
- glog>=0.3.1 ([pip install glog](https://github.com/benley/python-glog))
- pyxis>=0.3.dev0 ([pip install --upgrade https://github.com/vicolab/ml-pyxis/archive/master.zip](https://github.com/vicolab/ml-pyxis))
- numpy>=1.11.0
- matplotlib>=1.5.1
- h5py==2.6.0
- Pillow>=1.1.7
- scipy>=0.17.0
- scikit_learn>=0.19.1
- scikit_image>=0.9.3
- pydot>=1.1.0

It also needs several tools for automatic download and compile dependencies:
- CMake>=3.5.1
- Git>=2.7.4

Setup
-----

1. Assume this project locates at:
```
    /homes/yourID/FoldingNet
```

2. Inside /homes/yourID/FoldingNet, execute the following
```
    python prepare_deps.py
```
which will download, compile, and install our modified Caffe (make sure you have setup your system following [the official Caffe install guide](http://caffe.berkeleyvision.org/installation.html)) in
```
    /homes/yourID/caffe
```
and download our Caffe front-end in
```
    /homes/yourID/caffecup
```

3. Inside /homes/yourID/FoldingNet, execute the following:
```
    python prepare_data.py
```
which will download ModelNet/ShapeNet/ShapeNetPart data in
```
    /homes/yourID/FoldingNet/data
```
and our pre-trained networks and training logs in
```
    /homes/yourID/FoldingNet/experiments
    /homes/yourID/FoldingNet/logs
```

Note: During the above step 2, if you encounter any:
1. **protobuf** related issue, make sure your install python protobuf with the version matched the output of `protoc --version`.

2. **Tkinter** related issue, make sure you install Tkinter by `sudo apt-get install python-tk`.
Then remove the automaticall downloaded caffe and rerun the step 2 command.

## Notes on Network Short Names

|Short Name |Explanation|Note|
|-----------|-----------|----|
|FoldingNet |Basic FoldingNet as described in the paper, without any rotation augmentation||
|noCov      |Without Covariance|According to our updated experiments, without covariance the results seem to be also good, which is different from the results we obtained during the CVPR paper submission.|
|noGM       |Without Graph Max Pooling|Adding graph pooling can make PointNet more robust, as discussed in details in the appendix.|
|OrthoRot   |With random axis-aligned rotation augmentation|Achieved **88.4%** Linear SVM transfer classification accuracy on ModelNet40.|
|RandRot    |With random rotation augmentation|This decreases the network transfer classification performance.|
|3D         |Using 3D regular grid|3D grid has only marginally better reconstruction than 2D for ModelNet/ShapeNet data.|

## Usage

### Training Networks

The codes were tested with an NVIDIA TitanX GPU with 12GB memory.

```
cd /homes/yourID/FoldingNet
# Generate Caffe network files
python FoldingNetOrthoRot.py
# Start training (remove --no-srun if you do use srun)
python FoldingNetOrthoRot.py brew --no-srun
# Clean intermediate files (only save the max-test-loss caffemodel, default on ModelNet40Test)
python FoldingNetOrthoRot.py clean --no-srun
# transfer linear SVMm, default on ModelNet40Test
python FoldingNetOrthoRot.py svm --no-srun
# plot training log (python base.py --help)
python base.py
```

Note that you can change **FoldingNetOrthoRot** into other network short names to train other networks.

### Visualization

```
# interpolation (change aid and bid to id of object instances to be interpolated)
cd /homes/yourID/FoldingNet
python FoldingNetOrthoRot.py interp --aid 16 --bid 25 --vis

# visualize folding profile (change data file inside the script)
cd /homes/yourID/FoldingNet
python vis_folding_profile.py
```

## Citation

If you use the software, please cite the following ([TR2018-042](https://merl.com/publications/TR2018-042)):

```
@inproceedings{Yang2018jun,
author = {Yang, Yaoqing and Feng, Chen and Shen, Yiru and Tian, Dong},
title = {FoldingNet: Point Cloud Auto-encoder via Deep Grid Deformation},
booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
year = 2018,
month = jun,
doi = {10.1109/CVPR.2018.00029},
url = {https://www.merl.com/publications/TR2018-042}
}
```

## Contact

Tim K Marks (<tmarks@merl.com>)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for our policy on contributions.

## License

Released under `AGPL-3.0-or-later` license, as found in the [LICENSE.md](LICENSE.md) file.

All files:

```
Copyright (C) 2017-2018, 2023 Mitsubishi Electric Research Laboratories (MERL).

SPDX-License-Identifier: AGPL-3.0-or-later
```
