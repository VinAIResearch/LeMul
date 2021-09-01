##### Table of Content

1. [Introduction](#toward-realistic-single-view-3d-object-reconstruction-with-unsupervised-learning-from-multiple-images)
1. [Getting Started](#getting-started)
    - [Datasets](#datasets)
    - [Installation](#installation)
    - [Pretrained models](#pretrained-models)
1. [Experiments](#experiments)
    - [Demo](#demo)
    - [Training & Testing](#training-and-testing)

# Toward Realistic Single-View 3D Object Reconstruction with Unsupervised Learning from Multiple Images

Recovering the 3D structure of an object from a single image is a challenging task due to its ill-posed nature. One approach is to utilize the plentiful photos of the same object category to learn a strong 3D shape prior for the object.
We propose a general framework without symmetry constraint, called **LeMul**, that effectively Learns from Multi-image datasets for more flexible and reliable unsupervised training of 3D reconstruction networks. It employs loose shape and texture consistency losses based on component swapping across views.

<img src="./image/teaser.png" width="800">


Details of the dataset construction, model architecture, and experimental results can be found in [our following paper]().

```
@inproceedings{ho2021lemul,
      title={Toward Realistic Single-View 3D Object Reconstruction with Unsupervised Learning from Multiple Images},
      author={Ho, Long-Nhat and Tran, Anh and Phung, Quynh, and Minh Hoai},
      booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
      year={2021}
}
```
**Please CITE** our paper whenever our datasets or model implementation is used to help produce published results or incorporated into other software.

## Getting Started

### Datasets
1. [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) face dataset. Please download the original images (`img_celeba.7z`) from their [website](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and run `celeba_crop.py` in `data/` to crop the images.
2. Synthetic face dataset generated using [Basel Face Model](https://faces.dmi.unibas.ch/bfm/). This can be downloaded using the script `download_synface.sh` provided in `data/`.
3. Cat face dataset composed of [Cat Head Dataset](http://academictorrents.com/details/c501571c29d16d7f41d159d699d0e7fb37092cbd) and [Oxford-IIIT Pet Dataset](http://www.robots.ox.ac.uk/~vgg/data/pets/) ([license](https://creativecommons.org/licenses/by-sa/4.0/)). This can be downloaded using the script `download_cat.sh` provided in `data/`.
4. [Youtube Faces dataset](http://www.cs.tau.ac.il/~wolf/ytfaces/). This can be found here: [Google Drive](https://drive.google.com/drive/folders/1B1EcY6LXTlYFUPiMERzLxp4HjiOHgtiD?usp=sharing)

Please remember to cite the corresponding papers if you use these datasets.

### Installation:
```
# clone the repo
git clone https://github.com/VinAIResearch/LeMul.git
cd LeMul

# install dependencies
conda env create -f environment.yml
```


### Pretrained Models
Pretrained models can be found here: [Google Drive](https://drive.google.com/drive/folders/1B1EcY6LXTlYFUPiMERzLxp4HjiOHgtiD?usp=sharing)
Please download and place pretrained models in `./pretrained` folder.

## Experiments
### Demo
```
python demo/demo.py --input path-to-cropped-image-folder --result path-to-result-folder --checkpoint path-to-checkpoint.pth
```

*Options*:
- `--gpu`: enable GPU
- `--render_video`: render 3D animations using [neural_renderer](https://github.com/daniilidis-group/neural_renderer) (GPU is required)


### Training and Testing
Check the configuration files in `experiments/` and run experiments, eg:
```
# Training
python run.py --config experiments/train_BFM.yml --gpu 0 --num_workers 4

# Testing
python run.py --config experiments/test_BFM.yml --gpu 0 --num_workers 4
```
