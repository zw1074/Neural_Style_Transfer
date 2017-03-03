# Neural Style Transfer
This is my personal project. I found that many successful neural net model (e.g. VGG16/19, GoogleNet) have interesting property. Neural Style is one of application that is built based on this. Many detail can be found [here](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)

## Requirement
python version: 2.7.12 with [Anaconda](https://www.continuum.io/downloads)

Package: [Tensorflow](https://www.tensorflow.org/), [SciPy](https://www.scipy.org/)

VGG16 Weight: [vgg16.npy](https://drive.google.com/file/d/0BzIp01PoYYptNm5vSDhJdXVkMnM/view?usp=sharing)

## How to run
Running is very simple. Just follow this command:
```shell
python Neural_Style.py [-h] [-c CONTENT] [-s STYLE] [-v VGG] [-o OUTPUT] [-l LOOP] [-a ALPHA] [-b BETA] [-w WEIGHT]
```

Here is the detail of the option:
```shell
optional arguments:
  -h, --help            show this help message and exit
  -c CONTENT, --content CONTENT
                        input content image
  -s STYLE, --style STYLE
                        input style image
  -v VGG, --vgg VGG     vgg weight path
  -o OUTPUT, --output OUTPUT
                        output path
  -l LOOP, --loop LOOP  loop time
  -a ALPHA, --alpha ALPHA
                        alpha parameter, big value will focus on content
  -b BETA, --beta BETA  beta parmeter, big value will focus on style
  -w WEIGHT, --weight WEIGHT
                        weight parameter for variance in loss
```
Have fun!

## Result
Original Image:
