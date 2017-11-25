# tensorflow-cnn-finetune

This repo is about finetuning some famous convolutional neural nets for [MARVEL](https://github.com/avaapm/marveldataset2016) dataset (ship image classification) using TensorFlow.

ConvNets:

 * [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
 * [VGGNet](https://arxiv.org/pdf/1409.1556.pdf)
 * [ResNet](https://arxiv.org/pdf/1512.03385.pdf)
 * [TODO] Inception


Requirements:

 * Python 2.7 (Not tested with Python 3)
 * Tensorflow >=1.0
 * NumPy
 * OpenCV2


## Marvel

[MARVEL](https://github.com/avaapm/marveldataset2016) is a dataset contains over 2M ship images collected from shipspotting.com. For image classification in the paper they use 237K images labelled in 26 superclasses.

You can download the whole dataset with [python repo they provided](https://github.com/avaapm/marveldataset2016).

Or you can download just needed images directly from [this dropbox link](https://www.dropbox.com/s/tuzrz8hckxli6x3/marvel-dataset.zip?dl=0).

After downloading the dataset, you need to update the paths `data/train.txt` and `data/val.txt`.

## Custom Dataset

You can update `data/train.txt` and `data/val.txt` files for your custom dataset. The format must be like following:

```
/absolute/path/to/image1.jpg class_index
/absolute/path/to/image2.jpg class_index
...
```

`class_index` must start from `0`.

> Do not forget to pass `--num_classes` flag when running `finetune.py` script.

## Usage

Make sure dataset is downloaded and file paths are updated.

```bash
# Go to related folder that you want to finetune
cd vggnet

# Download the weights
./download_weights.sh

# See finetuning options (there is some difference between them, like dropout or resnet depth)
python finetune.py --help

# Start finetuning
python finetune.py [options]

# You can observe finetuning with the tensorboard (default tensorboard_root_dir is ../training)
tensorboard --logdir ../training
```

## Examples

- [AlexNet](https://github.com/dgurkaynak/marvel-finetuning/blob/master/alexnet/examples.sh)
- [VGGNet](https://github.com/dgurkaynak/marvel-finetuning/blob/master/vggnet/examples.sh)
- [ResNet](https://github.com/dgurkaynak/marvel-finetuning/blob/master/resnet/examples.sh)
