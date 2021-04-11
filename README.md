# tensorflow-cnn-finetune

This repo is about finetuning some famous convolutional neural nets using TensorFlow.

ConvNets:

- [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
- [VGGNet](https://arxiv.org/pdf/1409.1556.pdf)
- [ResNet](https://arxiv.org/pdf/1512.03385.pdf)


Requirements:
- Python 2.7 or 3.x
- Tensorflow 1.x (tested with 1.15.1)
- OpenCV2 (for data augmentation)


## Dataset file

You need to setup two dataset files for training and validation. The format must be like following:

```
/absolute/path/to/image1.jpg class_index
/absolute/path/to/image2.jpg class_index
...
```

`class_index` must start from `0`.

Sample dataset files can be found at [data/train.txt](data/train.txt) and [data/val.txt](data/val.txt).

> Do not forget to pass `--num_classes` flag when running `finetune.py` script.

## AlexNet

Go into `alexnet` folder

```bash
cd alexnet
```

### Finetuning

Download the weights if you hadn't before.

```bash
./download_weights.sh
````

Run the `finetune.py` script with your options.

```bash
python finetune.py \
    --training_file=../data/train.txt \
    --val_file=../data/val.txt \
    --num_classes 26
```

| Option | Default | Description |
|-|-|-|
| `--training_file` | ../data/train.txt | Training dataset file |
| `--val_file` | ../data/val.txt | Validation dataset file |
| `--num_classes` | 26 | Number of classes |
| `--train_layers` | fc8,fc7 | Layers to be finetuned, seperated by commas. Avaliable layers: `fc8`, `fc7`, `fc6`, `conv5`, `conv4`, `conv3`, `conv2`, `conv1` |
| `--num_epochs` | 10 | How many epochs to run training |
| `--learning_rate` | 0.0001 | Learning rate for ADAM optimizer |
| `--dropout_keep_prob` | 0.5 | Dropout keep probability |
| `--batch_size` | 128 | Batch size |
| `--multi_scale` |  | As a preprocessing step, it scalse the image randomly between 2 numbers and crop randomly at network's input size. For example if you set it `228,256`: - Select a random number between 228 and 256 -- S - Scale input image to `S x S` pixels - Crop it 227x227 randomly |
| `--tensorboard_root_dir` | ../training | Root directory to put the training logs and weights |
| `--log_step` | 10 | Logging period in terms of a batch run |


You can observe finetuning with the tensorboard.

```bash
tensorboard --logdir ../training
```

### Testing a dataset file

At the end of each epoch while finetuning, the current state of the weights are saved into `../training` folder (or any folder you specified with `--tensorboard_root_dir` option). Go to that folder and locate the model and epoch you want to test.

You must have your test dataset file as mentinoned before.

```bash
python test.py \
    --ckpt ../training/alexnet_XXXXX_XXXX/checkpoint/model_epoch1.ckpt \
    --num_classes 26 \
    --test_file ../data/test.txt
```

| Option | Default | Description |
|-|-|-|
| `--ckpt` |  | Checkpoint path; it must end with ".ckpt" |
| `--num_classes` | 26 | Number of classes |
| `--test_file` | ../data/val.txt | Test dataset file |
| `--batch_size` | 128 | Batch size |

### Predicting a single image

```bash
python predict.py \
    --ckpt ../training/alexnet_XXXXX_XXXX/checkpoint/model_epoch1.ckpt \
    --input_image=/some/path/to/image.jpg
```

| Option | Default | Description |
|-|-|-|
| `--ckpt` |  | Checkpoint path; it must end with ".ckpt" |
| `--num_classes` | 26 | Number of classes |
| `--input_image` |  | The path of input image |

## VGGNet

Go into `vggnet` folder

```bash
cd vggnet
```

### Finetuning

Download the weights if you hadn't before.

```bash
./download_weights.sh
````

Run the `finetune.py` script with your options.

```bash
python finetune.py \
    --training_file=../data/train.txt \
    --val_file=../data/val.txt \
    --num_classes 26
```

| Option | Default | Description |
|-|-|-|
| `--training_file` | ../data/train.txt | Training dataset file |
| `--val_file` | ../data/val.txt | Validation dataset file |
| `--num_classes` | 26 | Number of classes |
| `--train_layers` | fc8,fc7 | Layers to be finetuned, seperated by commas. Avaliable layers: `fc8`, `fc7`, `fc6`, `conv5_1`, `conv5_2`, `conv5_3`, `conv4_1`, `conv4_2`, `conv4_3`, `conv3_1`, `conv3_2`, `conv3_3`, `conv2_1`, `conv2_2`, `conv1_1`, `conv1_2` |
| `--num_epochs` | 10 | How many epochs to run training |
| `--learning_rate` | 0.0001 | Learning rate for ADAM optimizer |
| `--dropout_keep_prob` | 0.5 | Dropout keep probability |
| `--batch_size` | 128 | Batch size |
| `--multi_scale` |  | As a preprocessing step, it scalse the image randomly between 2 numbers and crop randomly at network's input size. For example if you set it `228,256`: - Select a random number between 228 and 256 -- S - Scale input image to `S x S` pixels - Crop it 224x224 randomly |
| `--tensorboard_root_dir` | ../training | Root directory to put the training logs and weights |
| `--log_step` | 10 | Logging period in terms of a batch run |


You can observe finetuning with the tensorboard.

```bash
tensorboard --logdir ../training
```

### Testing a dataset file

At the end of each epoch while finetuning, the current state of the weights are saved into `../training` folder (or any folder you specified with `--tensorboard_root_dir` option). Go to that folder and locate the model and epoch you want to test.

You must have your test dataset file as mentinoned before.

```bash
python test.py \
    --ckpt ../training/vggnet_XXXXX_XXXX/checkpoint/model_epoch1.ckpt \
    --num_classes 26 \
    --test_file ../data/test.txt
```

| Option | Default | Description |
|-|-|-|
| `--ckpt` |  | Checkpoint path; it must end with ".ckpt" |
| `--num_classes` | 26 | Number of classes |
| `--test_file` | ../data/val.txt | Test dataset file |
| `--batch_size` | 128 | Batch size |

### Predicting a single image

```bash
python predict.py \
    --ckpt ../training/vggnet_XXXXX_XXXX/checkpoint/model_epoch1.ckpt \
    --input_image=/some/path/to/image.jpg
```

| Option | Default | Description |
|-|-|-|
| `--ckpt` |  | Checkpoint path; it must end with ".ckpt" |
| `--num_classes` | 26 | Number of classes |
| `--input_image` |  | The path of input image |

## ResNet

Go into `resnet` folder

```bash
cd resnet
```

### Finetuning

Download the weights if you hadn't before.

```bash
./download_weights.sh
````

Run the `finetune.py` script with your options.

```bash
python finetune.py \
    --training_file=../data/train.txt \
    --val_file=../data/val.txt \
    --num_classes 26
```

| Option | Default | Description |
|-|-|-|
| `--resnet_depth` | 50 | ResNet architecture to be used: 50, 101 or 152
| `--training_file` | ../data/train.txt | Training dataset file |
| `--val_file` | ../data/val.txt | Validation dataset file |
| `--num_classes` | 26 | Number of classes |
| `--train_layers` | fc | Layers to be finetuned, seperated by commas. Fully-connected last layer: `fc`, tho whole 5th layer: `scale5`, or some blocks of a layer: `scale4/block6,scale4/block5` |
| `--num_epochs` | 10 | How many epochs to run training |
| `--learning_rate` | 0.0001 | Learning rate for ADAM optimizer |
| `--dropout_keep_prob` | 0.5 | Dropout keep probability |
| `--batch_size` | 128 | Batch size |
| `--multi_scale` |  | As a preprocessing step, it scalse the image randomly between 2 numbers and crop randomly at network's input size. For example if you set it `228,256`: - Select a random number between 228 and 256 -- S - Scale input image to `S x S` pixels - Crop it 224x224 randomly |
| `--tensorboard_root_dir` | ../training | Root directory to put the training logs and weights |
| `--log_step` | 10 | Logging period in terms of a batch run |


You can observe finetuning with the tensorboard.

```bash
tensorboard --logdir ../training
```

### Testing a dataset file

At the end of each epoch while finetuning, the current state of the weights are saved into `../training` folder (or any folder you specified with `--tensorboard_root_dir` option). Go to that folder and locate the model and epoch you want to test.

You must have your test dataset file as mentinoned before.

```bash
python test.py \
    --ckpt ../training/resnet_XXXXX_XXXX/checkpoint/model_epoch1.ckpt \
    --num_classes 26 \
    --test_file ../data/test.txt
```

| Option | Default | Description |
|-|-|-|
| `--ckpt` |  | Checkpoint path; it must end with ".ckpt" |
| `--resnet_depth` | 50 | ResNet architecture to be used: 50, 101 or 152
| `--num_classes` | 26 | Number of classes |
| `--test_file` | ../data/val.txt | Test dataset file |
| `--batch_size` | 128 | Batch size |

### Predicting a single image

```bash
python predict.py \
    --ckpt ../training/resnet_XXXXX_XXXX/checkpoint/model_epoch1.ckpt \
    --input_image=/some/path/to/image.jpg
```

| Option | Default | Description |
|-|-|-|
| `--ckpt` |  | Checkpoint path; it must end with ".ckpt" |
| `--resnet_depth` | 50 | ResNet architecture to be used: 50, 101 or 152
| `--num_classes` | 26 | Number of classes |
| `--input_image` |  | The path of input image |
