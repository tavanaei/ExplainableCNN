# XCNN: Embedded Encoder-Decoder in Convolutional Networks Towards Explainable AI
------
This paper proposes a new explainable convolutional neural network (XCNN) which represents important and driving visual features of stimuli in an end-to-end model architecture. This network employs encoder-decoder neural networks in a CNN architecture to represent regions of interest in an image based on its category. The proposed model is trained without localization labels and generates a heat-map as part of the network architecture without extra post-processing steps. The experimental results on the CIFAR-10, Tiny ImageNet, and MNIST datasets showed the success of our algorithm (XCNN) to make CNNs explainable while offering a simple architecture that can be reapplied to any CNN classifier.

***
## Implementation Setup:
#### Main libraries
* Python (python=3.7)

* PyTorch (torch==1.4.0)

* Keras (keras==2.2.4)

* Other required libraries (such as cv2, matplotlib, numpy, etc. used in the Training and Test codes)


******
## Files and Folders:
1- **Code**: All codes are here

2- **Models**: Includes trained PyTorch XCNN models for heatmap generation; and Trained Keras VGG-16 models for the Innvestigate Tool

3- **TestImages**: Sample images used in the paper

4- **tinyimagenet_data, mnist_data, cifar_data**: These are the folders that will be used when you download the datasets.

5- **Results.ipynb**: Regenerating the results shown in the paper and to generate new results using new images

*****
## Data Collection:
**MNIST**: Download in the code using Torchvision

**CIFAR-10**: Download in the code using Torchvision

**TinyImageNet** Download from https://tiny-imagenet.herokuapp.com/ in the 'tinyimagenet_data' folder

****

## Training
*Example: CIFAR-10 training*

`python TrainCifar.py -epoch 300 -lr 0.001 -wd 0.000001 -beta 0.9 -cuda -gpu 0 -batch 128 -lr_decay 0.95`

- XCNN model architecture can be changed in the Train files by changing the generator and discriminator kernels, fully connected layers, and other hyperparameters.

- The XCNN model details can be found in XAI.py

- Data.py prepares data for TinyImageNet

- Default hyperparameters are the ones used in the paper

***

## Testing/Demo

Please see the detail in 'Results.ipynb' to generate heatmaps by the XCNN and other methods. Example:

<p> <img src="res.png"> </p>

****
## Paper/Citation:

FirstName(s) Last Name(s) 'Embedded Encoder-Decoder in Convolutional Networks Towards Explainable AI', 2020