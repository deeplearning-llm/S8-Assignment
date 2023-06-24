# S8-Assignment

The Assignment Aims to Building CNN network with a Skip Connection  to Classify Images of the CIFAR10 dataset and is am effort to see the effectiveness of the network for rate of convergence, training and test accuracy and losses for 
1. Batch Normalization
2. Layer Normalization
3. Group Normalization

### Dataset
The images in CIFAR-10 are of size 3x32x32, i.e. 3-channel color images of 32x32 pixels in size. The Images belongs to 10 Classes i.e[‘airplane’, ‘automobile’, ‘bird’, ‘cat’, ‘deer’, ‘dog’, ‘frog’, ‘horse’, ‘ship’, ‘truck’.]

## Image Normalization
All the images in the train and test are standardized to have ranges [-1,1]

## Model Architecture 
The CNN Network has the following network defined :
       &darr;-----------------skip connection -------------&darr;                        

Convolution Block 1 ---> Convolution Block 2 ---> Transition Block1--> Max Pooling 1 --> Convolution Block 3 --> Convolution Block 4 --> Convolution Block 5 --> Transition Block 2--> Max Pooling 2 ---> Convolution Block 7 --> Convolution Block 8 -->  Convolution Block 9 --> Global Average Pooling(GAP)--> Convolution Block 10
