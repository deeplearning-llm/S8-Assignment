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

       &uarr;----------------skip connection -------------&darr;                        

Convolution Block 1 ---> Convolution Block 2 ---> Transition Block1--> Max Pooling 1 --> Convolution Block 3 --> Convolution Block 4 --> Convolution Block 5 --> Transition Block 2--> Max Pooling 2 ---> Convolution Block 7 --> Convolution Block 8 -->  Convolution Block 9 --> Global Average Pooling(GAP)--> Convolution Block 10

## Model Metrics

<img width="463" alt="image" src="https://github.com/deeplearning-llm/S8-Assignment/assets/135349271/1b418039-e512-414e-b123-22918e591b19">

#### Batch Normalization 
- Training and Testing Accuracy and Losses
![BatchNorm](https://github.com/deeplearning-llm/S8-Assignment/assets/135349271/e6df4f42-0abf-4ec0-af64-bcca53475a5e)

- 10 Missclassfied Images
  ![BatchNorm_images](https://github.com/deeplearning-llm/S8-Assignment/assets/135349271/3974751e-cb77-405f-998d-3ce799678fdc)

#### Layer Normalization
- Training and Testing Accuracy and Losses
![Layer_Norm](https://github.com/deeplearning-llm/S8-Assignment/assets/135349271/5dbc6a0c-5a27-459a-a09e-d52831113b97)

- 10 Missclassfied Images
![Layer_Norm_images](https://github.com/deeplearning-llm/S8-Assignment/assets/135349271/fb567649-562b-423c-9fb4-e598f55118ec)

#### Group Normalization
- Training and Testing Accuracy and Losses
![Group_Norm](https://github.com/deeplearning-llm/S8-Assignment/assets/135349271/d48583cb-919d-4225-9106-a8139582973c)

- 10 Missclassfied Images
![Group_Norm_images](https://github.com/deeplearning-llm/S8-Assignment/assets/135349271/26356182-3f7c-4499-a98f-105d0bb264d4)

Analysis:
In the terms of number of parameteres in descending orders :  Layer Normalization  > Batch Normalization == Group Normalization 
In terms of Training and Testing Curves we see that 



