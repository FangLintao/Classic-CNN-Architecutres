# AlexNet
![image](https://github.com/FangLintao/Classic-CNN-Architecutres/blob/master/AlexNet/images/model.png)
###### Reference: " imagenet-classification-with-deep-convolutional-neural-networks "
## Description
AlexNet architecture, published on paper "imagenet-classification-with-deep-convolutional-neural-networks" in 2012, outperformed all competitors in ImageNet LSVRC-2010, and it is outstanding by reducing the top5 error from 26% to 15.3%, which makes itself such significant history position. AlexNet is similar to LeNet5 in general architecture but it is special because of its deeper CNN layers.   
## Data Loading
In AlexNet training stage, because of limitation on GPU, I choose CIFAR10 with ten classes instaed of ILSVRC-2010 recommended in its papaer
Pytorch CIFAR10

    torchvision.datasets.CIFAR10(root="./Datasets",train=True, transform = transform,download=True) -> training
    torchvision.datasets.CIFAR10(root="./Datasets",train=False, transform = transform,download=True) -> testing

## Architecture
From paper, AlexNet has deeper architecture. It has five convolution layers, three maxsampling layers and three linear layers. However, except deeper convolutional layers, what is special is that Alexnet adapeted Local Response Normalization to processing datasets after ReLu activation layer. Also, according to this paper, dropout layer is attacthed in fully connection layers to avoid overfitting and the situation that parameters overrelies on the neural percetrons.
1. the first Convolution Block:  
-> convolution layer: output features 96, kernel size 11, stride 4    
-> Relu activation  
-> Local Response Normalization: size 5,alpha 0.0001, beta 0.75, k 2  
-> max pooling layer: kernel size 3, stride 2  

2. the second Convolution Block:  
-> convolution layer: output features 256, kernel size 5, stride 1, padding 2  
-> Relu activation  
-> Local Response Normalization: size 5,alpha 0.0001, beta 0.75, k 2  
-> max pooling layer: kernel size 3, stride 2  

3. the third Convolution Block:  
-> convolution layer: out features 384, kernel_size 3, stride 1, padding 1  
-> convolution layer: out features 384, kernel_size 3, stride 1, padding 1   
-> convolution layer: out features 384, kernel_size 3, stride 1, padding 1  
-> max pooling layer: kernel size 3, stride 2  

4. the Linear Block:  
-> dropout: p 0.5  
-> linear layer: input features 9216, out_features 4096  
-> dropout: p 0.5  
-> linear layer: input features 4096, out features 4096  
-> linear layer: input features 4096, out features 10  

## Challenges
For AlexNet, the accuracy is usually below 90% in my training stage. at the beginning, the training accuracy is just around 40%, which is pretty low. However, as training epochs are getting larger, accuracy is climbing up to around 80%. 
## Results
overall, the accuracy in AlexNet neural network has the maximum 80.36% (because the size is more than 100M, I cannot upload to )
![image](https://github.com/FangLintao/Classic-CNN-Architecutres/blob/master/AlexNet/images/loss.png)
## [Visualize](https://github.com/FangLintao/Classic-CNN-Architecutres/blob/master/Tools/Visualize.py)
by using feature visualize, so that pictures coming from convolution layers can be visualzied

    Visualize.feature_visualize(network, network.layer,layer,training_image)

##### convolution layer
on this feature visualization in one of concolution layers, for the same image, different features are extracted from different channels
![image](https://github.com/FangLintao/Classic-CNN-Architecutres/blob/master/AlexNet/images/conv2.png)

    Visualize.image_visualize(image,prediction)

##### Testing 
![image](https://github.com/FangLintao/Classic-CNN-Architecutres/blob/master/AlexNet/images/testing.png)
## Reference
1. "imagenet-classification-with-deep-convolutional-neural-networks",Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton, January 2012, Advances in neural information processing systems,DOI: 10.1145/3065386
