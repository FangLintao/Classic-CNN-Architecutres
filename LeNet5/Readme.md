# LeNet5
![image](https://github.com/FangLintao/Classic-CNN-Architecutres/blob/master/LeNet5/images/model.png)
###### reference: "Gradient-Based Learning Applied to Document Recognition",Yann LeCun, Léon Bottou, Yoshua Bengio, Patrick Haffner Published 1998
## Description
LeNet5 architecture, published on paper "Gradient-Based Learning Applied to Document Recognition" in 1998, is mainly used to recognize hand-written numbers from 0-9.However, because of its simple architecture, it is constrained by the aviliability ofcomputing resources.
## Data Loading
Pytorch MINIST

    torchvision.datasets.MNIST(root="./Datasets", train=True, transform = transform, download=True) -> training
    torchvision.datasets.MNIST(root="./Datasets", train=False, transform = transform, download=True) -> testing

## Architecture
From paper, LeNet5 has simple architecture. It has two convolution layers, two subsampling layers and three linear layers. However, in this project, I add additional layers, like batchnorm layers and remove softmax layer in order to achieve the accuracy above 99%.
1. convolution layer one: output features 6, kernel size 5
2. max pooling layer: kernel size 2
3. batch norm layer: features 6
4. convolution layer two: out features 16, kernel_size=5
5. max pooling layer: kernel_size 2
6. batch norm layer: features 16 
7. linear layer: input features 16*5*5, out_features 120
8. linear layer: input features 120, out features 84
9. linear  layer: input features 84, out features 10
## Challenges
By following the architecture  on the paper, the accuracy of this architecture only reaches up to 98%~98.8%, which is close to the 99%. After analyzing, I do some modification on architecture and image processing.
1. Architecture: Batch Normalization is added after each convolution layer in order to avoid polar distribution via data transformation; Softmax is removed
2. image processing: Normalizing image data is usually recommended by mean 0.5 adn std 0.5. However, this just slightly improves the accuracy 0.1%. By setting [mean 0.1307 and std 0.3081](https://nextjournal.com/gkoehler/pytorch-mnist), accuracy is above 99%
## Results
overall, the accuracy in LeNet5 neural network has the maximum [99.25%](https://github.com/FangLintao/Classic-CNN-Architecutres/blob/master/LeNet5/Saved_model/LeNet5_with_accuracy%3D99.25)  
## [Visualize](https://github.com/FangLintao/Classic-CNN-Architecutres/blob/master/Tools/Visualize.py)
by using feature visualize, so that pictures coming from convolution layers can be visualzied

    Visualize.feature_visualize(network, network.layer,layer,training_image)

##### convolution layer one
![image](https://github.com/FangLintao/Classic-CNN-Architecutres/blob/master/LeNet5/images/conv1.png)
##### convolution layer two
![image](https://github.com/FangLintao/Classic-CNN-Architecutres/blob/master/LeNet5/images/conv2.png)  
by using image visualize, so that pictures coming from testing stage can be visualzied

    Visualize.image_visualize(image,prediction)

##### Testing 
![image](https://github.com/FangLintao/Classic-CNN-Architecutres/blob/master/LeNet5/images/testing_results.png)
