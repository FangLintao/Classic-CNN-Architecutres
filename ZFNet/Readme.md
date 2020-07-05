# ZFNet
![image](https://github.com/FangLintao/Classic-CNN-Architecutres/blob/master/ZFNet/images/cover.png)  
###### Reference: "Visualizing and Understanding Convolutional Networks",Matthew D. Zeiler, Rob Fergus, Dept. of Computer Science, Courant Institute, New York University
## 1.Brief Introduction
ZFNet shares the same architecture structure with AlexNet but with different parameters. According to its original paper ["Visualizing and Understanding Convolutional Networks"](https://arxiv.org/abs/1311.2901), what makes it special is that in this paper, it introduces a method to visualize how convolution layers work - Deconvolution Network.
## 2. Dataset
In this project, because of limitation on 4G GPU, I choose CIFAR10 with ten classes instaed of ILSVRC-2013 recommended in its paper

    torchvision.datasets.CIFAR10(root="./Datasets",train=True, transform = transform,download=True) -> training
    torchvision.datasets.CIFAR10(root="./Datasets",train=False, transform = transform,download=True) -> testing
    
## 3. ZFNet Architecture
### 3.1 Forward Convolution Architecture
Ⅰ. the first Convolution Block:  
-> convolution layer: output features 96, kernel size 7, stride 2    
-> Relu activation  
-> max pooling layer: kernel size 3, stride 2  

Ⅱ. the second Convolution Block:  
-> convolution layer: output features 256, kernel size 5, stride 2, padding 0  
-> Relu activation   
-> max pooling layer: kernel size 3, stride 2  

Ⅲ. the third Convolution Block:  
-> convolution layer: out features 384, kernel_size 3, stride 1, padding 1  
-> convolution layer: out features 384, kernel_size 3, stride 1, padding 1   
-> convolution layer: out features 384, kernel_size 3, stride 1, padding 1  
-> max pooling layer: kernel size 3, stride 2  

Ⅳ. the Linear Block:  
-> dropout: p 0.5  
-> linear layer: input features 9216, out_features 4096  
-> dropout: p 0.5  
-> linear layer: input features 4096, out features 4096  
-> linear layer: input features 4096, out features 10  

#### 3.1.1 Difference with AlexNet
7x7 filters in ZFNet while 11x11 filters in AlexNet  

    larger filter size will loss more information in image pixels, which lower accuracy in prediction

### 3.2 Deconvolution
Deconvolution is an approach firstly introduced by ZFNet, helping to figure out how convolution works. By using the following codes to implement decovlution architecture  

    from ZFNet.DeConv import DeConv  
    deconv = DeConv()

#### 3.2.1 Principle
decovolution is backward process of normal convolution, which means that parameters in convolution process should not be changed. In ZFNet, the backward convolution should follow the architecture in below picture.   
![image](https://github.com/FangLintao/Classic-CNN-Architecutres/blob/master/ZFNet/images/decov.png)  
##### Deconvolution layers
Unpooling layer is to extract pixels that are fixed by pooling indices, filling layers with these pixels at places determining by polling indices and filling zeors in other places. This is reason why indeconvolution, we only see the objects' appearance rather than other objects in images.  

    Ⅰ. For Pooling Layer:   
      -> Convolution Process : torch.nn.Maxpool2d(kernel_size,stride,padding)   
      -> Deconvolution Process : torch.nn.MaxUnpool2d(kernel_size,stride,padding)   
    Ⅱ. For Activation Layer:  
      -> Convolution Process : torch.nn.ReLU()   
      -> Deconvolution Process : torch.nn.ReLU()  
    Ⅲ。 For Convolution Layer:  
      -> Convolution Process : torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)    
      -> Deconvolution Process : torch.nn.ConvTranspose2d(out_channels, in_channels, kernel_size, stride, padding, bias=False)   

#### 3.2.2 Deconvolution Results   
Ⅰ. Deconvolution in CIFar10 datasets   
##### The reason of unclear and intact appearance of objects:  
images in CIFar10 is way blur, so information in image pixels is smaller than sharp image pixels. In this case, the deconvolution image at conv1 layer cannot form the clear appearance of objects[see the below images coming from ZFNet deconvolution architecture]   
![image](https://github.com/FangLintao/Classic-CNN-Architecutres/blob/master/ZFNet/images/1.png)   
![image](https://github.com/FangLintao/Classic-CNN-Architecutres/blob/master/ZFNet/images/2.png)    
Ⅱ. Deconvolution in sharp dataset  
the follwoing result is from a pretrained model trained on ImageNet-1K dataset over 140 epoch.   
###### Reference: blog ["ZFNet/DeconvNet: Summary and Implementation"](https://hackmd.io/@bouteille/ByaTE80BI)    
![image](https://github.com/FangLintao/Classic-CNN-Architecutres/blob/master/ZFNet/images/good.png)    
the follwoing result is from the model trained in CIFAR10 datasets over 30 epoch. Accuracy is 81.22%    
![image](https://github.com/FangLintao/Classic-CNN-Architecutres/blob/master/ZFNet/images/bad.png)   
##### Analysis&Conlusion
Analysis:

    Ⅰ. ZFNet in both training set and validation set has the same architecture and convolution;  
    Ⅱ. Images in ImageNet-1k are sharp and contain clear information ahout objects while Images in CIFAR10 are blur and objects are unclear;  
    Ⅲ. Deconvolution in CIFAR10 shows more noises which disturb prediction of convolution process than Deconvolution in ImageNet-1k;
    
Conlusion:
The quality of Images have large influence on convolution process, affecting accuracy of detecting objects. 
## 4. Features Visulization   

    Vis = Visualize(categories=categories)   
    Vis.feature_visualize(network, network.layer,"layer",batch)

##### Ⅰ. Features at conv1 layer, showing 5 channels of images   
![image](https://github.com/FangLintao/Classic-CNN-Architecutres/blob/master/ZFNet/images/conv1.png)  
##### Ⅱ. Features at conv2 layer, showing 5 channels of images   
![image](https://github.com/FangLintao/Classic-CNN-Architecutres/blob/master/ZFNet/images/conv2.png)
## Tesing
most of prediction in testing images are correct.
![image](https://github.com/FangLintao/Classic-CNN-Architecutres/blob/master/ZFNet/images/testing.png)

