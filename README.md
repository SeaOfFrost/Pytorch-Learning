# Pytorch-Learning

Programs for networks that I wrote in order to learn the syntax and how to use PyTorch. 

## 1. MNIST Classifier

The first program that I built was the MNIST classifier based on the very helpful tutorials given [here](https://github.com/yunjey/pytorch-tutorial) and [here](https://github.com/pytorch/examples/). 
It is a simple network with two convolutional layers and two fully connected layers and achieved an accuracy of 99.04%

## 2. Fashion-MNIST Classifier

The above program was then modified to tackle the [Fashion-MNIST dataset](https://github.com/zalandoresearch/fashion-mnist). A similar architecture was used and achieved an accuracy of around 88.44%.

## 3. CIFAR-10 Classifer

In order to tackle images with 3 channels (RGB), I also tried incorporating another similar architecture (i.e. two convolutional layers followed by two fully connected layers) for the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html). The rather small network achieves an accuracy of about 62.91%.
