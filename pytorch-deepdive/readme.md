# Convolutional Neural Networks
What are they and how do they affect us? CNN's integrate automatic feature extraction and discriminative classifier in one model, which is the main difference between them and traditional machine learning techniques. This property allows CNN's to learn hierarchical
[representations](https://arxiv.org/pdf/1710.05381.pdf). Just because you know neural networks doesn't mean your tooling employs backward propagation. That is a big deal here, and should not be quickly shunned off as just another set of replicated tooling to do the same thing. 

CNN's are built consisting of fully connected layers and a number of blocks called convolutions, activation function layer, and max pooling [3, 4, 5]. The complex nature of CNN's require significant computational power for training and evaluation of the networks, which is addressed with the help of modern graphical processing units (GPUs).

# PyTorch and deep learning at deeplizard.com
Main functionality is tensors and sitting on top of cuda library and gpu processing. We need to be mindful of hardware here from NVIDIA or get on making our own copy function with Numa Nodes or something similar. Original torch was built around the lua language in 2016. We are revisiting Linear Algebra at its finest to train our convolutional neural networks and employ deep learning techniques to do things such as train the rover to drive itself efficiently underwater, and using some matrix transforms to recognize objects in batch processes in real time. With Tensorflow, PyTorch, and things like scikit-learn we can appreciate the MNIST data set out of the box. The torchvision library will be used here to load training sets into projects with both the MNIST data set and the more complex fashion-mnist-dataset. Keep in mind that PyTorch implementation is an override of the MNIST dataset class, as it is a swap out of the urls to fetch with. See 05-... notes for more detail after getting through the first four foundational notes.

PyTorch was designed as a means to keep programming as pure as possible to programming neural networks. We will also branch off the UI for our ROV with some in browser detection techniques with TensorFlow.js a little later. For now we will focus on deployments with conda, numpy, Quandl, and a few other modules in Python to get us up and running. 

Andrej Karpathy from Tesla spoke about The new paradigm of software development. We train the SVM (or possibly other linear regression algorithms) through Stochastic Gradient Descent with weights instead of C++, nad in this case for the time being in PyTorch. When working with neural networks it is the goal to maintain the data sets that drive input through the model to attain knowledge and a better code base. This is not training a new measurement model. In production it is WAY DEEPER. You maintain the code base.

New feature demands requires iteration and love over time. We do not use the traditional "knobs". We tune the dataset, tune the model class architecture, tune the optimization, all through seemingly countless iterations to train your model.

### The Big Deal
In reality the acquisition of data is often the hardest part of learning. Who created it? How was it formed? What were the transformations to get there during preprocessing? What is the data's intent? Unintentional consequences? Biased results? Nah. Really? How about ethics? Have fun with Dick and Jane... 

## Tensor 
A PyTorch Tensor is very similar to N demensional arrays in numpy. With tensors the GPU support is built in with the call to .cuda() to leverage highly efficient memory calculations.

PyTorch gives more than just the tensors

built-in|description
|---|---|
torch|             top level lib
[torch.nn](https://pytorch.org/docs/stable/nn.html)| nueral layers, weights, cuda(device=None)
torch.autograd|      workhorse deritive and differential Tensor operations.
torch.nn.functional| for loss and activation functions and convolution operations
torch.optim|         SDG and Adam    
torch.utils |        util classes data sets and data loaders
torchvision   |      popular datasets, model architectures, image transforms for computer vision

* Real close to the actual neural network programming from scratch
* PyTorch is geared to stay out the way and learn the neural networking portion
* Writing in PyTorch is extending only standard python classes
* Its debugging in Python not C++ so you can trace it better than other tooling like...

To optimize neural networks we need to calculate derivatives- area under the curve so to speak.
Use computational graphs to graph the function operations that occur on tensors inside neural networks. These are used to compute the derivatives needed to optimize the nn weights.

Tensor has dynamic computational graphs and they happen on the fly as the operations occur, not like static graphs already done before the operation.

As we add data to neural networks and give it time to learn, its like giving our brain more information to process- with no theoretical limitations of what it may do. More data, more time = smarter code...

## Anaconda python package manager

As per any modern revisit or rehash, please open the current documentation and follow that. This is just my notes from 2018. Planning is everything here. 

1. Install Anaconda (Go with latest python)
2. Check homebrew
3. Go to PyTorch for the commands and config options per os distro to 
4. Install PyTorch

* Cuda came with PyTorch then.

## We Are Peers 

Started as a concept in France from 1824 and 1833 and even today we share knowledge deeper with the masses. Ask a 6 year old what ended in 1824 and they might tell you 1823. Revolutionize how we share and learn together. This is not to say that the model of the Professor and student is not a valid way, nor that peer learning was to forward thinking at its inception, because believe me I was thinking it to be more than just a Socrates and Plato ancient means to validate whether or not you graduate on time- by your peers judgement of wisdom. Even if there are political issues at stake, this will not stay unchanged for long. Eventually even the stubborn come around. How long? It is up to us.

