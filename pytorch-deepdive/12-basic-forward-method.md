# Forward()
Method gets called by the __call__() as a result of US calling the object instance directly. This holds true also for 
```py
class Network(nn.Module):
and class layers as attributes
```
as they extend PyTorch fundamental constructs. So from our Network class we have
```py
import torch
import torch.nn as nn
import torch.nn.functional as F
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        #super(Network, self).__init__() was a bug or lack of underscores after __init?
        # 5 layers defined as attributes
        # 2 convolutional 3 linear
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        # 3 linear, fully connected layers, or also referred to as dense
        #PyTorch uses linear (nn.Linear class name)
        self.fc1 = nn.Linear(in_features=12 * 4 * 4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10) # last one is OUTPUT
        
    def forward(self, t):
        # implement the forward pass
        return t
# get an instance by calling the constructor
network = Network()
```
which reveals the forward doing nothing more than returning the same tensor `t` passed in. This will be built out to return the ouput of the network. This means we make use of all the infprmation we place inside the constructor. This means that it is in this forward() method that we define ALL of the networks transformation. This forward method IS the mapping that maps the input tensor to a prediction output tensor.

Above we have 5 inner constructs defined- 2 conv and 3 linear layers. If we count the actual input this becomes a network with 6 layers. The input layer is determined by the input data. If the input tensor data had 3 elements the input layer tensor will have 3 nodes for its `in_features=3`. and we can also think of this as the identity transformation layer and this is the definition for any function f(x) = x. The in is the data out. t=t. Trivial. Often we do not see this layer in Neural Network API's and exist implicitly. This is in the forward. 

```py
def forward(self, t):
    # (1) input layer not hidden
    t = t # the identity that is a given may also take up space, sometimes this is left out
    # (2) hidden conv layer 1 
    t = self.conv1(t) # callable python instance will hit the __call__ for us
    t = F.relu(t) # activation operation
    t = F.max_pool2d(t, kernel_size=2, stride=2)
    # (3)then the second hidden layer convolution
    t = self.conv2(t) # callable python instance will hit the __call__ for us
    t = F.relu(t)
    t = F.max_pool2d(t, kernel_size=2, stride=2)
    # (4) hidden linear flattens before sent to linear layering #12 coming in from the prev conv layer
    t = t.reshape(-1, 12 * 4 * 4) # 4*4 is the H,W for each of the 12 output channels
    t = self.fc1(t)
    t = F.relu(t)
    # (5) no need reshape to flatten just the 2nd  hidden linear layer
    t = self.fc2(t)
    t = F.relu(t)
    # (6) output not hidden
    # when we pass the tensor to our output layer the result will be our prediction tensor
    t = self.out(t)
    #t = F.softmax(t, dim=1) # activation function not needed during training, only to infer and predict after the fact. The cross entropy loss function is used for training.
    return t
```
The first step is to pass the tensor to the first convolution layer. Then to the second. Each has its own collection of weights and set of operations. The weights, again are encapsulated within the nn.module layer class instances, dictated above in the class `Network(nn.Module):` The relu() and max()are just pure operations without weight tensors and can be called directly from the Functional class, hence the 'F' dot there. 

Somtimes we hear pooling operations as pooling layers and .relu() activations as non-linear activation layers, but what makes the layer disctint from these operations is that the layer has weights. These two .relu and .max_pool can just be called operations because there are no weights behind them, but we will view them as operations added to the collection of layer operations. So the above example has 3 operations for the first conv1. 
```py
t = self.conv1(t) # callable python instance will hit the __call__ for us
t = F.relu(t) # activation operation
t = F.max_pool2d(t, kernel_size=2, stride=2) # pooling
```
In an artificial neural network, an activation function is a function that maps a node's inputs to its corresponding output.The activation function does some type of operation to transform the sum to a number that is often times between some lower limit and some upper limit. This transformation is often a non-linear transformation.

## Sidenote on Activation functions
A few examples of inpujt transformations and the intuition of the funtion.  Activation function is biologically inspired by activity in our brains where different neurons fire (or are activated) by different stimuli. Nuerons fire or they do not, and can be represented with zero for no fire and 1 for on fire!
```js
//interject a little js pseudo
if (smell.isPleasant()) {
    neuron.fire();
}
```
### Relu rectified linear unit activation function
Not all activation functions fire between 0 and one, meaning the transformation of the input may not yield values between 0 and 1. It literally does this `ReLU(x) = max(0, x)` where it returns 0 or the original input itself. It removes negatives and represents them with 0. The more positive the neuron is, the more activated it is.
```js
//more pseudo js, why not mix it up
function relu(x) {
    if (x <= 0) {
        return 0;
    } else {
        return x;
    }
}
```

### Sigmoid 
* For negative inputs, sigmoid will transform the input to a number close to zero.
* For positive inputs, sigmoid will transform the input into a number close to one.
* For inputs close to zero, sigmoid will transform the input into some number between zero and one.
* zero is the lower limit, and one is the upper limit.

### Mapping in deep neural networks are more complex than linear functions
An important feature of linear functions is that the composition of two linear functions is also a linear function. This means that, even in very deep neural networks, if we only had linear transformations of our data values during a forward pass, the learned mapping in our network from input to output would also be linear. 

Most activation functions are non-linear on purpose. this way our neural network can compute arbirarilty complex functions. The use of relu is to create such a scenario.

## Back to the forward() implementation

We just need to know which operations are backed by weights. These ones will be called layers. The WHOLE network is just a culmination of functions, we are just defining the composition more inside the forward method. As we move into the linear shaping we have 12\*4\*4 where 12 was the input channels coming from the conv layer previous to it. The 4 * 4 is the height and width for each 12 output channels. We started with an image (1,28,28) input tensor and by the time it gets to the linear layer shape it is reduced to H=4 W=4 and this is due to the convolution and pooling operations above it. We will discuss this more. When we pass the tensor to our output layer the result will be our prediction tensor. We know we need 10 prediction classes based off the fashion-mnist 10 classes, so to built the forward() section here we dictate 10 classes. The values inside the 10 components correspond to the prediction value for each of our prediction classes.

For hidden layers we normally use relu for activation. On the output it is softmax as the activation function because there is a single category we are trying to predict. The softmax will return a positive prediction for each of the prediction classes with the probabilities summing to one. 

However during the training process we comment out the softmax due to the loss function. We will be using the cross entropy loss function from the `nn.funtional` class (F.cross_entropy()) that implicitly performs the softmax function on its input. So for now we only return the result of the last linear transformation. 

This means the network will be trained using the softmax operation but will not need to compute the additional operation when the network is being used for inference after the training process is complete. This is how to implement the forward() method for a convolutional neural network in PyTorch.

