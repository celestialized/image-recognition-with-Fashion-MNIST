# Model = Network
Same schmell. We want our network to ultimately model or approximate a function that maps image inputs to the correct output class.

## Review OOP
Extend the torch.nn.Module PyTorch class using OOP in python. We orient code and data around objects in python. 

Create an object of a class = we call the object an instance of the class. It has methods and attributes. Methods are thought of as code and behaviors while Attributes is the data or described characterists and properties. Both are defined by the class. The class has the internal state- the encapsulated attributes and methods.

```py
class Lizard: #class declaration
    def __init__(self, name): #class constructor (code)
        self.name = name #attribute (data)
    
    def set_name(self, name): #method declaration (code)
        self.name = name #method implementation (code)
``` 
## Self like this.props
We do not have to intatiate a lizard object from the Lizard class passing self. It is handled by python. We just pass the values we nee to be held as state in that instance of the class, as an object to be used later.

`lizard = Lizard('deep')` The BIG "L" is the class. The little "l" is the usable object in code. See the other notes in the project reseach directory that handle calls to super and deeper topics on extending functionality with classes. 

## OOP with `torch.nn` package
We use `import torch.nn as nn` as the means to build a foudational neural network component layer. We need several classes within the package to construct these layers. 
## Multiple layers
Neural Networks are layers that each have two main components:
* A transformation (implemented in code)
* A collection of weights (represented by data)
These are what we see as OOP object layers defined using classes. In code the layers will be objects. From the imported nn package there is an actual class called `Module`, a little ambiguous I know, which is the base class for all of neural network modules which includes layers.
## BIG Deal
So ALL layers in PyTorch extend the `nn.Module` class and inherit all of PyTorch’s built-in functionality within the `nn.Module` class. Even neural networks and layers extend the `nn.Module` class. So to make a new layer we must also extend the `nn.Module` class.
## \#MindBlown
Think of a neural network as one big layer. A composition of functions in itself is a function.
## `nn.Module` have a forward() method- TensorFlow forward pass
This is the way we pass a tensor to the network as input. Every nn.Module has to implement that, and in doing so it becomes the ACTUAL transformation. The tensor flows forward though each layer transformation until the tensor reaches the output layer. How far does one have to go in reading before this is explained simply enough? Each layer has its own transformation code and the tensor passes forward through each layer. The composition of all the individual layer forward() passes defines the overall forward pass transformation for the network.
# REALLY BIG DEAL
The main purpose of the layered transformation is to mold (transform or map) the input to the correct prediction output class, and during the training process, the layer weights (data) are updated in such a way that cause the mapping to adjust to make the output closer to the correct prediction.
## nn.Functional package
Implementing the forward() method of our nn.Module subclass usually involves also using functions from the `nn.Functional` package for network operations needed to build layers. More specifically many of the nn.Module layer classes employ the `nn.functional` functions to perform their operations.
## Hammer it home
The nn.functional package contains methods that subclasses of nn.Module use for implementing their forward() functions. Check out the `nn.Conv2d` convolutional layer class.
## Build it with three basic steps
* Extend the nn.Module base class 
* Define layers as class attributes
  * using pre-built layers from torch.nn in the constructor
* implement the `forward()` method
  * Use the network’s layer attributes as well as operations from the nn.functional API to define the forward pass

Start with a KISS and a dummy layer, but this does not extend the nn.Module. It is a neral network WITHOUT pytorch.
```py
class Network:
    def __init__(self):
        self.layer = None #single dummy layer in the constructur

    def forward(self, t): # takes in the tensor 't' and transforms it (dummy nothing, we're creating a neural netowrk!)
        t = self.layer(t)
        return t
```
## Extend nn.Module get behind the scenes weights
Next add the extended nn.Module in the input and super for the upper class call to make it a pytorch neural network! Also this will allow pytorh to assign the weights in the background contianed in each layer. It becomes helpful in the training process when the weights need to be updated
```py
class Network(nn.Module):
    def __init__(self):
        super().__init()#calls the above module constructor
        self.layer = None #single dummy layer in the constructur

    def forward(self, t): # takes in the tensor 't' and transforms it (dummy nothing, we're creating a neural netowrk!)
        t = self.layer(t)
        return t
```
## Linear Layers and Convolutional layers
To build a CNN we use these two pre-built layers from the `nn` library. This makes for a little more definition to the "Web" we weave. Each of the 5 layers have 2 components- a set of wieghts we dont see and a transformation that is defined inside the forward method

```py
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
## NVIDIA Jensen Huang
Alex with the big band on AI a few years ago 8 layers deep and a million params. The front layers of the convolutional network could discover the essetial patterns we classiify an object automatically WITHOUT writing software

Cambrian Explosion

RNN for sequential patterns CNN for spatial and the combination of the two do really amazing things.

Generative adversarial networks

CNN has two networks. One trying to generate an image, a sound, or a story. The other one is testing the output of the generation. Is it good? Is it real? Is it good enough?

Jim Kwik talks about learning how to learn- fight for your limitations and you get to keep them. 