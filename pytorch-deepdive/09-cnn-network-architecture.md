# Define Architecture
For a CNN we have two "networks". One trying to generate an image, a sound, or a story. The other one is testing the output of the generation. Is it good? Is it real? Is it good enough?

We look into the Linear class and weights. This is what gets updates as the network learns during the training process. Just assign the attributes in the Network module and we get it inherited. The Module base class sees this and registers our weights as `learnable parameters` automatically. More on that later.

## Parameters vs Arguments
The difference between a parameter and an argument in this context is that the parameter is used inside the function definitions. This can be thought of as our placeholder. The arguments are the actual values passed to the function wehn the function is called. Parameters are like the local variables that live inside the function, and the arguments are the values assigned to these variables from the outside by the caller of the function.

## Specifics to Network Attributes terms may be thrown around loosely
More sprecifically for the Network (the attributes are the layers) the layer names inside the parenthesis of each layer are the parameters and the values specified after the '=' are the arguments.
The term perameter is loosely used and confuses me sometimes, as I try to serarate ambiguity of terms in my head, but there are soooooo many in CS its hard not to get side tracked- and dont start talking about the squirrel's add ethier. 

## Parameters are just...
Placeholders that eventually will get a value in the network. So we may build stubs without them at first and the runtime will supply them as we cuda crunch numbers and gpu cycles, blah blah blah. Just know that we have to find appropriate values for parameters along the way. We may set a few unknows at first and attempt to discover them through training with the network. These get learned.

## Two types of parameters
### Hyperparameters
Manually chosen and "arbitrarily" chosen. We do this as "developers" because we knew they worked well in the past. We know these to be from the CNN as the kernel_size, out_channels, out_features. This is the job of the network designer to choose- not just arbitrary. Thats why that scientist is of high regard. The network does not derive this.

During the training process later we see more of these hyperparamters. Learn. able.

#### kernel_size
Another word for filter. Convolutional filter and convolutional kernel are interchangeable.
Inside the convolutional layer the channels are paired with the convolutional filter to perform the convolution operation.

The filter "convolves" ($10) the input channels and the result of that operation is an output channel. The actual number of filters are set when we set the number of channels.

#### out_channels
Output channels are also known as feature maps. If we are dealing with linear layers we do not call them feature maps because these outputs are only rank-1 tensors. We just call them out_features.
We increase our `out_channels` as we add in additional convolutional layers. For the first conv layer within the Network attributes, the example of 6 out_channels also means we want 6 corresponding filters to exist for that one layer. 

As we increase the number of attribute convolutional layers inside the Network class we normally also increase the number of out_channels.
#### out_features
After we switch to linear layers we shrink our `out_features` as we filter down to our number of output classes we have. We choose these features based arbitrarily on how many nodes we want on our layer.

These setting directly affect the weight tensors in eack layer. This is elaborateds on with learnable parameters after Data Dependent ones next. 

As we discuss data dependent parameters we see a caveat with the output layer- the last linear layer. This is the means to the end.
### Data Dependent Hyperparamters
The values depend on the data. The 2 big parameters are the start of the netwrok and the end of the network- so the first attribute convolutional layer `in_channels` and the last `out_features` of the linear layer.

The first attribute convolutional layer `in_channels` (for images) depend on the number of colors present- so for our example gray-scale should be '1'.

The last `out_features` of the output layer depend on the number of classes that are present in our training set. For the fashion-mnist we know we have 10 different types of clothing (0-9 classes) means that this value should be '10'. These ARE the predictions for each category from th enetwork.

#### Data Dependent flow summed up 
In general the outputs of one layer become the inputs of the next, and so the example below for the fashin-mnist representation with 2 conv layers with 2 corresponding linears mean data is passing through these inevitably to the final output (3rd linear layer).

#### Revisit flatten()
The flatten has to happen as we pass through the conv to linear layer. We need to `flatten()` the data from the conv ones, so for the first linear layer `in_features` argument 12*4*4 the 12 comes from the number of channels from the last conv out and the 4*4 marrys the elements for the transformations. We need the length.
todo
verify
4*4 means output of the last CNN is 4x4 image/filter:
outputSizeOfCov = [(inputSize + 2*pad - filterSize)/stride] + 1

#### Flow
Will be discussed more with the forward() implementation.
## Class Attributes
5 attributes below as layers defined with 2 convolutional layers with 3 parameters, and 3 linear ones with two parameters, with the last linear one bing the outoput layer. Each of them extend the nn.Module class, each MAY have explicitlt defined argument values -the ones below do. 

#### AAARRRRGGGGHHH Ambiguity
Convolutional layers deal with "channels". This is the CNN's in and out- don't confuse with your IoT devices, sensor channels etc. nor the remote proceedure call channels, or the one you drive your boat to not crash in... And with Linear we are talking in and out features. This may be like sentdex features vs labels for manual creating columns for stock predictions. More on that later.
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

## Review the parts

Layer|	Param name|	Param value	The param value is|
|---|---|---|
conv1|	in_channels	1|	the number of color channels in the input image.
conv1|	kernel_size	5|	a hyperparameter.
conv1|	out_channels|	6	a hyperparameter.
conv2|	in_channels	6|	the number of out_channels in previous layer.
conv2|	kernel_size	5|	a hyperparameter.
conv2|	out_channels|	12	a hyperparameter (higher than previous conv layer).
fc1|	in_features	12 * 4 * 4|	the length of the flattened output from previous layer.
fc1|	out_features	120|	a hyperparameter.
fc2|	in_features	120|	the number of out_features of previous layer.
fc2|	out_features	60|	a hyperparameter (lower than previous linear layer).
out|	in_features	60|	the number of out_channels in previous layer.
out|	out_features 10|	the number of prediction classes.

