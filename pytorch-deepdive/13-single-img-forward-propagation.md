# Forward Propagation
Process of transformaing an input tensor to an output tensor. At the core a neural network is a function that maps an input tensor to an output tensor. The forward propagation portion is the passing of an input tensor to a network and receiving output from that network. This is really just passing the sample data input to the networks forward method.

Sending the data forward throught the network is the first part. Then during the training process there is back propagation that happens after forward. So our CNN process example forward propagation is the process of passing the image tensor forward through to get the output prediction from the last notes. From data sets and data loaders we can access a single sample image tensor and a batch of image tensors from the data loader. With all of this now defined we should be able to pass the image through the network and receive an image prediction.   

## Predicting with the network: Forward pass
We have to trun off PyTorch’s gradient calculation (derivative) feature. This stops PyTorch from automatically building a computation graph as our tensor flows through the network. Tracking calculations happens in real-time, as the calculations occur. Remember that PyTorch uses a dynamic computational graph. we are turning that off.

Using a computation graph it may keep track of the network mapping by tracking each computation that happens. The graph is used during the training process to calculate the derivative (gradient) of the loss function with respect to the network’s weights.

We do not require gradient cqlculations (derivatives) until we train, and when we train we plan on updating weights, which means we will turn it back on when we are ready to train. Turning it off is not necessary but having it off does reduce memory consumption since the graph is stored in memory. This turns it off:

```py
torch.set_grad_enabled(False) 
# <torch.autograd.grad_mode.set_grad_enabled at 0x17c4867dcc0>
```
Now lets start the whole process with all the imports.
```py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

torch.set_printopotions(linewidth=120)
# get the dataset from the package
train_set = torchvision.datasets.FashionMNIST(
    root = './data/FashionMNIST' # watch where the first launch of jupyter is- it may make it there
    ,train=True
    ,download=True
    ,transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

class Network(nn.Module):
    def __init__():
    super().__init__()
    #define the convolution layers first
    self.conv1 = nn.Conv2d(in_channels = 1, out_channnels = 6, kernel_size =5)
    self.conv2 = nn.Conv2d(in_channels = 6, out_channnels = 12, kernel_size =5)
    #now the tensor flattens, and linear layers should shrink it
    self.fc1 = nn.Linear(in_features = 12*4*4, out_features=120)
    self.fc2 = nn.Linear(in_features = 120, out_features=60)
    self.fc1 = nn.Linear(in_features = 60, out_features=10)

def forward(self, t):
    # hidden convolutions skipped th t=t input 
    # last time we separated the callable instance sequentially from the relu
    # t = self.conv1(t) # callable python instance will hit the __call__ for us
    # t = F.relu(t) # activation operation
    # t = F.max_pool2d(t, kernel_size=2, stride=2)
    t = F.relu(self.conv1(t)) # now the above is combining 2, and dont call the __call__ yourself
    t = F.max_pool2d(t, kernel_size=2, stride=2)
    t = relu(self.conv2(t))
    t = F.max_pool2d(t, kernel_size=2, stride=2)
    #linear
    t = F.relu(self.fc1(t.reshape(-1, 12*4*4))) # heres the flatten in execution
    t = F.relu(self.fc2(t))
    t = self.out(t)

    return t

# get a usable network instance from the class
network = Network()
# unpack a single sample image, label through deconstruction and then verify the images shape
# the network accepts a batch though, so a batch of size 1? batch size, in channels, height, width
# the modules expect the tensors to be of 4d or rank-4 and this is standard most networks like batches
sample = next(iter(train_set))
#image,label = next(sample) will skip the first?
image, label  = sample
image.shape
#torch.Size([1,28,28]) # now we know single color channel image 28x28 is there
image.unsqueeze(0).shape # to give us the batch with size 1
#torch.Size([1,1,28,28])
#and we still need to shape it to get the prediction
pred = network(image.unsqueeze(0)) # image shape needs to be B,C,H,W
pred # shape is 1x10
#tensor([-0.1234, 0.0987, 0.1717, 0.0420, 0.0036, -0.0520, 0.0955, -0.1469, 0.0777, 0.0187])
pred.shape
# torch.Size([1, 10])
# we have one image in our single iimage "batch" with 10 prediction classes
# to see if the prediction matches the actual image compare the label with the argmax index returned 
label
9 
# the highest prediction value happened at the class represented at the index 2 - a pullover!
# the actual value was an ankle boot
pred.argmax(dim=1)
# tensor([2]) # for the 0.1717, but that isnt an ankle boot its a pullover
# if we wished values to just be probabilities use the softmax from nn.functional package
# this changes all the weights on each instance
net1 = Network()
net1(image.unsqueeze(0)) # go ahead and use the callable instance method

net2 = Network()
net1(image.unsqueeze(0))

pred = network(image.unsqueeze(0))
F.softmax(pred, dim=1)
#tensor([[0.0891, 0.1085, 0.1091, 0,1036, 0.0989, 0.0961, 0.0995, 0.0883, 0.1063, 0.1004]])
F.softmax(pred, dim=1).sum()
```
The prediction here may be expected for weights that are generated randomly when instantiang the classes during the forward(). Most of the predictions were around 10% because our network is guessing, and we have 10 prediction classes froma balanced dataset. So if we guess randomly we would be guessing accurately to about 10%. Also when we create new instances of the network the weights again will be generated randomly and the predictions will be different. if we created 2 different networks the predictions will not be the same.