# Pass image batch to PyTorch CNN
This time we will push a batch through the network
```py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

torch.set_grad_enabled(False) #turn off the gradient/derivative comp tracking feature for now
# <torch.autograd.grad_mode.set_grad_enabled at 0x11bbf2320>
# torch.set_grad_enabled(True)      
# Out[8]: <torch.autograd.grad_mode.set_grad_enabled at 0x11bc22630>
print(torch.__version__)
print(torchvision.__version__)
#1.1.0
#0.2.2

print
torch.set_printopotions(linewidth=120)
# get the dataset from the package
train_set = torchvision.datasets.FashionMNIST(
    root = './data/FashionMNIST' # watch where you launch jupyter from it may create it there
    ,train=True
    ,download=True
    ,transform=transforms.Compose([
        transforms.ToTensor() #this takes the image data and turns it into a tensor 
    ])
)
data_loader = torch.utils.data.DataLoader( #no underscore data loader is syntax error - 1 assignemt
    train_set # is this ours from above or the package?
    ,batch_size=10
)
#pulls the data from the batch and populate the labels and images from this in a minute
#we either name the variables singular, or in this case plural for images as we know we got 10 here
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
    t = F.relu(self.conv1(t)) # dont call __call__ yourself
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
# unpack batch naming samples images, labels through deconstruction and also verify
batch = next(iter(data_loader))# the above we created a batch size of ten to the data_loader
#if using a single image we need to unsqueeze the image tensor to make a 4d or rank 4 with adding an axis
# this time we have a batch, so it already has the b,c,h,w 4d layout
images, labels = batch
images.shape #[B,C,H,W] rank 4 tensor
# torch.Sie([10,1,28,28])
labels.shape
# torch.Size[10] is just a single axis with length 10
# we are working with a batch so squeeze is not needed to add in a dimension to tensor
# image.unsqueeze(0).shape # single way to give us the batch with size [1,1,28,28]
# pred = network(image.unsqueeze(0)) # old single image way

#now we wish to see some predictions off data, but we have created random values in the background- no weights
predictions = network(images)
predictions.shape
#torch.Size([10, 10]) reflects 10 images each with ten prediction classes
predictions
# tensor(
#     [
#         [ 0.1072, -0.1255, -0.0782, -0.1073,  0.1048,  0.1142, -0.0804, -0.0087,  0.0082,  0.0180],
#         [ 0.1070, -0.1233, -0.0798, -0.1060,  0.1065,  0.1163, -0.0689, -0.0142,  0.0085,  0.0134],
#         [ 0.0985, -0.1287, -0.0979, -0.1001,  0.1092,  0.1129, -0.0605, -0.0248,  0.0290,  0.0066],
#         [ 0.0989, -0.1295, -0.0944, -0.1054,  0.1071,  0.1146, -0.0596, -0.0249,  0.0273,  0.0059],
#         [ 0.1004, -0.1273, -0.0843, -0.1127,  0.1072,  0.1183, -0.0670, -0.0162,  0.0129,  0.0101],
#         [ 0.1036, -0.1245, -0.0842, -0.1047,  0.1097,  0.1176, -0.0682, -0.0126,  0.0128,  0.0147],
#         [ 0.1093, -0.1292, -0.0961, -0.1006,  0.1106,  0.1096, -0.0633, -0.0163,  0.0215,  0.0046],
#         [ 0.1026, -0.1204, -0.0799, -0.1060,  0.1077,  0.1207, -0.0741, -0.0124,  0.0098,  0.0202],
#         [ 0.0991, -0.1275, -0.0911, -0.0980,  0.1109,  0.1134, -0.0625, -0.0391,  0.0318,  0.0104],
#         [ 0.1007, -0.1212, -0.0918, -0.0962,  0.1168,  0.1105, -0.0719, -0.0265,  0.0207,  0.0157]
#     ]
# )

# notice the values are not that high
# we have 10 images in our batch each with 10 prediction classes
# to see if there is a match pass the predictions tensor to argmax specifying the second dimension
# the 2nd dimension is the last dimension in our predictions tensor
# the argmax, again for batched returns the index for each max found per row not just the single max from last single image
predictions.argmax(dim=1)
#then check it against the labels

#to acheive the comparison the equals function is used for element wise equals
predictions.argmax(dim=1).eq(labels) # returns [1,10] with 0 and 1's representing true if matched
#tensor([0, 0, 1, 0, 0, 1, 0, 0, 1, 0], dtype=uint8)
predictions.argmax(dim=1).eq(labels).sum()
# tensor(3) is not that accurate
#then turn this into a function to get your answer with
def get_num_correct(preds, labels):
    return predictions.argmax(dim=1).eq(labels).sum().item()
#try it out
get_num_correct(preds, labels)
# 3