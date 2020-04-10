# PyTorch objects Dataset and a DataLoader.
We have two created obejcts from previous notes.
* train_set is the data of 60k images
* train_loader we wrapped into the data loader
```py
train_set = torchvision.datasets.FashionMNIST( #instance of the fashionmnist class
    root='./data' # or ./data/FashionMNIST is the on disk location?
    ,train=True # we want it to be training data
    ,download=True
    ,transform=transforms.Compose([ #compose class allows a compostion of transformations
        transforms.ToTensor() # happens on the data elements, this case just turning it toa tensor as a SINGLE transformation
    ])
)
train_loader = torch.utils.data.Dataloader(
    train_set, batch_size=10
)
```
From the last one we did not specify the batch size, but here we wish to start seeing more than the default 1 image.
```py
import numpy as np
import matplotlib.pyplot as plt
torch.set_printoptions(linewidth=120)#just to format stdout
```
## Realize the data
Start by using some helper functions.
```py
len(train_set) #tells you the amount of images
#60000 based on the fashion-mnist data
```

### See the labels
The API has changed since 0.2.1
```py
# Before torchvision 0.2.2
train_set.train_labels # is the label tensor that describe the images
# tensor([9, 0, 0, ..., 3, 0, 5])
# Starting with torchvision 0.2.2
train_set.targets
# tensor([9, 0, 0, ..., 3, 0, 5])
# 9 was an ankle boot and a zero (two of them after 9 is at-shirt)

#realize how many of each label exist we use bincount()
# Before torchvision 0.2.2
train_set.train_labels.bincount()
# tensor([6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000])
# Before torchvision 0.2.2
train_set.targets.bincount()
# tensor([6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000]
```
## bin = frequency distribution
The above notes are if you have a version different than observed. Frequency distribution is uniform per bin as all amounts are equal from the fashion dataset
## Class imbalance: Balanced and unbalanced datasets
Its clear the above is balanced, while varying numbers for the samples would mean unbalanced. Imbalance is a common problem.
### There is some deep learning in 07a-handling-class-imbalance notes
I paused here and wanted to know more. This was great, because if you are like me and are curious, it pays dividends to know what is hidden behind the scenes, the history of the formation of tensorvision, the deeper learning nature of CNN's, and how it all is used to train a machine to act like a brain.

##Jeremy Howard from fast.ai
Coming back into the series we see the open excel spreadsheet. oooohkaaaaayyy. The validation set and test set in general should have the same mix or frequency of observations that you will see in production or the real world. So the passe way was to use the math from the spreadsheets instead of learning a really simple language like python and matplotlib? Howard says the training set should have an equal # in each set. The deep dive on how to get those trained numbers is vital, so go read it. he just says replicate the least number one until you get it equal. Wow. He hints to the fact that all these "Scholastic" papers validate why you just oversample and its okay. Well we wish for Aerospace grade efficiency. He does come around and say sklearns random forests have the class weights parameter and for deep learning for the minibatch make sure its not randomly sampled but a stratified sampled so the less common class is picked more often. Thats where the cornell thoughts come in. Just ask yourself why.

## Back to coding
We now jump to the next image in the sample batch. We use the iter() to get us an object that represents a stream of data that we can iterate over. Then using the next() we can get that particular following element. From this call we expect to get a single sample item, but the sample is of type tuple which are image label pairs. enumerate() is pythonic for this in a loop. Remember each element in the train_set are tensors
```py
sample = next(iter(train_set))
len(smaple)#2
type(smaple)#tuple
#so break the tensors off through sequence unpacking "decoposition or deconstructing the object" of the sample- its a python sequence type (as a stream), so we can acess each tenors by sequenced index and assign them as so
image, label = sample
# is easier than creating two tensor objects like this
# index = sample[0]
# label - sample[1]
image.shape # 1 is gray color, 3 would be rgb
# torch.Size([1,28,28])
label.shape #i s a scalar value no shape- its a single value
#torch.Size([])
plt.imshow(image.squeeze(), cmap='gray') #color channel is squeezed off, cmap is color map
print('label:',label)
#label: tensor(9) #for the ankle boot in the preceeding image

# now for the data loader, remember
# display_loader = torch.utils.data.DataLoader(
#     train_set, batch_size=10
# )
# remember that shuffle means the order is different and the label may not be a 9
batch = next(iter(display_loader))
len(batch) # 2
type(batch) # list
images, labels = batch #unpacking is more than one from the loader instance
images.shape
# torch.tensor([10,1,28,28])
labels.shape
# torch.Size([10]) # rank-1 tensor of 10 elements- ten corresponding lables
# then set the batch up to display with matplotlib 
# self explanitory images is the images tensor and nrow is 10 on one row for a single row
grid = torchvision.utils.make_grid(images, nrow=10)
#specify pyplot size
#use numpy to transpose the grid to meet the grid specs that the imshow() requires
plt.figure(figsize=(15,15))
plt.imshow(np.transpose(grid, (1,2,0)))
# plt.imshow(grid.permute(1,2,0)) # can be used in place of transpose

print('labels:', labels)
# labels: tensor([9, 0, 0, 3, 0, 2, 7, 2, 5, 5])
#
#
# From Barry Mitchell on another way to plot images
# the label should appear above the image when displayed (from mapping)
# I think the batch size is jacked up
how_many_to_plot = 20 # break the loop with this

train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=1, shuffle=True #theirs will be only one in the batch as random?
)

mapping = {
    0:'Top', 1:'Trousers', 2:'Pullover', 3:'Dress', 4:'Coat'
    ,5:'Sandal', 6:'Shirt', 7:'Sneaker', 8:'Bag', 9:'Ankle Boot'
}

plt.figure(figsize=(50,50))
for i, batch in enumerate(train_loader, start=1): # nice to enumerate the tuple
    image, label = batch
    plt.subplot(10,10,i)
    fig = plt.imshow(image.reshape(28,28), cmap='gray')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.title(mapping[label.item()], fontsize=28)
    if (i >= how_many_to_plot): break
plt.show()

```
## Andrew Zeitler
Tedx talks scary talking to your friend the computer that cooks your breakfast, plans your day, drives you to work...
