# Extract, Transform, and Load (ETL) with PyTorch
Think about yet another pipeline. It gets simpler because this starts with a traditional ETL approach.
* Extract data from a data source.
* Transform data into a desirable format.
* Load data into a suitable structure.
Then describe it as a fractal process. This can be applied on a small scale. Start KISS then Enterprise. Build a one off then scale to an entire network of huge systems that handle the individual parts.

## Hammer it home in your head for neural networks
* Prepare the data
* Build the model
* Train the model
* Analyze the model’s results.

For ETL and PyTorch we use:
```py
import torch
import torch.nn as nn # neural networks package
import torch.optim as optim # optimization with SGD and Adam
import torch.nn.functional as F #interface for loss functions and convolutions. how well is well?

import torchvision # access to popular datasets, model architectures, and image transformations
import torchvision.transforms as transforms # for image processing

# standard data science packaging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
#from plotcm import plot_confusion_matrix # a local file for later

import pdb #if you forget this, you better know about rookout

torch.set_printoptions(linewidth=120)

```

## Hammer ETL home again
* Extract – Get the Fashion-MNIST image data from the source.
* Transform – Put our data into tensor form.
* Load – Put our data into an object to make it easily accessible.

PyTorch uses 2 classes to help with this:

`orch.utils.data.Dataset`	An abstract class for representing a dataset.
`torch.utils.data.DataLoader`	Wraps a dataset and provides access to the underlying data.

#### The jist of stock trades
And the two methods that have to be written are next. As we approach the fashion set, we will not have all this to do, as it is built in
```py
class OHLC(Dataset):
def __init__(self,csv_file):
    self.data = pd.read_csv(csv_file)

def __getitem__(slf, index):
    r = self.data.iloc[index]
    label = torch.tensor(r.is_up_day, dtype=torch.long)
    sample = self.normalize(torch.tensor([r.open, r.high, r.low, r.close]))
    return sample, label

def __len__(self):
    return len(self.data)
```
# Know what we are doing in code
In python like java or other similar classing constructs we need to implement the abstract methods so that we may create a custom dataset. We accomplish this by creating a subclass that extends the functionality of the `Dataset` class. This will create the CUSTOM dataset using PyTorch. Extend the `Dataset` class. In other words we are creating a subclass that implements these required methods. When we do this, our new subclass can then be passed to the a PyTorch `DataLoader` object. This is how we "wrap" the dataset and give additional functionality.

# IMPORTANT With this fashion mnist example we do not do the above!
The functionality for this example is built into `torchvision`, including the fashion dataset. The Fashion-MNIST built-in dataset class is doing this behind the scenes. 

* All subclasses of the Dataset class must override __len__, that provides the size of the dataset, and __getitem__, supporting integer indexing in range from 0 to len(self) exclusive.

## Methods required to be implemented: 
* __len__ returns the length of the dataset.
* __getitem__ gets an element from the dataset at a specific index location within the dataset.

## Torchvision
Gives access to the following:
* Datasets (like MNIST and Fashion-MNIST)
* Models (like VGG16)
* Transforms
* Utils

## How Pytorch implements the fashion dataset
This all relates to processing images using deep learning for computer vision tasks. From the arXiv paper we seen that this should be a drop in for the original MNIST dataset. This is where PyTorch only has to manipulate the fetch url. The PyTorch FashionMNIST dataset simply extends the MNIST dataset and overrides the urls found from within the `class FashionMNIST(MNIST):` class which contains a `urls =`

## Explore the package on your computer
Navigate to the torchvsion package: 
* Lib/site-packages/torchvision (might be different on your machine depending on how it was installed)
then from within the directory `code .` to get the vscode instance. Navigate to:
* /datasets/mnist.py

Observing from the top the fashion-mnist class extends the mnist class as we expected. AND we also knew the INTENT of the dataset was to be a drop in replacement for the mnist set and the ONLY requirement was a url swap. This is for the data fetching. 

Other than the URL changes the class definition for the fashion-mnist dataset is the same as the mnist. So we are trying to figure out which class the mnist dataset extends, and where does the mnist class fetch its data from, and find the domain name from where the data is fetched from. Then figure out the significance of the name within the domain name as it refers to CNN's.

## Work with it
Objective: Get a dataset and wrap it with a data loader.
```py
import torch
import torchvision # access to popular datasets, model architectures, and image transformations
import torchvision.transforms as transforms # for image processing

### PyTorch Dataset Class
# We need to get an instance of this like so:
# name it as it relates to the 60K train set instead of the 10K test set

train_set = torchvision.datasets.FashionMNIST(
    root='./data/FashionMNIST' # root argument used to be './data' and updates to torchvision may have changed this, this is where on disk it is and first time using jupyter launch will create .data in CWD 
    ,train=True # data here is for the training set - remember 60k is the training set and 10 k in testing data
    ,download=True # download it if the location we specified doesnt have it
    ,transform=transforms.Compose([ # a composition of transformations that should be formed on the dataset element
        transforms.ToTensor() # because we want them turned into Tensors as images we use this toTensor()
    ])
)
```
Then our parameters here to manipulate:

Parameter |	Description
|---|---|
root|	The location on disk where the data is located.
train|	If the dataset is the training set
download|	If the data should be downloaded.
transform|	A composition of transformations that should be performed on the dataset elements.

## The data loader
The following line wraps the train_set we described above into the Data Loader Object instance. Now we can leverage the loader for tasks that would be rather difficult to load by hand like batch size,thread management, and shuffle capabilities as notables, but there are more in there.
```py
train_loader = torch.utils.data.Dataloader(train_set)

train_loader = torch.utils.data.DataLoader(train_set
    ,batch_size=1000
    ,shuffle=True
)
```
From ETL perspective we did this using torchvision. We Extracted the raw data from the web using the urls. The Transform was the toTensor() Transformation object of the raw data. Then the Load the one line above training set was wrapped into the data loader to give us access to the loaded data in our DESIRED FORMAT. The data loader gives us access to the data AND data querying capabilities. WE can also shuffle and have a batch size that will give us the different types of querying capabilities we may be looking for during the training process. We will want to change the batch size and be able to shuffle which are examples of querying capabilities. This shows the torchvision module provided in pytorch and how we streamline etl tasks. Next we use datasets to access and view individual samples as well as batches of samples.

## Jeff Dean, Senior Level "Fellow" at google
Leads the Brain team, the AI team in mountainview CA
He calls out layers of AI
* Machine learning -learns to be smarter
* Deep learning- as just one type of machine learning
* Artificial intelligence- big project to cereate non-human intelligence

We wish to enable them to learn- to learn how to learn through their observations of the world and to make inferences based off this. Deep learning is a particular field within this. With Deep learning we expose a system to examples of the behavior we want to have, and the examples teach it, and the computer learns how to do this. Deep learning has a particular way to do this that is particularly important. It builds layers of abstractions automatically as part of the learning process, where the lowest level things are like- does this part of the image contain is a little splotch of brown. Then as you go up throught the layers things get more complicated. Is there an ear or a couple things that look like eyes. These features "emerge" automatically as part of the learning process. WE dont tell it how to tell the difference between a cat and a god, we just enable it to learn there are whiskers and are more often found in pictures that resemble- CATS!
