# [PyTorch](https://pytorch.org/docs/stable/index.html) docs
This should be taught along side Calc 3 in school- maybe it already is.
This deepdive will bring you back to martix transforms with linear algebra, and also start to enforce derivatives for pushing data through neural networks.

## Config python before proceeding
Anaconda makes it easy to run this series with conda and environment management. See the [notes](../..//anaconda-and-pycharm/conda-cheatsheet.pdf) to get you going. If you don't want details, look at cloning the base into a newly created environment and make sure you deal with Pytorch and torchvision with the appropriately configured binary if you wish to take advantage of .cuda(). CUDA will require a seprate install for your system and is specific to hw like the Jetson Nano. Better refined conda approach is to figure out how to generate your "requirements" from an existing project and then make a reuseable .yml from that. If you are digging, look at DGX-2, numa nodes, and creating your own kernel for accelerated performance. Then you may seek to write your own graphics processing interface. Just remember- get rich with sharing knowlegde first.

Primary tool inputs outputs transformations all use tensors. Its a mathmatical generalization of other concepts- suppose this to be a 1x6 matrix with some "strings" that have associated "meanings", but in transforming data into tensors, it must be numerical. See the list of transformable types.

number
scalar
array
vector
2d array
matrix

Transform this "data" into two main groups with terms vertically analogous from left to right
"shape" 2x3 looks like

number,array, 2darray  for cs terms
scalar,vector,matrix   for math

Then transform/"reshape" it to (shape) 3 groups of 2|indices needed to access values|
|---|---|
number, scalar |                                         0
array, vector  |                                         1   
2d array, matrix |                                       2
As we go past to the nth array  |   nd-tensor 

This nth is the multidimensional array

## Dimensions are not as clear
One thing to note about the dimension of a tensor is that it differs from what we mean when we refer to the dimension of a vector in a vector space. The dimension of a tensor does not tell us how many components exist within the tensor. If we have a three dimensional vector from three dimensional euclidean space, we have an ordered triple with three components. A three dimensional tensor, however, can have many more than three components. Our two dimensional tensor dd for example has nine components.
```py
> dd = [
[1,2,3],
[4,5,6],
[7,8,9]
]
```
## Indices funtamentals comprise these 3
### Rank
number of dimentsions of a tensor. rank-2 tensor means this all holds true: matrix, 2d array, and its a 2d tensor. rank 2 means we need 2 indices to access all the elements. the rank says how many axises it has 

### Axes
The length of each axis tell us how many indices are available along the axes. Each element of the first axis is an array. And likewise each element on the last axis is a number.

### Shape 
* Defined by the length of the axes
* PyTorch the Size and the shape are the same
* The rank is equal to the length of its shape
`len(t.shape)`

## Reshaping
Happens often. When tensors flow through nn, certain shapes are expected at certain points, and we need to know what this looks like anywhere in the neural network.

use the `t.reshape(1,9)` then verify it `t.reshape(1,9).shape` is like calling Size

## Convolutional Neural Networks and Feature Maps

CNN length typical is 4 [B,C,H,W]
example image input 28x28 pixels or 224x224 bgg16 nn 
H=height
W=width
C=color scale 3 for r,g,b or 1 for greyscale 
[NOTE:] Color only applies for the input, after that point in a neural network it passes through a convelutional layer and the data changes, C is output data as a "channel" afterword.
B= batch and the length tells us how many samples are in the batch.

## Convolution flow
Will alter the height and width and the color channels, and is based off the number of filters in the layer. Suppose we have 3 convelutional filters. The output is called channels of the color input axis. the height and width may change dependiing on the output filter dimensions. Now the output channels are not color channels, we cll these feature maps, which are the ouputs of the convelutions that take place using the input color and the filter. We call it feature as the map represents things like edges that are learned into the output. 

Alex in Ontario wrote the big bang algorithm to modernize ai (from CEO of NVIDIA)

## Back to Tensors

First thing we usually do is data preprocessing routines. These will trandform your data to tensors to fuel the neural network.
```py
import torch
import numpy as np

#use class constructor
t=torch.Tensor()
type(t) #make sure it is
```
We already used attributes rank axis shape specific to ALL tensors. Now we have some pytorch specific attributes:

Attribute|Description|
|---|---|
`t.dtype` |data type float32 - and all data should be uniform same one of type
`t.device` | cpu 
`t.layout`  |torch.strided

then if you wish to allocate the cuda device
`device = torch.device('cuda.0')`
If you call out `cuda.3` and you dont have a 3rd gpu it errors out here index= ...refers to the gpu index

Also know that there cannot be computations on tensors of non uniform type
`t = torch.Tensor([1,2,3])`
type is not same as
`t2 = torch.Tensor([1.,2.,3.])`
`t1 + t2` will fail with error that finds the mismatched type.
Same goes for the device
`t = torch.Tensor([1,2,3])`
`t2 = torch.cuda()`
`t1 + t2` will fail with error that finds the mismatched device

### Creating tensor objects with and without data before hand
Two big concepts to creating tensors: 
* Create them with existing data
  * Remember transforming data to a PyTorch tensor must be numerical for the tensor.
  * Often we are working with numpy arrays here
  * If you wish to see the ways to avoid transformation errors, look past the next set of creation examples but make sure you know how to make them before proceeding.
* Without using predefined functions with common data values with zeros, ones, or randoms.

The next set of examples use `numpy.ndarray` four different ways to create `tensor` objects WITH data and numpy array. Then we discuss doing so without any data.
If you are using black box logs or other aggregated data, first make sure you get/sanitize/migrate/transdform the data to a numpy array. See the deep dive on pandas with Quandl to bring in data frames as a list, define the features and labels, and create the "array" (because python doesnt have arrays) with numpy out of the df- or whatever you named the data.
### With data
```py
data = np.array([1,2,3,4]) #using integers
#always check the type to be sure
type(data)
# >>> numpy.ndarray
```
Use the Class constructor `torch.Tensor(data)` will create floats, 
not infer the type like a factory method will.
```py
torch.Tensor(data) # calling Class constructor  
>>> tensor([1.,2.,3.,4.])
```
Calling the Class constructor behaves different than the next three factory methods.
### Using a factory function 
```py
o2 = torch.tensor(data) #to build tensor objects for us factory style, 
```
matching the `dtype` from the `np.array` we should see:
```py
# >>> tensor([1,2,3,4], dytpe=torch.int32) 
```
Type Inference looks for "the dot on the fly". Also 
```py
torch.as_tensor(data)
```
yields output
```py
# >>> tensor([1,2,3,4], dytpe=torch.int32) 
```
`from_numpy()` will also yield the same output `dtype` with as_tensor() when creating the tensor.
```py
o4 = torch.from_numpy(data)
```
Each of the last three have slight memory variances and implications. The above code can be transformed cleaner to look like this:
```py
o1 = torch.Tensor(data)
o2 = torch.tensor(data)
o3 = torch.as_tensor(data)
o4 = torch.from_numpy(data)
print(o1)
#>>> tensor([1., 2., 3.])
print(o2)
#>>> tensor([1, 2, 3], dtype=torch.int32)
print(o3)
#>>> tensor([1, 2, 3], dtype=torch.int32)
print(o4)
#>>> tensor([1, 2, 3], dtype=torch.int32)
```
If you are working in jupyter notebooks you may see aggregated differences with grouped print lines that may output sequentially different than ipython or REPL or IDLE or whatever IDE you may be in. ipython will use >>> to represent cli output where notebooks uses Out[number] and some boxed shading

### Now without data options
Use built in functions to do linear algebra. Also this is just a short list see the [docs](https://pytorch.org/docs/stable/index.html) for more pre-fabs.
```py
# torch.eye(2) returns a 2-D tensor (identity matrix) with ones on the diagonal and zeros elsewhere 
print(torch.eye(2))
# tensor([
#     [1., 0.],
#     [0., 1.]
# ])

# torch.zeros([3,3]) returns a zero'ed matrix
print(torch.zeros([3,3]))
# tensor([
#     [0., 0., 0.],
#     [0., 0., 0.],
#     [0., 0., 0.]
# ])

# torch.zeros([3,3]) returns a zero'ed matrix
print(torch.ones([3,3]))
# tensor([
#     [1., 1., 1.],
#     [1., 1., 1.],
#     [1., 1., 1.]
# ])

```
These may be called with other than "Sqaure" sizing for appropriate sized tranforms.
Last is a random filled matrix:
```py
print(torch.rand([2,2]))
# tensor([
#     [0.0465, 0.4557],
#     [0.6596, 0.0941]
# ])
```
## Deal with transformation of data into PyTorch
We need to make sure we are working with the appropriatly created "array" of data to send through the neural network. Look back at these three factory methods and how they are creating PyTorch Tensors.

PyTorch tensors are instances of the `torch.Tensors` class. The capitalization is everything in the code. The abstract concept of tensors and PyTorch tensors is that we can actually use the latter in code. Dealing with calling the class outright we may see some type mismatch of data, and using the various factory methods we can see how these differences relate to use in neural networks. Just remember the capital "T" is the class constructor.

In python the print() is the string representation of objects. The constructor uses the global default to create the dtypes when instantiating the object, but the factory functions will "infer" the data type. Check global default with `torch.get_default_dtype()`. Thats where we see the float32 in the first create above. Now the type inference way of a factory function is in the incoming data.
```py
import torch
import numpy as np
o1 = torch.tensor(np.array([1,2,3,4])) #lower t and no dots
print(o1)
# >>> tensor([1,2,3,4], dytpe=torch.int32)
o2 = torch.tensor(np.array([1., 2., 3., 4.])) # ah the dots
# >>> tensor([1,2,3,4], dytpe=torch.float64)
```
### Now the quick twitch of the finger, and magic happens!
```py
o2 = torch.tensor(np.array([1, 2, 3, 4]), dytpe=torch.float64) # ah NO dots and some default overrides
# we may have been taught to get into the habit of explicitly calling out data types for compatibility
# >>> tensor([1,2,3,4], dytpe=torch.float64)
```
So your tactic may be read the docs, notice the `=none` in the arguments, and this means we are getting closer to understanding documentation telling you there is no set default. Or, use print lines and see what instantiating tensor objects yield you- either way we need to understand the big "T" class does not infer, just uses the default. Try to understand the sometimes difficult to read docs, then printlines may mean more for you.

Also on the subject of dtype we last need to reiterate the "T" uses the default dtype, and we can use code to reinforce what  we know about how an instance object creation is happening. 
```py
o1.dtype == torch.get_default_dtype()
True
# or for the little "t" the dtype is inferred based on the incoming data
o2.dtype == torch.get_default_dtype() 
False #if and only if the default was float32 and the o2 used the little "t" with data([1,2,3]) as torch.int32
```
So if we are flying by writing scripts and clobbering data with a simple miss on capitalization, this ia a valid test and check to your developmenmt flow, both from a notebook or ipython view and even to have it as a unit test with an `assert` that your instantiations happen as wanted and expected.

## Memory sharing vs copying
Again from python 101, we need to remember how to manipulate data that gets sent to the interpreter. It may be such that it gets compiled right before being spit out to you in real time on the fly, or it may sit there for a while in a variable that never changes while you think you are overwriting the correct memory location. Thats why we looked at 4 ways to create the objects from `data`. The as_tensor() and from_numpy() methods will not perform the same way as we may have originally thought. The last two prints should show how numpy and pytorch work at getting along with the runtime environment.
```py
# you may have to re-initializa the data object 
data = np.array([1,2,3])
data # be careful here on BIG data, call .head or .tail for just a little bit of verification
# >>> array([1,2,3])
o1 = torch.Tensor(data)
o2 = torch.tensor(data)
o3 = torch.as_tensor(data)
o4 = torch.from_numpy(data)

# edit the np.array values that we used to create the tensors, but tensors are already assigned to objects o1,o2,o3,o4.
data[0]=0
data[1]=0
data[2]=0 

# then test out wehat you got...o1, and o2 print the "copied" data- class instantiated "T" data with dots (float default) and the dtype=torch.int32 for the o2 string representation of the 02 object. 
print(o1)
# >>> tensor([1., 2., 3.])
print(o2)
# >>> tensor([1, 2, 3], dtype=torch.int32)
# but now the as_ and from_ created tensors will actually "share" data from the original location, which means we should see the just set data to all zeros before it outputs to the screen.
print(o3)
# >>> tensor([0, 0, 0], dtype=torch.int32)
print(o4)
# >>> tensor([0, 0, 0], dtype=torch.int32)
```
### Sum it up
Share | Copy
|---|---|
torch.as_tensor()|torch.tensor()
torch.from_numpy|torch.Tensor()

# USE THE RIGHT CREATION METHOD!!!!
This is how PyTorch is interoperable with numpy- NumPy bridge for zero-memory copy is efficient. This means they have the same pointer aiming down the barrel. Moving between numpy arrays and pytorch tensors can be quick as the data is shared and NOT copied behind the scenes when creating new pytorch tensors.

## Which one is best when? 
That's up to you, but try to start with the lower case "t" factory method torch.tensor() for every day use, and when dealing with memory optimizations then look at the .as_tensor() for sharing with speed in mind. The from_numpy() only accepts numpy arrays, but they are both otherwise ok. Also Deeplizard shares a few more ideas for this.

* `numpy.ndarray()` allocates an object on the cpu, whereas `as_tensor()` copies the data from cpu to gpu for cuda. So thats why short computations may benchmarch slower with ab test or otherwise timed testing.
* The increased time benifit of `as_tensor()` is when we see heavy context switiching between `numpy.ndarray` objects and `tensor` objects. Single load operations we may not notice significant swing in performance.
* Don't bother with pyhton data structure `lists` and `as_tensor()` they do not play well in the sanbox (yet).
* as_tensor() requires the developer to know about sharing and the pointer issues. It helps maintain data integrity and not clobbering too much of it:) and how it impacts multiple objects.
