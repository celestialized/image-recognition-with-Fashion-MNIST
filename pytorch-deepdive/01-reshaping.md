# Recap Tensors in PyTorch
These are the code versions of abstract topic in deep learning and neural networks for sending transformations through the pipeline. Make sure you know how memory sharing impacts data integrity and what the cost is for copying over cpu objects to gpu processing with cuda. Cuda is built into PyTorch so we may need to figure out how that may look if we were first trained to build all of these data structures in C and then ported these abstractions over to a hard coded means of segregating your available gpu's for some serious numbers crunching and transformations. Read 00-tensors.md first please.

## 4 General operations we do with Tensors
Reshaping operations
Element-wise operations
Reduction operations
Access operations

## Tie it together
We take Input Data and send it through a series of transformations by taking a function that maps those inputs to correct outputs. Neural Networks are designed to shape data as we see fit for it. Wonder how that is used when talking statistics in a bored meeting...Deeplizard's take is that data is an abstract concept- wow- and that we leverage this with pytorch and concrete tensor data structures in code. The tensor is the foundation to produce a product- called "intelligence". With great power comes great responsibility... With neural networks it is often a series of shaping and reshaping data to suit each use case. We operate on the tensors for this to flow through the network as desired output.

### Peter Diamandis from Tedx LA 2016
This is the transformation of the human race...

## Reshape
Most important from a size perspective to use the entire data set efficiently.
```py
t = torch.tensor([
    [1,1,1,1],
    [2,2,2,2],
    [3,3,3,3]
], dtype=torch.float32)
```
This is a 3x4 rank 2 tensor. 2 vars to access all dimensional elements. Get the shape info in either of the next two ways programatically:
```py
t.size()
t.shape
```
both should return `torch.Size([3, 4])`, and for the access of dimensional elements:
```py
len(t.shape)
```
should return `2`. tensors store all axes, rank, and indices info.

Product of the component values in the shape gives us the number of elements contained within the tensor:
```py
t = torch.tensor([
    [1,1,1,1],
    [2,2,2,2],
    [3,3,3,3]
], dtype=torch.float32)
#convert the shape to a tensor, then take the product of it- this is on the fly even though t exists
torch.tensor(t.shape).prod()
# >>> tensor(12) is the number of components and this can also be referred to as the 12 "scalar" components for the 3x4 defined above.
t.numel() # is the exact same thing
# >>> 12
```
We care about the number of elements for shape, and any reshaping must account for all 12 of the elements. It wont affect the actual data at each navigable indices, just the shape of it. Now we can do reshape 
#### Without changing the rank.
```py
# using 2 factors axb the component values are factors of 12, and so the product is 12
t.reshape([3,4])
t.reshape([4,3])
t.reshape([6,2])
t.reshape([2,6])
t.reshape([1,12])
t.reshape([12,1])
```
### Change the rank
This uses 3 factors
```py
t.reshape([2,2,3])
```
## Squeeze
Start by reshaping and verifying its shape
```py
print(t.reshape([1,12])) # this does the ordering
print(t.reshape([1,12]).shape) # then verify torch.Size([1,12])

#these may change the rank of our tensor
print(t.reshape([1,12])).squeeze()) # shrink reshape with a squeeze removes all axes with length = 1
print(t.reshape([1,12])).squeeze().shape) # torch.Size([12])

print(t.reshape([1,12])).squeeze().unsqueeze(dim=0)) # Expand reshape with a squeeze, unsqueeze adds 1.
print(t.reshape([1,12])).squeeze().unsqueeze(dim=0).shape) # put back torch.Size([1,12])
```
## Flatten
Build a function to reshape and squeeze called flatten. This just creates a lower rank tensor than you started with. PyTorch can use a -1 to find or figure out what the output len should be.
`ndarray` with `x.shape` outputting (50000, 784) is a rank 2 tensor. When we flatten it we create a 1-d array with all the scalar components of the tensor. that would make 50000x784 "items" in a 1-d array.
```py
#you may pass any input tensor shape named t
def flatten(t):
    t = t.reshape(1,-1) #it will figure it out for you based on the other value and the input elements
    # the above example has len 12 so the 12 has to be the second axis to fit all the elements into the new tensor
    t = t.squeezee()
    return t

#then call it and notice the [[]]  
flatten(t) 
```
Another option is to just use `t.reshape([1,12])` somehow. I'm thinking first find the greatest common factor, scratch that if the t shape was a 3x4 we need a product and then a len. and then a call to reshape with that as the second argument. You may have to deal with stripping the outer set of brackets if that is not handled internally- eg think remove a list of a list too...

The idea for the option here is to flatten a whole tensor, but we may not need that for batch image processing, as seen later with gray scale example.

### When do we use a flatten?
This MUST happen when transitioning a neural network from a convolutional layer to a fully-connected layer. We take the output of a convelutional layer which is given in the form of output channels, then we flatten these out into a single 1-d array.
## Squeeze

## Test understanding of shape with concatination operations
The way we concat affects the shape- like 2 [2,2] can be stacked top on bottom or stuck together left to right- this is either row-wise axis-0 or col-wise axis-1 operation
```py
t1 = torch.tensor([
    [1,2],
    [3,4]
])
> t2 = torch.tensor([
    [5,6],
    [7,8]
])
# row-wise (axis-0)
torch.cat((t1, t2), dim=0)
# column-wise (axis-1) but keep in mind this may already be in your interpreter 
# from the previous run, and may need to be redefined to replicate these results
torch.cat((t1, t2), dim=1)
# tensor([[1, 2, 5, 6],
#         [3, 4, 7, 8]])
```
concat increases the num ele's for the resulting tensor. This causes the component values within the shape (lengths of the axes) to adjust to account for the additional elements. Notice the `dim=` below.

```py
torch.cat((t1, t2), dim=0).shape
# torch.Size([4, 2])

torch.cat((t1, t2), dim=1).shape
# torch.Size([2, 4])
```
Reshape is to affect the shape of the resulting tensor. 

### Maurice Conti - futurist
In 20 years the way we do our work will change more than what has happened in the past 2000 years. Design will be such that we feed our idea to the machine and its learned to be so smart about it it says, ummmm, try that again...

[Next:]() CNN flatten operation for greyscale operations on a figure 8 image.



