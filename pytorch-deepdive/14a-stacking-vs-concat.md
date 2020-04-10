# Concat vs Stack
Concat joins a sequence of tensors on an existing axis. Stacking tensors joins a sequence on a new axis. Concat on existing axis is straight forward, but when we want to do so on a new axis (they call that stacking) it becomes more challenging. Stack is to create a new axis inside all of our tensors, then we concat along this new axis.

## Create a new axis for any given Tensor
Create new axis first then stack and concat. The add an axis in PyTorch is next.

```py 
import torch
t1 = torch.tensor([1,1,1])# just a shape of three rank 1 tensor
t1.shape # torch.Size [3]   
t1.unsqueeze(dim=0)# add dimension at index 0 of this tensor, 
# torch.tensor([[1,1,1]]) returning a shape of 1x3 rank 2 tensor
t1.unsqueeze(dim=0).shape
# torch.Size([1,3]) # rank 2
t1.unsqueeze(dim=1)# dim 1 will create the 3x1
# torch.tensor([[1],
#               [1],
#               [1]])
t1.unsqueeze(dim=1)# on the last or second axis?
torch.Size([3,1])# still rank 2
```
## Just a reshape of the Tensor, nothing more
The calls to unsqueeze affects the arrangement of data, but not the actual data itself.  
## Concat
For an existing axis it is also extending the length of an existing axis.
## Stack
Will create a new axis across all tensors in the sequence, then concat along the new axis.

```py
# notice these three all have single axis declarations
t1 = torch.tensor([1,1,1]) # single axis tensor each with axis length 3.
t2 = torch.tensor([2,2,2])
t3 = torch.tensor([3,3,3])
# result of cat will also be along the single axis. cats don't always behave this way, but will be along an EXISTING one. The only existing one here IS the single and only first axis.    
torch.cat(
    (t1,t2,t3)
    ,dim=0
)
# dim=0
# tensor ([1,1,1,2,2,2,3,3,3]) single axis tensor with axis length [9]
torch.stack(
    (t1,t2,t3)
    ,dim=0 # insert another axis at the first index, it is the zero'th one, it happens in the background
)
# dim = 0, but this is along the first axis. Also the first axis had length of three and the new shape is now 3x3. The three tensors are concatinated along the first axis. 
tensor([[1,1,1],
        [2,2,2],
        [3,3,3]])
```
To see that the stack will actually happen this way we defined we can also write the code out in the described fashion for a stack functionality using cat and unsqueeze to add in the needed reshaping.

```py
# cat with the added in axis creation is the exact same thing as a stack.
torch.cat(
    (
        t1.unsqueeze(0)#the new axis insertion is handled within the stack function call
        ,t2.unsqueeze(0)
        ,t3.unsqueeze(0)
    )
    ,dim=0
)
# same output just exposed the behind the scenes calls
# tensor([[1,1,1],
#         [2,2,2],
#         [3,3,3]])

To concat on a second axis we need a tensor WITH a second axis, there was none above. BUT, we can stack. A stack on dim=1 lines the data up vertically
torch.stack(
    (t1,t2,t3)
    ,dim=1
)
# tensor([[1,2,3],
#         [1,2,3],
#         [1,2,3]])
# or the following will be identical "vertical" lining of data
torch.cat(
    (
        t1.unsqueeze(1) # dim=1
        ,t2.unsqueeze(1)
        ,t3.unsqueeze(1)
    )
    ,dim=1
)
# tensor([[1,2,3],
#         [1,2,3],
#         [1,2,3]])
```
## When to use this functionality

For image processing we are talking about batches. This also means we are talking about combining a group of images into a batch. One image may have a channel, a width, and a height- no batch. When we need to have them into a group we have to create the batch axis [b],[c],[h],[w] from a [c],[h],[w]. This is where the stack looks cleaner in code and achieves the multiple steps we need- add the batch axis and preserve the images.

Also in a similar fashion we may have individual images with an axis containing a batch size 1 for each individual image. [1],[1],[28],[28] then another [1],[1],[28],[28] and a third [1],[1],[28],[28]. And now we combine these three into one grayscale or single channel tensor [3],[1],[28],[28]. This time there is an existing batch to concat on, and no need to stack. The prior example had the need to add in a new axis and then concat.

## Common usage
And last is the common occurance. We have a batch of images and we are adding in the three new images. This means that we may already have one tensor with a batch (like above [3],[1],[28],[28]) and we wish to add in a single image [1],[28],[28] and another [1],[28],[28] and one more [1],[28],[28]. This means we have to stack then concat the three new ones- and normally we might see this done in a loop.

For TensorFlow we see that the numpy arrays gets created for us automatically with the constant 
```py
# tensorflow first, syntax is a little different than pytorch
# axis = is the dim =  

import tensorflow as tf
t1 = tf.constant([1,1,1])
t2 = tf.constant([2,2,2])
t3 = tf.constant([3,3,3])
tf.concat(
    (t1,t2,t3)# each tensor has single axis, result is single axis with longer length (nine)
    ,axis=0 #dim= in pytorch
)
# notice the output also shows the numpy array usage for tf.Tensor output too 
tf.stack(
    (t1,t2,t3)# each tensor has single axis, result is the 3x3 stack
    ,axis=0 #dim= in pytorch
)
```
And then in the same manner as before we wish to see the two step stack and concat to see the same results, but this time in tensorflow. tf.expand_dims of each tensor is the unsqueeze in pytorch

```py
tf.concat(
    (
        tf.expand_dims(t1,0)
        ,tf.expand_dims(t2,0)
        ,tf.expand_dims(t3,0)
    (
    ,axis=0
)

```
Likewise we are going to see the stack at index 1 (dim=1 for pytorch) in tensorflow axis=1 should be the same data rearrangement with respect to "verticals". Keep in mind what is happening here if we are looking to have batches of images and what happens if we are doing reshaping functions.
[1,2,3,4] 
[1,2,3,4] 
[1,2,3,4] 
[1,2,3,4]
means we just snatched out a piece of each one if we had
[1,1,1,1]
[2,2,2,2]
[3,3,3,3]
[4,4,4,4]
```py

tf.concat(
    (
        tf.expand_dims(t1,1)
        ,tf.expand_dims(t2,1)
        ,tf.expand_dims(t3,1)
    (
    ,axis=1
)

```
Now the last module with similar functionality is numpy. Each flavor module has renamed the concat function- numpy is concatinate

```py 
import numpy as np
t1 = np.array([1,1,1])
t2 = np.array([2,2,2])
t3 = np.array([3,3,3])
#np uses axis= like tensorflow instead of dim = pytorch
np.concatinate(
    (t1,t2,t3)
    ,axis=0
)
np.stack(
    (t1,t2,t3)
    ,axis=0
)
# get the stack functionality with the concatinate function there is expand_dims again like squeeze
np.concatinate(
    (
        np.expand_dims(t1,0) # again to add an axis
        ,np.expand_dims(t2,0)
        ,np.expand_dims(t3,0)
    )
    ,axis=0
)
```
```py
t1 = torch.tensor([[3],[1],[28],[28]])
t2 = torch.tensor([[1],[28],[28]])
t3 = torch.tensor([[1],[28],[28]])
t4 = torch.tensor([[1],[28],[28]])

torch.stack(

)
```
