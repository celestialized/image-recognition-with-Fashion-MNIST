# Element wise operations
Other same meaning terms can be element-wise,  point-wise, or component-wise
Indices based operations on the same corresponding elements within two tensors. They are corresponding if the two elements occupy the same position within the tensors. Position is given within a tensor given by the indices.
```py
import tensor
import numpy as np
#rank-2 tensors with a shape of 2 x 2, first axis elements are arrays, second axis numbers
t1 = torch.tensor([
    [1,2],
    [3,4]
], dtype=torch.float32)

t2 = torch.tensor([
    [9,8],
    [7,6]
], dtype=torch.float32)
# Example of the first axis
print(t1[0])
# tensor([1., 2.])
# Example of the second axis
print(t1[0][0])
# tensor(1.)
# elements corresponding are t1[0][0] and t2[0][0] which above are 1. and 9. respectively
```
The element must correspond to do these operations. Now we see that they must have equal number of scalar components (elements), more specific same shape for element wise operations, to work on each other. So the same number of axes and each axes are the same length. BUT, you may do so with different size or shaped tensors with not so easily contrived results. Read on.

```py
# addition, sub, div, and mult all do element-wise operation
t1 + t2
# tensor([[10., 10.],
#         [10.,10.]
# ])
t1 + 2
# tensor([[3., 4.],
#         [5., 6.]])
t1 * 2 # same as t1.mul(2)
# tensor([[2., 4.],
#         [6., 8.]
# ])
t1 - 1 # same as t1.sub(1)
t1 / 2 # same as t1.div(2)
```
### Scalear values and broadcasting with `np.broadcast_to()`
Scalar values are rank 0 tensors. They have no shape. We need to understand tensor broadcasting. This is the way tensors of different shapes are treated during element wise operations. So for the simple t1 + 2 the scalar value (in this case '2') tensor gets broadcasted to the shape of the tensor t1. Using the `np.broadcast_to(2, t1.shape)` we can see what this looks like in code, and how the same shape rule is still enforced. This is happening under the hood:
```py
# under the hood
np.broadcast_to(2, t1.shape)
# array([[2,2],
#         [2,2]])
t1 + 2
# tensor([[3., 4.],
#         [5., 6.]])
# is really this
t1 + torch.tensor(
    np.broadcast_to(2, t1.shape)
    ,dtype=torch.float32
)
# tensor([[3., 4.],
#         [5., 6.]])
```
## Broadcasting gets more challenging
This is valid due to broadcasting the lower rank t2 tensor object up to the t1 shape:
```py
t1 = torch.tensor([
    [1,1],
    [1,1]
], dtype=torch.float32)

t2 = torch.tensor([2,4], dtype=torch.float32)

t1.shape
#torch.Size([2, 2])
t2.shape
#torch.Size([2])
t1 + t2 
# the smaller t2 has to be broadcast to t1 size and 
# then element wise operation happens as usual
# check the broadcast transformation using the broadcast_to()
np.broadcast_to(t2.numpy(), t1.shape)
# array([[2., 4.],
#         [2., 4.]], dtype=float32)
t1 + t2
# tensor([[3., 5.],
#         [3., 5.]])
```
## Sneak in some JS for a minute
The technique of broadcasting comes into play with normalizing routines and preprocessing data. The deeplizard series on TensorFlow.js goes into the bgg16 preprocessing code. We also have another quick notes file just for scratch paper comparison rules and setup results.

We compare shapes to see if operations can be done, looking/comparing the last axes and working back to see if they are compatible. Broadcasting goal is to get them to the same shape to operate on them together. If there is no way to reshape them to be compatible you may not be able to do as you wish.

So for a (1,3) and a (3,1) the rule is if (this example the last two dimension values) they are equal (they are not), or if one value is a one- it is- then we look further (the first two also pass the same rule). Then to figure out the resulting size of a sum operation we calc the max of the last dimension, then the first. Result for the example is a 3,3. The broadcast of these example tensors can be "thought" of as copying to expand the tensors from 1 to three by for both axes to result in an addition that is two new tensors both with shape (3,3). They also offered a no go operation for checking if a rank 3 tensor shape (1,2,3)  and another tensor rank 2 3x3 (3,3) as the last two match but the middle and first do not pass match or one rule. If one were to use a rank 0 or scalar for broadcasting to compare with either of the two above (1,3) or (3,1) you use ( , )
and substitute a 1 in the blank values to evaluate the rules- it default should always pass, but allows the ability to see the resulting shape of the new tensor that becomes of the operation.

Also the node example shows a massive reduction of code when assigning the tensors as they abstract the tensor object instantiation from an arbitrary javascript object. So things like calculating the mean image net and the processing of the indices and centered rgb can be one line operations for your resulting tensor creations.

## Become a broadcast guru!
Quick side note- Jeremy Howard, lead dev at fast.ai, talks on scientists that do not deeply understand broadcasting they often write loops to do things like subtracting the mean over the channels for preprocessing with computer vision. If you know broadcasting then this is a trivial step that eliminates TIME out of the equation. Then he talked about programming in J for one liner mathematical computations. Jd is a columnar RDBMS. 

## Back to PyTorch and comparisons
Comparison operations return each element with a zero or a one. zero = flas eh one = true dude. Either they are or they are not- the droids you are looking for... ok. in action:
```py
t = torch.tensor([
    [0 ,5, 7],
    [6, 0, 7],
    [0, 8, 0]
], dtype=torch.float32)
t.eq(0)
# tensor([[1, 0, 1],
#         [0, 1, 0],
#         [1, 0, 1]], dtype=torch.uint8)
t.ge(0)
# tensor([[1, 1, 1],
#         [1, 1, 1],
#         [1, 1, 1]], dtype=torch.uint8)
t.gt(0)
# tensor([[0, 1, 0],
#         [1, 0, 1],
#         [0, 1, 0]], dtype=torch.uint8)
t.lt(0)
# tensor([[0, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0]], dtype=torch.uint8)
t.le(7)
# tensor([[1, 1, 1],
#         [1, 1, 1],
#         [1, 0, 1]], dtype=torch.uint8)

# remember relate the broadcasting portion for each operation.
# verify the t.le(7) as a broadcast_to()
t <= torch.tensor(
    np.broadcast_to(7, t.shape)
    ,dtype=torch.float32
)

# tensor([[1, 1, 1],
#         [1, 1, 1],
#         [1, 0, 1]], dtype=torch.uint8)
#and equivalently this:

t <= torch.tensor([
    [7,7,7],
    [7,7,7],
    [7,7,7]
], dtype=torch.float32)

# tensor([[1, 1, 1],
#         [1, 1, 1],
#         [1, 0, 1]], dtype=torch.uint8)
```
## Functions for element-wise operatons
It is ok to assume the function gets applied to each element of the tensor.
* .abs()
* .sqrt()
* .neg() negative not negation