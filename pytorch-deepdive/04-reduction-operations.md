# Argmax and reduction operations
Argmax is a mathematical function that tells us which argument, when supplied to a function as input, results in the function’s max output value. 
## What's a reduction, precious?
Reduction reduces the tensor to a smaller, often single axes at a time, down to sometimes a single element scalar value tensor operating on all the tensors elements.
So we start by proving in code that the sum operation is a reduction operation.
```py
t = torch.tensor([
    [0, 1, 0],
    [2, 0, 2],
    [0, 3, 0]
], dtype=tensor.float32)

# for a specific axes we just pass a value as a parameter to refer to that axes- but crawl before we walk
t.sum()
# tensor(8.)
t.numel() # number of elements
# 9
t.sum().numel()
# 1
# proof
t.sum().numel() < t.numel
# True because the first tensor was definately larger than the sum, 
# and which reduced the values to one element
# here's more that reduce the tensor to a single element scalar value tensor operating on all the tensors elements 
t.prod()
# tensor(0.)
t.mean()
# tensor(0.8889)
t.std()
# tensor(1.1667)

# now step it up a notch with a rank 2 3x4 tensor to show how reduction shrinks axes at a time
t = torch.tensor([
    [1,1,1,1],
    [2,2,2,2],
    [3,3,3,3]
], dtype=float32)
t.sum(dim=0)
# tensor([6., 6., 6., 6.])
t.sum(dim=1)
# tensor([4., 8., 12.])
```
## #mindblown 
Get with the `dim=1` output. Remember axes one is arrays, axes two is numbers. when we take a sum of the first axis (dim=0 above) we are taking the sum of the elements of the first axis (via element-wise operation) like this:
```py
# no action here just list them out
t[0]
# tensor([1., 1., 1., 1.])
t[1]
# tensor([2., 2., 2., 2.])
t[2]
# tensor([3., 3., 3., 3.])
# here is how we got all six's, due to element wise addition
t[0] + t[1] + t[2]
# tensor([6., 6., 6., 6.])
```
The second axis in this tensor (dim=1) contains numbers that come in groups of four. Since we have three groups of four numbers, we get three sums, I see it by 3 "rows"
```py
t[0].sum() #add all the ones in first row
# tensor(4.)
t[1].sum()
# tensor(8.)
t[2].sum()
# tensor(12.)
# and finally the #mindblown
t.sum(dim=1)
# tensor([ 4.,  8., 12.])
```
## Argmax
Reduction to one value for which argument when supplied to a function as input, results in the functions maximum output value. The actual call tells us the index location for the max value found inside a tensor. The tensor is reduced to another single tensor index value where the max value is inside the tensor.
```py
t = torch.tensor([
    [1,0,0,2],
    [0,3,3,0],
    [4,0,0,5]    
], dtype=torch.float32)
t.max() #5.
t.argmax() #sits at index 11 for the 11th element 0 based for a "flattened" tensor
t.flatten()
# tensor([1.,0.,0.,2.,0.,3.,3.,0.,4.,0.,0.,5.])
```
## Argmax dimension specifics and indices
Working with specific axis (dim=0) yields another set of results. First the result should be 2 tensors returned. The first will be the actual values and the second will be the indices found for each. When dim=0 we look at the first axis, or the first column for each call below to compare max.
```py
t.max(dim=0)
# tensor ([4,3,3,5]),tensor(2,1,1,2)
```
The second set 2 is the last row, 1 middle, 1 middle, 2 middle. Then to use (dim=1) we look at the elements within each array horizontally.
```py
t.max(dim=1)
# tensor([2,3,5]) at the positions tensor([3,1,3]) 
#keep in mind the middle row has two 3's and max does not swap out location for the second comparison x > y?
```
## Real life application
We use argamx on a neural networks outut prediction tensor, to tell us which categaory has the highest prediction value. This is usually because each index of the ouput tensor corresponds to a particular prediction category or prediction class.

## Accessing values in the tensor
```py
t = torch.tensor([
    [1,2,3],
    [4,5,6],
    [7,8,9]
]dtype=torch.float32)
t.mean() #5. is the indices not the value?
t.mean().item() #5.0 

t.mean(dim=0).tolist() # compare all the columns return list
# [4.0,5.0,6.0]
t.mean(dim=0).numpy
#array([4.,5.,6.], dtype=float32)
```
## [SciPy.org](https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html)
There is quite a bit more ways for advanced indexing and slicing. the index 0-9 describes the types of clothing. We have all seen these great scaled images that contain thousands of individual smaller pictures with transformed colors to complete the larger topic.