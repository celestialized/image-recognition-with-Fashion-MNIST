# Selective flattening on specific axeeees of Tensors
To allow tensors to flow through network as `batches`. This practice is common becaue convoltuion layer outputs are passed to fully connected layers, and MUST be flattened out before the fully connected layer will accept them as input. The flatten function, specific to reshape and squeeze methods combined to squash all the axes together. 

## How to flatten
A tensor needs 2 axes to flatten, which means we don't start with something that is already flat. 
Example is the classic 8 that has been "cropped" down to 18x18 from 28x28 from the MNIST data set.
We shoot for 18^2 here = 324 is the flat length, or single axis with length 324 (or 28x28 for the more detailed version). 

Each node in a fully connected layer receives this flattened output as input. This entire figure 8 example 18x18 resolution was flattened to single axis, but we may opt to only take a certain portion and flatten that particular axis instead. Often this is the case dealing with CNN's as we are not dealing with a single tensor but we may be dealing with a tensor that is a batch dealing with multiple images.

todo
For the ROV we now can appreciate a better understanding of a greyscale, and realizing the upper limits of the embeeded devices processing power, may allow for a maximized amount of batch processing of aggregated image samples. This also maxamizes the accuracy, maximizes the algorithm to autopilot with, and a whole slew of other issues around the use of data streaming from the camera for both the UI and the AI side of things.

## Code the CNN input tensor shape
Start small. KISSes and hugs here. Again, CNN input typical is 4 axes. \[B,C,H,W] ?= \[1,1,28,28] \| \[1,1,18,18]. To start we can make a simpler three representations of three images that are 4x4 rank 2 tensors to create a `batch` that can be passed to a CNN. We will populate these images with fake pixels of values that correspond to each of the single image tensors. These first three images are considered to be separate tensors in their own right, which will then be combined into one tensor using `stack()` as the "batch tensor".

```py
# create 3 4x4 rank 2 tensors 
t1 = torch.tensor([
    [1,1,1,1],
    [1,1,1,1],
    [1,1,1,1],
    [1,1,1,1]
])

t2 = torch.tensor([
    [2,2,2,2],
    [2,2,2,2],
    [2,2,2,2],
    [2,2,2,2]
])

t3 = torch.tensor([
    [3,3,3,3],
    [3,3,3,3],
    [3,3,3,3],
    [3,3,3,3]
])
# now combine these using stack()- to become a 3 axes instead of a 2
t = torch.stack((t1, t2, t3))
t.shape
# torch.Size([3, 4, 4]) and here B=3 for the batch
```
We should know that length should be 3 as we have now a single tensor comprised of 3 individual 4x4's, hence the shape output of [3,4,4]
```py
#this is now a rank 3 tensor that contains a batch of 3 4x4 images 
t
tensor([[[1, 1, 1, 1],
         [1, 1, 1, 1],
         [1, 1, 1, 1],
         [1, 1, 1, 1]],

        [[2, 2, 2, 2],
         [2, 2, 2, 2],
         [2, 2, 2, 2],
         [2, 2, 2, 2]],

        [[3, 3, 3, 3],
         [3, 3, 3, 3],
         [3, 3, 3, 3],
         [3, 3, 3, 3]]])
```
## Turn the batch tensor into a form the CNN expects
The next step is to add in the 4th dimension so the expected input from the CNN can be attained. We need to add in the axis for the color channels. An implicit single color channel for each of these image tensors would be a grayscale image. So to do this we reshape, again.
```py
t = t.reshape(3,1,4,4) # the grayscale is the "1"
t
# tensor(
# [#Batch
#     [#Channel
#         [#Height
#             [1, 1, 1, 1],
#             [1, 1, 1, 1],
#             [1, 1, 1, 1],
#             [1, 1, 1, 1]
#         ]
#     ],
#     [
#         [   #scalar components are the pixels
#             [2, 2, 2, 2],
#             [2, 2, 2, 2],
#             [2, 2, 2, 2],
#             [2, 2, 2, 2]
#         ]
#     ],
#     [
#         [
#             [3, 3, 3, 3],
#             [3, 3, 3, 3],
#             [3, 3, 3, 3],
#             [3, 3, 3, 3]
#         ]
#     ]
# ])
```
The additional axis of length 1 should not change the number of elements in the tensor. This is because the product of the components values doesn't change when we multiply by one. The B=3 for 3 images. for each image we have a single color channel. Each of these channels contain 4 arrays of which each of the 4 arrays contain 4 numbers or scalar components. Then to index into each of the individual tensors:
```py
# the first image
t[0]
# tensor([[[1, 1, 1, 1],
#          [1, 1, 1, 1],
#          [1, 1, 1, 1],
#          [1, 1, 1, 1]]])
# We have the first color channel in the first image.
t[0][0]
# tensor([[1, 1, 1, 1],
#         [1, 1, 1, 1],
#         [1, 1, 1, 1],
#         [1, 1, 1, 1]])
# We have the first first row of pixels in the first color channel of the first image.
t[0][0][0]
# tensor([1, 1, 1, 1])
# We have the first pixel value in the first row of the first color channel of the first image.
t[0][0][0][0]
# tensor(1)
```
## Flatten ALL axes as a single tensor batch is WRONG, but correct to show what we are aiming for...
The procedure is to get this into a stream of bytes (or whatever transport mechanism) to make it to the next stage of the network flow. In this case it is passing the batch to the CNN, so we do not want to flatten ALL of it, we are only concerned with flattening the image tensors within the batch tensor.

The next example is BOGUS. We NEVER DO THIS. ITS NOT RIGHT. We really want to have individual predictions for each image within our batch tensor. So really we wish to have flattened output for each image while retaining/maintaining the `batch axis`. We really want to flatten the color channel axis with the height and width axes ([C,H,W]) only.

For educational purposes lets see what happens when we do bite it big and accidentally flat it all. This will also answer the optional flatten from previously thought about flattening options.
This was part of the deeplizard challenge.
```py
t.reshape(1,-1)[0] # Thank you Mick!
# tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
#     2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])

t.reshape(-1) # Thank you Aamir!
# tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
#     2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])

t.view(t.numel()) # Thank you Ulm!
# tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
#     2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])

t.flatten() # Thank you PyTorch!
# tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
#     2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])
```
The last one is what comes with pytorch. The big picture is the way in which we are lining up the image pixels. From above we need to maintain the batch axis for each image, so we flatten only PART of the tensor. We wish to flatten the color channel axis with the height and width axes- the \[C,H,W]. And again this is built into pytorch with `flatten()`
## Flatten the SPECIFIC axes of a tensor
The `start_dim` is the axis to begin with. The one is the index for the color channel, which is the second axis for \[B,C,H,W] "skipping" the batch axis to leave in tact.
```py
t.flatten(start_dim=1).shape # tells us what the outcome should look like
# torch.Size([3, 16]) 3 images with 16 pixels 
t.flatten(start_dim=1)
# rank-2 tensor with three single color channel images that have been flattened out into 16 pixels:
# tensor(
# [
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
#     [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
# ]
# )
```
There is also a way to flatten just using the reshape fcn directly to reshape the batch tensor
## Flatten a roy g biv (RGB, where C=3 in [B,3,H,W])
We started the notes with recognizing that the color channel gets manipulated after initial input. So this means now we are addressing what that will look like in code. WE simply flatten each color channel out first. Then, the flattened channels will be lined up side by side on a single axis of the tensor.
```py
# Make a RGB image tensor with a height of two and a width of two with pytorch
r = torch.ones(1,2,2)
g = torch.ones(1,2,2) + 1
b = torch.ones(1,2,2) + 2

img = torch.cat(
    (r,g,b)
    ,dim=0
)
# verify it- three color channels with the height and width of 2
img.shape
# torch.Size([3, 2, 2])
#verify that tensors data too
img
# tensor([
#     [
#         [1., 1.]
#         ,[1., 1.]
#     ]
#     ,[
#         [2., 2.]
#         , [2., 2.]
#     ],
#     [
#         [3., 3.]
#         ,[3., 3.]
#     ]
# ])
# flatten that whole thing first, this time start actually has no B for batch to skip, so use 0.
img.flatten(start_dim=0)
# tensor([1., 1., 1., 1., 2., 2., 2., 2., 3., 3., 3., 3.])
# if you only flatten the channels:
img.flatten(start_dim=1)
# tensor([
#     [1., 1., 1., 1.],
#     [2., 2., 2., 2.],
#     [3., 3., 3., 3.]
# ])
```
### Joseph Redmon 2017 Vancouver BC from TED
He does object detection. Released his software to the public for use. Power of open source.
