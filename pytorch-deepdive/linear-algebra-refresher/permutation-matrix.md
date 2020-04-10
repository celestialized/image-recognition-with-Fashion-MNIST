# Special Matrices
Some of the material here may not directly apply to math needed for reshaping but for me it will help unlock some neural pathways not traversed for about 5 years. The goal is to find a lesser set of words to accurately describe what is happening here as to keep certain audiences captive during the parts where application of these operations are critical. e.g. if we are talking to an 8th grader about algebra we probably shouldn't be using $10 words like convolve the entire square output tensor to a single data point through the appropriated linear sequence- it might better look like we need to know what this one value is to be able to move on to the next step. The latter just sounds more appealing to me.

These matrices operations are needed for particular types of transformations. As we dig through a few simple operations with them we will focus our attention on application detail for things like relu() activation proceedures being a non-linear problem on purpose, pooling and its role in neural networking, the flattening of tensors for the sake of transitioning convolution operations to a "one by x" stream of numpy element values fed to linear operations for forming output, and the proper direction of expansion and shrinkage as we prepare for both forward and backward propagation of assignable weights within the matrices involved at these various steps. A little ironing out with ambiguity of terminology might be in order from the higher ups here too.

## Permutation Matrix

Are matrices that have a value of 1 in only one row and column at the same time, and the rest contain zeros for values. Identity Matrices fall into this category of permutations and are symmetrical but may do nothing. Lets demonstrate a quick multiplication of a "regular" permutation matrix and see the output.

```py
import numpy as np
