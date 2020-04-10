# Callable Recall, Review first
Network and layer forward methods get influenced here. The last notes was on matrix tranformations on Linear layers with mathmul() and how it transforms in_features to out_features. Input features are received in the input layer passed in the form of flattened one dimensional tensors and THEN are multipled by the weight matrix. The out is also flat. The weight was the 2d and the in_features is the 1d 4x1. The result is also a 1d output.
```py
in_features = torch.tensor([1,2,3,4], dtype=torch.float32 )
weight_matrix = torch.tensor([
    [1,2,3,4],
    [2,3,4,5],
    [3,4,5,6]
], dtype=torch.float32)

weight_matrix.matmul(in_features)
# tensor([30., 40., 50.]) comes about like so:
# (1*1) + (2*2) + (3*3) + (4*4) = 1+4+ 9+16=30
# (2*1) + (3*2) + (4*3) + (5*4) = 2+6+12+20=40
# (3*1) + (4*2) + (5*3) + (6*4) = 3+8+15+24=50
```
The weight_matrix defines a linear function that maps a 1 dimensional tensor with 4 elements to a 1 dimensional tensor with 3 elements, but we explained it like a 4d euclidean space to a 3d euclidean space the last time. This is also how linear layers map as well. They map in featues space to an out features space using a weight matrix

##Pytorch defined Linear Layer equivolent
Declare 4 in's AND do the tranform of them to 3, using a weight matrix. 
```py
fc = nn.Linear(in_features=4, out_features=3) # thiis executes the above, but wheres the execution?
```
Where is the weight matrix? Its obviously in the PyTorch Linear layer class. The line above creates the 3outx4in weight matrix, again by taking the height from the out and the width from the in. Inspect the PyTorch source code for `class Linear(Module)` and scroll to the `__init__()` class constructor. Can we inspect with inspect module and dis like they did with a click to step into that module? Its obviously an IDE config... I think. In any event, the init is creating another wrapped tensor with the Parameter Class, ultimately getting a weight tensor created.
```py
self.weight = Parameter(torch.Tensor(out_features, in_features)) # notice the class 'T' not the factory method here
```
And we can "call the layer" now by passing the in_features tensor. 
```py
fc(in_features)
#tensor([-3.2957,    0.1233, -0.1837], grad_fn=<AddBackward0>)
```
This calls the object instance like this in PyTorch because the neural network modules are callable python objects. We got the 1d tensor with 3 elements output but the __init__ is done in pytorch creating a weight matrix with random values. This means we are using different functions to produce these outputs we just typed. this shows that when we update the weights we are changing the function. The previous `fc(in_features)` direct call used the randomly filled weight matrix, but we can explicitly fill that fc by passing the creation of the wrapped Parameter the original matrix from before:
```py
fc.weight = nn.Parameter(weight_matrix) # create the 3x4 first, fill it like you wish, then assign it to the weight matrix directly in the fc variable
```
PyTorch Module weigths need to be 'p' paramters, or instances of the 'P' Parameter class inside the neural network module. This should transform the input if you wish to verify this with the new weight matrix. So we should see numbers closer to the 30, 40, 50 originally. We also need to be aware that the learned parameters from before also has a bias tensor in there, and this bias is created and is also used here.
```py
fc(in_features)
# tensor([30.2137, 40.3312, 50.1837], grad_fn=<AddBackward0>)
# tensor([30.2137, 40.3312, 50.1837], grad_fn=<SqueezeBackward3>)
```
We can trun this `bias` off sending a false flag into the constructor
```py
fc = nn.Linear(in_features=4, out_features=3, bias=False)
fc.weight = nn.Parameter(weight_matrix) #assign the original 3x4 we have been playing with
fc(in_features)
# tensor([30., 40., 50.])
```
Then if we come across the gen line equation y=Mx+b as linear equation y=Ax+b the A is the weight matrix tensor, x input tensor,b bias, and y is the output.
## Callable Python Objects
If the class implements the `__call__` method this gets interesting. The PyTorch Module classes implement it.  
`__call__` was called when the Layer instance object `fc` was used earlier as if it were a function call directly `fc(in_features)`. This will happen with PyTorch `forward()` at the same time. We DO NOT call the `forward()` directly but call the object instance directly, which then in turn will invoke the `__call__()` that contains the call to `forward()`. It holds true for all PyTorch neural network modules, mainly for networks and layers.

## Debug it
Write in a short couple line and inspect all the underscores. Set a breakpoint at the Linear layer call and assignment to fc.
```py
import torch
import torch.nn as nn
fc = nn.Linear(in_features=4, out_features=3) # place debug here, and step through 
t =torch.tensor([1,2,3,4], dtype=torch.float32)
output = fc(t) # this will invoke the __call__ eventually
print(output)
```
Stepping into not over the creation of the Linear layer fc we should see the __init__ step to the `self.weight` being created from `Parameter` class getting passed the 3 and the 4 for the in and out features from our hand written code, then also see it step out past the `self.reset_parameter()` back to our assignment line to `t` so thast we can create a tensor to send to our Linear layer callable pytorch object `fc(t)` and assign that to output.

This step brings us to the __call__ from within the pytorch `nn/module/module.py` class. Observe after the `for` loop the `else:` actually calls the `forward()`. 
```py
# within the pytorch linear.py
# dec
@weak_script_method 
def forward(self, input):
    return F.linear(input, self.weight, self.bias)
```
This is from the torch.Functional package, hence the 'F' call on this return. The three elements needed here are within the parenthesis. Step again into the functional.py that contains the implementation of the .linear() method. We can see the matmul() being used to generate the output in the else block that issues the creation of the matrix: `output = input.matmul(weight.t())` and since our `input.` is on the left hand side the weight tensor has to be transposed for this operation to move forward. When that returns we are back into the __call__ in module.py, it returns and then we can step to the print we did to check.
```py
print(output) #rank-1 tensor with 3 elements
```
The concept here to ALWAYS remember in pytorch is dont call the forward method directly- EVER- as you now see the extra stuff that happens in the background with the __call__(). If we want to invoke it, call the object instance directly and let it do its work. It applies to network and layers because they both are pytorch neural network modules.

Next is the implementation of the `forward()`