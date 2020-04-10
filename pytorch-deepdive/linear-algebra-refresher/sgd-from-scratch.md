# Stochastic gradient descent
Linear Regression is only ML. No neural network. Gradient descent happens in almost all neural nets. Linear Regression is just one way to simply show how gradient descent can be used to minimize a loss function.

This is an iterative method for optimizing an objective function with suitable smoothness properties. It is either differentiable or sub-differentiable, meaning we are looking to reduce the complexity to a manageable state. Stochastic refers to using random shuffled samples to evaluate gradients. We really can look at this as a stochastic approximation of gradient descent, based from 1951 Robbins-Monroe algorithm. this may be more than an 8th grade explanation, but my goal is to feed this to people in a watered down way they can relate to, so that will come after a good run through torchvision and the fashionmnist data-set.

## Basic Principals
Minimize function as a Sum.

<img src = "./minimize-as-a-sum.svg">

w is the parameter to be estimated. Q sub i is the iteration observation used in the training data-set. In stats we learned sum-minimization with least-squares and maximum-likelihood estimation (independent). For the ROV we will need to link images and observations, but this may be done in a separate iterative fashion when dealing with OpenCV and batch processing video, independent of simple image recognition and predictive analysis.

### Terms can get intermingled with neural networks and training for using them

Data with inputs that have labels become a "supervised" problem. The learning process is about using the training data to produce a predictor function. This is the function that takes inputs x and tries to map them to labels y. We want the predictor to work even on examples it hasn't yet seen before in the training data. That also means we want it to be as generalized as possible. Because we want general we need a priciple mathematical approach.

So for each datapoint x we compute a series of operations on it. Whatever operations our model requires to produce a predicted output we then compare that predicted output to the actual output to produce an error value. This is what we minimize during the learning process using an optimization strategy like gradient descent. The way we compute the value is by using a loss function. It will quantify how wrong we will be if we use the model to make a prediction on x when the correct output is actually y. We are trying to minimize this. 
#### Loss function: 
Depends on the presence of outliers, the choice of machine learning algorithm, the time efficiency of gradient descent, and the confidence of predictions. Quad Shannons information theory in 1948. The goal is to be reliable and efficient at transmitting a message from a sender to a receiver. Shannons jist was to transmit one bit to the recipient meant to reduce the recipients uncertainty by a factor of 2.

Soccer example: World cup 50/50 split odds for teams to win. The prediction function chooses one as the winner. It supplies us with one bit of useful information where out of two possible outcomes we now have one better piece of info. It has reduced our uncertainty by a factor of two. There were two equally likely outcomes, now there is just one. This "bit" of information can be a string, an image, a series of bytes of info, but it is still represented as a single bit of useful information. This is how the output from pushing a dataset through our network is viewed.

If there were 8 equal teams that could win and a predictor service returns the winner the reduction is of a factor of 8. 2^3=8 or log base 2 of 8 is 3, or 3 bits of useful information. We can compute the number of bits communicated with the log. This is done with the binary log of the uncertainty reduction factor- which is 8 here. 

But what if it is not an equal split? Say the original 2 team example was a 75/25 split, and the prediction service chose the lesser 25% to win. The equation to compute the number of bits here would be the negative binary log of the probability of 25. The entropy is measure of the average amount of information that we get from one sample drawn from a given probability distribution p where H(p)=-Esubi P sub i log base 2 (of p sub i). The larger the variation of data the larger the entropy.

The cross-entropy is the average message length, and can be expressed by both the true probability distribution p and the predicted distribution q. How long will we refresh this to frontal lobe eeprom? PDQ. If the  prediciton is perfect then the cross entropy is equal to the entropy. But if distributions differ, then the cross entropy will be greater than the entropy by some number of bits. This excess amount over is the relative entropy or also called the KL-divergence.

Big Picture Loss functions get categorized into two types: classification and gression loss. Example Music dataset of a bunch of mp3 files and labeled genres. We make a prediction using a trained model, it outputs a bunch of class probability values- one for each genre in question. For loss try use cross entropy (also called log loss) This is the measure output probability between 0 and 1. The cross entropy loss increases as the predicted probability diverges for the actual label. Predicting a value of .24 when the actual is 1 is bad and has a high loss value. Ideal model has a log loss value of zero.

Another classification loss is a hinge loss- usually in support vector machines. It penalizes predictions that are incorrect AND even when they are correct but not confident. It penalizes predictions that are REALLY off in a big way, not much that are off a little, and not at all for those that spot on. Hinge is quicker to compute than cross entropy. For speed use hinge. Faster to train with gradient descent a lot of the time the gradient is zero and you do not have to update the weights. If you have to make real time decisions with lesser accuracy depend on the hinge loss over cross entropy, but if accuracy over speed matters use cross entropy

Loss functions are sometimes termed cost function- which maps an event or values of one or more variables onto a real number intuitively representing some "cost" associated with the event. Loss functions are often used to estimate parameters. The event in question is some function of the difference between estimated and true values. In Convolutional neural networks we seek to minimize the loss function.

Mean Squared Error (MSE) is a loss function also called the L2 loss that measures the average amount the models vary on the predictions from the correct amount. This can be seen as the models performance of the training set. We calculate the difference between the predicted output and the actual output, square it, and do it for every data point, add them all up (sum) and divide by the total number of them. The square in there creates a quadratic, or convex representation. this plots the u shape with a minimum. So when we use a gradient descent, we wont get stuck in a local minimum- we find the global minimum to find the ideal input parameter values to optimize the objective function.

Mean Absolute error- popular regression function also known as the L1 loss to measure the average magnitude of the errors in the set of predictions, without considering their direction. This takes the average over the test sample of the absolute differences between our prediction and the actual output where all individual differences have equal weight. With MSE we penalize the large deviations more, squaring a big number becomes a lot larger problem relative to the others. So this means for the outliers MAE is better than MSE when having a lot of anomalies in your data set. MAE assigns equal weights to the data where MSE emphasis is on the extremes. The square of a really small number is even smaller (1/n vs 1/nsquared) and the square of a really big number is gianormous. 

Worth mentioning here is Huber loss which is like the MAE and is less sensitive to outliers than the MSE. It is quadratic for small values and linear for large values. 

### When to use a particular loss function?
#### Regression vs. Classification and Speed vs. Accuracy

M-estimators:
objective funtion is either a loss function or its negative.
local minimization is too restrictive for some problems of maximum-likelihood estimation

Stationary Points:

Likelihood function
Score function
other estimating equations

Sum minimization problem and emperical risk minimization means that for the above equation Q sub i (of w) is the loss function at the i-th example and Q(w) is the emperical risk.

Then to minimize we have a batch standard function or gradient descent method to do the following:

<img src = "./gradient-descent-method.svg">
where n is the machine learning rate or step size. The learning_rate is a hyperparameter ot "tuning knob". The problem is when we have huge data sets and no simple formula exists sum of gradients then becomes way more difficult. This is because evaluating the gradient requires eval of all the summand functions' gradient. This is why we sample subsets of the summand function. This works for the large scale ML problems.

Usiing Iterative methods
<img src = "./approx-at-single-example.svg">

## SGD Simple Linear Regression for minimizing from scratch with Python
If we do a basic plot of some x and y array using numpy and matplot lib with jupyter we can attempt to find the "best fitted" line that the plot will draw for these points using SGD.
The basic concept from a point slope concept is to find predicted values: solve for slope and y intercept and create this visually with the tools in python.

There is some line that explains the linear values for y. y is a function of x. Here, the linear function of x. We know y = mx + b and that m = slope and b = intercept.  

We need to define the cost function and the slope and intercept gradients. Notice the following is just the derivative of the MSE function- with respect to slope and intercept.
<img src="./b-and-m-gradient-formula.png">

Then we use the predicted values to choose the "best" line. This means we use linear regression to say/extroplate what the value for y is approximated at if x is 5.5. The aim of SGD is to get the m and the b value better than the y_predicted by means of trial and error. WE can look at the values of y and say there is a factor of 5 for the first few then 10 then take an average in your head and guestimate m at 7ish and the intercept at something too. WE really want to use all the possible combinations of x and y and get an exact with SGD.

### Using MSE 
We call the Mean Squared Error function a cost function. This is the error that we want to minimize. This example may out line some shortcomings. We have to optimize how we use epochs not go off the parabola. Notice below the lr function may be the permeations we aim at not having to write in modernized practices. This may be the traditional approach to "figuring" it out the hard way...is there an easier way?

<img src = "./mse-cost-fcn.svg">

The actual values versus the predicted values for y are squared, add them (sum) together and divide by the total number of data points n. The first y sub i is the fixed data points and second the y sub i with the ^ over it is not fixed and is the predicted line. The second one changes as the parameters change (the points in each np array. So we also can say the loss function MSE is a function of slope and a function of intercept. We square the subtracted values because we need all positives to sum. The value is not as important as the magnitude from an abstract sense of wanting to minimize the magnitude.

If we plot the MSE it is an upward opening parabola with the min at the desired best slope w. As w changes the cost function changes. How do we know at the min we have the desired cost function? You can graph a basic minimization function and see SGD in action. As we traverse down the open parabola we are getting close to the desired result. `m_new = m_new - (learning_rate * m_grad)` The initial value can also be seen as the initial weight. the learning_rate is how fast we wish to traverse the parabola to the desired location. The m_grad is the actual slope of the cost function with respect to m. From the right side of the parabola the m should be positive and from the left side negative due to the subtraction of the two.

The lr function attempts to use all the values to gather the prediction. We have to use reasonable values here for the learning_rate as to not make too big or too small jumps, and also if we set the epoch iterations too low we may never reach the bottom, and too high may be off the chart. We did this in Arnholt's class in R. here we go in Python.

```py
import numpy as np
import matplotlib.pyplot as plt
import os
#path=~./   
#os.chdir(path)
from Ipython,display import Image
%matplotlib inline # might have to remove this for real time showing of gradient descent
Image(filename='cost fn.png')# mse-cost-fcn.svg in our cwd to print formula to notebook
Image(filename='minimization,png')# to print the parabola
#linearregression #python #machinelearning
#write 2 lists (numpy array) 
x = np.array([1,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,6,7])
y = np.array([5,10,15,20,35,45,55,60,70,80,85,90,100,150,175,200,250,300])

y_predicted = 10*x-5 # just a guess
plt.scatter(x,y)
plt.plot (x,y_predicted)
# With SGD we start as a point on this parabola with a slope and a desire to move towards the optimal min value. We can say the initial value is the initial weight
#m_new = m_new - (learning_rate * m_grad) #learning is how fast we want the move. from point to point on the parabola.
# now work the equation to figure out the prediction
# m_new rand val of slope
# b_new was c_new but we know intercept to be b not c
# learning_rate depends on your parabola
# epoch is the number of times to run this iteration SGD times to reach min
# insufficient runs may not land you all the way down the parabola
# def y_new
def lr(x,y,m_new,b_new,learning_rate,epoch):
    N = float(len(y))# number of observations
    for i in range(epoch):
        y_new=m_new*x+b_new# y predicted we need for the cost
        cost=sum([t**2 for t in (y-y_new)])# the formula encoded here sum of squared errors
        b_grad=-(2/N)*sum(y-y_new)
        m_grad=-(2/N)*sum(x*(y-y_new))
        m_new = m_new - (learning_rate * m_grad)# slope
        b_new = b_new - (learning_rate * b_grad)# intercept remember is iterated by epoch and updated
        line = m_new*x+b_new #resulting line also keeps on changing
        plt.scatter(x,y,c='b')# may just need a 'b'?
        plt.plot(x,line)
        plt.pause(1)# thiis will let us see it update with dynamic visualization
    return b_new, m_new, cost

#then call it
lr(x=x,y=y,m_new=0,b_new=0,learning_rate=0.04,epoch=10)
```