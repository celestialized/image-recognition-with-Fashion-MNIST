# Deep Learning
The big picture is worth vocalizing. People write the code for the neural network, but the data, according to Jack Ma from Alibaba is the water we as programmers have to have our machines learn to drink. Computers are code and data. In neural networks the software is sort of the network itself. Really the weights that emerge automatically from the training process. The actual software can be procured by itself from  the training process. The programmer simple overseees the process and guides it. So we train the machine to write the computations for us. So if we tweak the software directly as traditionally done, we really want to look at how we actually tweak the data. 

Andrej Karpathy (Tesla's AI director) lays it out like so:
* Define a set of programs that we are going to search over
  * he called out c++ and javascript...
  * train the SVM with Stochastic Gradient Descent with weights.
  * ConvNetJS
* Introduce a data set to mold soft constrainsts for desireable funtionality of the program
* Using optimization we compile dataset and write it into the code

## Fashion MNIST [data set](https://github.com/zalandoresearch/fashion-minst) from Zalando paper found at Cornell University Library 
Hand written data set used to train machines for image learning. Modified National Institute of Standards and Technology. 70K handwritten digits to train with, both beginners and techs benchmarking alike. 60k in training set and 10k in the testing images. Stepping into a more complex modeling scheme we are looking into images of clothing to train with, as these highschool written images just do not cut it today. Zalando created the dataset for the fashion industry.

## Migration of data- no output layer modification
This set mirrors the MNIST set while introducing a higher level of difficulty. The purpose here is to replace the original data set with a more complex one through transforming the data. The 10 classes still are within the fashion set. This allows a neural network to be swapped over if the original was using the MINST and now wish to use the more complex fashion set, without having to change their output layers. It contains the same image size, data format and the structure of training and testing splits. Their take on the swap is to simply modify the fetch url. With the torchvision library we will do just that.

## Zalandos process
* Fetch images from the site and convert them to .pngs
* "Closely" trim the color of the corner pixels within 5% distance of the maximum possible color intensity of the rgb space.
* Resize the longest edge to 28 by subsampling pixels- whereby we skip some row/cols in the process
* Sharpen pixels using Gaussian operator of the radius and the standard deviation of 1.0, with an increasing effect near the outlines.
* Extending the shortest edge to 28 and put the image to the center of the canvas.
* Negating the intensities of the image.
* Convert image to 8 bit gray scale.

### Deep learning in a nutshell
They defined an architecture. Described it as the neurons of artificial brain cells. Nice. Defined a loss function. This is the measure of how good your system is. Then optimized loss on the actual data.
Next we use this fashion set to accurately predict output classes given an input fashion image.

Anima Anandkumar director of research from NVIDIA split this process into three pillars: 
* Data
* The Algorithm.
* Computing infrastructure.
Acedemia spends a lot of time dealing with the algorithm. The Bounds. Sap complexity. The Big Oh face... I mean code complexity:) Then the big issue touched upon from sentdex and the google stock predictions notes is how do I get the "noiseless" label data? We train on one data set then serve on another. Hmm. How do we deal with this and then how do we scale this out? Her speech was about how we merge academic research with industry research. From sentdex we can also start small with deeplizard and stock prices [tutorial](https://deeplizard.com/learn/video/d11chG7Z-xk).

### Sap?:
* number of elements (or subsystems)
* interactions between them
* operation between the subsystems
* diversity/variability
* environment (and its demands)
* activities and their objectives