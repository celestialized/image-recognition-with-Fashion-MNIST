# The beginning of AI on an autonomous or remotely operated vehicle

The first step in conversion of a manually driven rover underwater to an autonomous one finding "lures" off the bottom of a lake- or mapping nearly extinct corals and chewing up plastic bottles along the way- is to have a means to send it commands to execute and a training algorithm to feed data-sets into. 

## How to read this repo 

### Pytorch Deepdive

This extensive deep-dive was spurred from Deeplizard.com and was part of the discovery phase for a modern embedded project I embarked on to grow a deep appreciation for modern tolling behind application development. The deep-dive notes are pre-pended with a numbering system from one of my rover's playbook rules. It is merely a suggested path to follow when running through lengthy documentation on very technical subject matter.

Pytorch, torchvision, and the use of their neural network module with tensors to aide in training data-sets for image detection was another part of the puzzle. TensorFlow is a well developed means for coding this into projects, and the ROV will create its own custom train_set as a drop in replacement for the FashionMNIST data-set provided with torchvision. There are other options within the research as well, from Apple TuriML and Googles Brain and Borg. Deep learning is not so trivial as the CEO of fast.io might wish it to be, but we need to keep it in the public domain, available for the masses to consume if they so choose to take the time to interface with it.

### Findings along the way

The addition of Ansible, Confluent, Kafka, and/or Openshift into this project might aide in processing orchestration on more than one level. This can be thought of the traditional approach its inception was designed to assist- system orchestration from a centralized location. However, as the "*rabbitMQ* hole" was better realized, it also became clear that this mechanism may find itself embedded as a diagnostics tooling down in the weeds of Docker containers or K8's and pods, or even further implanted as OS kernel supplementals for scheduling implementations when assigning grouped tasks, GPID's and processes bound for specific memory and grouped at mathematical complexity per GPU. It may, however be just as easily discarded for the next best idea to not reinvent the wheel, or just be a means to get a systems position filled as ones peer network so requires.

## What does AI look like on my vehicle?

This is a two step process. Have one portion generate data-sets on the fly. This is the resulting "smarter" code as time goes on. Then the other process is designed as the checks and balances, as a set of tests that may taken over the resulting generated data to verify it is working within the target constraints dictated by the human. 

To start it is simple. Use TensorFlow and Python. Find an image. Tell someone about it. Then find and process batches. Questions to ask are how accurate do we need to be to escape false positives barrier with one, then a group. Can we adjust our algorithm for less and get more? What is more and what does that look like? Processing power at lower resolution? More frames per second at that nominal resolution resulting in maintaining the acceptable positive positive target. 

For an underwater rover we ask more questions- What about lighting and water condition? How can gray scale and color be best leveraged? First input on a nn when using tensor rank-4 [B][C][H][W] the C represents the 1 or 3 for RGB, then the transformation is relative to the next convolution operation only and the C is not the original input parameter any longer. What? 

How do we generate these new sets of linear equations to result in improved accuracy over time? Flattening tensors and feeding them through multiple functions build your reduction operation back to meet your targets format. What does that look like with the loss function? What are the aggregates? Weights? Acceptable reduction operations and output parameters? 

Lure recognition can be trained like the Fashion-MNIST set has. "10 manually Labeled Classes" close to 11 worm feature attributes. Directly dropped in. Shapes can be dictated and shifted accordingly. Want to find a kite instead? Match the shape of a diamond instead of ellipses for willow blades. Things can be modified once the base flow of tensors are built.

Then step up the game and learn how to learn driving techniques. Find the floor. Find a rock boulder outcropping. Find the sky. Find a tree. Find a branch. Classify all of them independently and assign probabilities that a certain vector might encounter the game stopper. Build targets and ween out the human interaction of: sending a command. Save some results. Map the data-set. Tweak the constraints by hand. My coffee roaster roast profiles is a great example of using Arduino and processing to display a simple graph, plot some points, check some IoT devices matching the criterion at a given Dt of t. The iterative process should look more like less permutations with expansion where we seek to use reduction functionality to effectively gain better outcome of our code generation process.

NVIDIA's CEO Huang said they are "all in". Cars. Planes. Robots. Whatever cuda crunchable GPU device you may dream up. It's definitely small in my mind like a Jetson Nano or simple to start learning like on a Raspberry Pi for this underwater vehicle. Some embedded processing boards boast the ability to use cameras to both navigate around a industrial plant and find and diagnose machinery in target. Under water it may look like a camera classifying images as it ingests them, notifying a team member of real time events, and moving along the water column making informative decisions about the space around it. For a flying device it may be an optical sensor trained to do the replacement work of a hal-effect sensor in a slip ring, with better precision and faster response than a traditional 60 Hz pwm frequency- or it may just be looking to auto correct a path out of an approaching flock of birds.

## How will we go about learning it here?
Actual development starts with an interactive interface. Build the "Administration Console". It should link simple sets of commands to execute at the chip level. Test each independently. One part at a time, one method for each part at a time. Define intuition on constraints in parsable text. Call it the todo list. Make your priority queue for that too. The admin console need to be designed with the thought of phasing its necessity out to build the constraints and phasing in the ability to interject situations that are deemed of curious nature- like the security camera which looks to maximize the normally lopped off tails or pay attention to the irregular spikes. This is where we may see seaweed in an coolant tube or a bird chopped in half. Hopefully never a 737 sensor failure.

The leap from old-skul data scientist with his vim and excel spreadsheet will be at which point to we relinquish or concede our need to control input parameter functionality and contortion and rely on our split duality software design to aide us in that balance. Think drive to linear not to parabolic. Think about how not to think for the device. Buddhist approach. Inner flame. Enlightenment comes in many forms.

Start K.I.S.S. Find a life coach. We are peers. Hive learning. Open Source. Revolutionize learning. It will physically be manifested as learning a little about architecture, understanding its constraints and how it is convolved or if it has noarch implications. Then we can talk about tooling and what that looks like to provide the ability to instill AI on any device. Then after we aren't afraid of doing the math anymore, make some simple linear algebra equations to solve that might be suited for an 8th grader to figure out. Teach it to our children before Geometry. Unleash the beast.

This also means we can have 2 sets of people here. Those that want to dig in deep and those that just want to see it work in their business. Either way the key is that we all are striving towards the same goal- empowering intelligence. 

## Project Research for Image Classification and Recognition 
This was the focus for me for some time, but now with the Covid crisis things have morphed a little. There is a deep dive on Tensors with PyTorch that directly hits the foundation for this project found [here](../../project-research/python-research/pytorch-deepdive). It requires getting at least conda and using Python- if you are comfortable with package management on your local machine, you may head right there and start programming a Convolutional Neural Network capable of ingesting batch images and doing cool stuff like developing your own classification algorithm. If that is making you sweat then maybe look at taking a step back and seeing what else this project may interest you in. 

## Futher Investigation performed in depth

### Docker
The world of computer science and software development is moving at light speed with this hiding under the surface of many key infrastructures- from AWS to Kubernetes down to Node and even here continual integration with embedded devices.

### Alpine or LFS
Strip down Unix flavor Linux and understand how POSIX and BSD machines work at a binary level. Use Cmake and clang and understand how gcc and friends are morphing into a musl library of lightweight utility functions to bundle up all this source code into "packages" that are piped through "channels" to do amazing things like develop real time diagnostics with minimal production grade overhead- or just make your own inlined addition to your favorite runtime/compiler/interpreter and push it through your own pipeline. In any event this rabbit hole goes deep, so choose the red or the blue one.