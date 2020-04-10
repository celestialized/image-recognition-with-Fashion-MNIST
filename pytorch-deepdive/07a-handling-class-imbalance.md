# Cornell Study found [here](https://arxiv.org/abs/1710.05381)
## A systematic study of the class imbalance problem in convolutional neural networks
Mateusz Buda, Atsuto Maki, and Maciej A Mazurowski. A systematic study of the class imbalance problem in convolutional neural networks. Neural Networks, 106:249–259, 2018.

This portion we are studying is to affirm and elaborate on some thoughts we may be having as we see other peoples take on manipulation of data-sets to suit use cases. The paper uses 3 datasets- MNIST, CIFAR-10 and ImageNet of increasing complexity to outline class imbalance and the means to address correction for it. The second paper [Salman H Khan, Mohammed Bennamoun, Ferdous Sohel, and Roberto Togneri. Cost sensitive learning of deep feature representations from imbalanced data. arXiv preprint arXiv:1508.03422, 2015.](https://arxiv.org/pdf/1508.03422.pdf) dives into the costs associated with imbalanced data. As these both are deep topics to address in a single sitting I am just going to begin with trying to relate the pieces of the deeplizard deepdive on tensorvision as it is up to this part of the notes. Then later come back and revisit each concept as a second means to nail down the components before tearing into the fashion-mnist and other potentially imbalanced real word examples in the near future.

Extensive comparison of several methods is used to tackle the issue: oversampling, undersampling, two-phase training, and thresholding that compensates for prior class probabilities.

These studies also introduce new methods to CNN where the system trained on the balanced data first and then the output layers are fine-tuned. Turning back to pytorch and torchvision we may relate this into procedures yet realized during this initial dissection. Wrapper functionality and drop in subtitution for datasets will also become clearer as these papers reveal themselves. The conclusions here is that Oversampling is the winner, but the researchers all state their contrived algorithms are superior to solving some subset of CNN/Deep Learning constructs, so the key take always are in understanding the use cases for each.

### Notes on classifiers and real life application  
We will see more examples in some classes than others for training sets. Thats the nature of the beast. This is class imbalance- with long tail distribution of data. For one class we may notice MAGNITUDES of difference in order over another class. Multilayer perceptions and traditional classifiers training can be effectively destroyed with this type of dis·sem·i·na·tion. Both convergence in the training phase and generalization of a model on the test set are the noticeable affects. The authors agree that there are probably similar issues prevailing in deep learning, but did not find any systematic studies on this. I concur. Looking at ways that others "Chop out the naysayers" and remove the n/a's lead me to question their techniques, the accuracy of output datasets, the intentions of the model, and a whole other slew of human introduced issues that mask true precision as noble as their attempts might be...some just try to make the code "Work"- I prefer why does it work?

### ROV and UI sidenote
todo
Natural Language Processing and speech recognition can also be modeled in this way: add in the Alexa skill for an easier way of marking waypoints to gps plotting and human interaction with the devices as a means to map our manual use of the rover. Simply say "mark" + "waypoint" and it goes.

https://www.linkedin.com/pulse/convoluted-neural-networks-roland-garceau

Before reading this I have developed a few preconceived notions of what the effect that naive (lets keep it innocent here) people may have on "molding" datasets while tackling torchvision and the Fashion-mnist drop in. If you haven't seen Jim Carey's Fun with Dick and Jane and Yes Man, its a comical approach to be or not to become "that guy". And reading this is yet again reinforcing me to rely on my instincts- you should too- develop the behavioral driven approach to ascertaining deep learning, and to model these rather challenging topics much in the same way we do in real life- that's the whole point here. Training computers to think like us really means application of our own ability to find introspect, and then to achieve this with high precision within our matrices. Can someone just please tell me how to learn judo in 5 minutes... My neck has taken its own joint lock here.

## Sampling methods
The most common approach is to operate on the data itself and not the model to achieve balance. We learned in stats to use oversampling, and then also undersampling. Then we also can discuss random majority undersampling which just removes (at random) portion of examples from majority classes. This is the "beware" zone. But we can also act on the classifier itself. Learning algorithms get modified- some by the introduction of weights to misclassification of examples from different classes or just by slight swiping magic of the hand with explicit tweaks to the class probabilities. Stranger danger.

## Dig into Cost Sensitive Learning
[Salman H Khan, Mohammed Bennamoun, Ferdous Sohel, and Roberto Togneri. Cost sensitive learning of deep feature representations from imbalanced data. arXiv preprint arXiv:1508.03422, 2015.](https://arxiv.org/pdf/1508.03422.pdf)
Deep learning networks need this approach: During training, our learning procedure jointly optimizes the class-dependent costs and the neural network parameters. Their comparisons with popular data sampling techniques and cost-sensitive classifiers demonstrate the superior performance of our (their)proposed method. I/you need more coffee?

The picture here is about not learning the characteristics of the tail distribution in minority classes, as we often do not pay as much attention to them unless directed to. We can, however, correctly classify examples from an infrequent and important minority class. Such things as realizing potentially dangerous activity on a security camera should be caught even though it may only happen infrequently. I have direct experience installing some security cameras that employ facial recognition, and assisting a software developer's brother make an "educated" decision in the matter. So in object detection this becomes a prevalent theme where we need to be able to adequately catch those minority classes to better correct the majority class sample. This means not cutting anything from the minority or majority classes to realize high precision. 

Relating this to resonant frequency testing for say, high stress aeronautical parts may mean that if you walk away from the channel sampling and come back to filtering out the noise later, you might want to know that the motor was on a off cycle when a particular observation spike happened, that way one may intuitively speak as to the validity of that particular channels recording and whether or not this information is of curious nature or not. Delete it if it is not, keep it if it is needed to train the model. There is a cost realization here of human, not artificial nature, however. 

Per the research paper: Most of the classification algorithms try to minimize the overall classification error during the training process. They, therefore, implicitly assign an identical misclassification cost to all types of errors assuming their equivalent importance. As a result the classifier tends to correctly classify and favour the more frequent classes. Moreover, they state that Class imbalance is avoided in nearly all competitive datasets during the evaluation and training procedures.

## Big Sets, little aspiration for accuracy? Nah, that's the point.
CIFAR−10/100, ImageNet, Caltech−101/256, and MIT−67 the collectors ensured that either all of the classes have a minimum representation with sufficient data, or that the experimental protocols are reshaped to use an equal number of images for all classes during the training and testing processes. The fashion-mnist reshaped to 6k each class too. hmm. The point pushes to the idea of collected object datasets as classes spread out drastically more than say a small handful of like minded representations, this mean that in order to do things such as say, take a picture and have your fancy recognition app spit out three hundred squares wrapped around every stinking thing that is in the image, assign a name to each of them, and then tell you with certainty that ALL of the outputs are above 99.9% accuracy, we have a lot more work to do Todo to get us out of Kansas.

## Invert the approach on the ROV
For dying reefs we may use the rover to find the surviving corals. Find the data on these little guys, mark a position, depth, etc. Send a requisition to grow them in a laboratory. Save mankind. Or at least the ocean. How about find all the shards of plastic in the water and siphon them all out?

## Back to Cost 
The reseach outlined is to jointly learn robust feature representations and classifier parameters, under a cost-sensitive setting. This enables us to learn not only an improved classifier that deals with the class imbalance problem, but also to extract suitably adapted intermediate feature representations from a deep Convolutional Network. With the CNN this allows us to modify the learning procedure to include class-dependent cost DURING training.

# Don't fall into the trap
Aim at escaping the "smarter" human inference problem of trying to tune the algorithm manually. e.g. no handcrafted cost matrix with tedious tasks for large number of classes- as I have forceably done trying to train a coffee roaster to make a gorgeous blonde roast while still cutting out enough caffeine without using chemical extraction methods, we surely could have benefitted from brining in a more qualified individual to write some revelation in C...

## Handling it better
So we can use data distribution and separability measures through stats analysis during the learning procedure. Don't be persuaded that more human interaction happens. During this process the class specific costs are only used during the training. Once the "optimal" CNN parameters are learnt, predictions can be made without any further modification to the trained network. So what does iterations do? Granularly add in layers of accuracy? Levels of transformation? Well, That is why we say that the optimal parameters have to be learned first, then we use it as needed. This is the vicious cycle and the crux of the chicken and the egg. This is their take on the $10 words- perturbation method.

## Learn the minute details
This is where the algorithm learns the more discriminative features. This is NOT data distortions, corrupted features, transformations, and activation dropout from traditional approaches, however.

## Cost research approach
* Adopt three widely used loss functions for joint cost-sensitive learning of features and classifier parameters in the CNN.
  * Improved loss functions have desirable properties such as classification calibration and guess-aversion.
* Analyze the effect of these modified loss functions on the back- propagation algorithm by deriving relations for propagated gradients.
* Their take was an algorithm for joint alternate optimization of the network parameters and the class- sensitive costs (Sec. III-D) that works for both binary and multi-class classification problems.

This needs a little validation here (It may have been done already), but their idea is that there is no significant increase to testing and training time (on what, the GPU cycles during runtime?) when introducing class-sensitive cost. Is it due to not assigning large negative values, propagating them up column data and removing (e.g. consuming the entire dataset instead of cropping). I'm not there yet nor fully convinced at this point. But, read on and see yourself. They say part IV-D shows it outperforms baseline procedures and state of the art approaches.

## They too state this
Class imbalance problem has concentrated mainly on two levels: the data level and the algorithmic level.
### Data drawback
Manipulate the data to be balanced through oversampling the minority classes and undersampling the majority for a distributed data sample. undersampling we lose potentially useful data of the majority, and oversampling makes training hard (the imbalance paper researchers agree) and is prone to cause over-fitting when we simply replicate random portions of the minority classes.

#### SMOTE Algorithm and its variants
To address over-fitting we generate new instances by linear interpolation between closely lying minority class samples. These synthetically generated minority class instances may lie inside the convex hull of the majority class instances, a phenomenon known as over-generalization.
#### Boarderline SMOTE
Only over-sample the close minority class samples which are near their class boundaries.
#### Safe-level SMOTE
Generate more synthetic samples. Ok. You told me to. Majority and minority class regions are not overlapping.
#### Local Neghborhood SMOTE
Neighboring majority class samples are considered when generating synthetic minority class samples and reports a better performance compared to the former variants. 

### Conclude Data level approaches
Mix under and over sampling procedures for balancing training data. Man this sounds like Arholt and stats all over again. Sentdex and deeplizard training CNN both reiterates this and the higher computational preprocessing costs to learning the classification model. Sentdex actually stuttered a little when speaking about creating the features necessary to predict google stock. Probably too much coffee. Re-record video and keep on trucking dude. This is why people spend lots of money hiring third party ATS and HR department filtration systems. These questions are hard to field from a board's perspective. No wonder people want proof of concept. Hire me with faith I will do the right thing. Sure. I've even seen this in Mission Statements for Huge companies that don't even know what the right questions to ask are. Should we plan for DR? Hell yeah.

## Algorithm approaches
Remember we are talking learning procedure here. This is aimed at cost sensitive learning and the means to improve sensitivity for the minority class. Its looking for the ink in MNIST, not the paper. 
This can be treated as subsetting and balanced subsetting smaller sets followed by intelligent sampling and cost-sensitive SVM to learn to deal with imbalance. Neurofuzzy models here use leave one out cross-validation on imbalanced sets. Also scaling kernel along with standard SVM was used to improve generalization ability of learned classifiers for skewed datasets. Li added in the idea of weights for minority classes with Adaboost with their Extreme Learning Machine and feed forward neural networking models. Great generalizing algorithm with speed 1000x faster than the backpropagation alg (BP). These were soft model SVMs aimed vis boosting, where each flavor uses distinct costs for different training examples to improve the performance of the learning algorithm. they all boast "superior performance". My buddy Neal and I did some cool radix sort optimizations in C some years ago, but we couldn't boast much more than being able to create our own buffer underruns efectively with calculated no-op insertions...POINT? None of these address imbalance learning of CNNS and SUPERVISED classification, segmentation, and recognition. They also are more bound to binary class problems, don't deal with joint feature and classifier learning. Also for our camera style vision tasks it doesn't deal with the often vastly segregated imbalanced class distributions.

### They do point to this, however, in the context of neural networks
This may need verification too:
Kukar and Kononenko showed that the incorporation of costs in the error function improves performance. However, their costs are randomly chosen in multiple runs of the network and remain fixed during the learning process in each run. In contrast, this paper presents the first attempt to incorporate automatic cost- sensitive learning in deep neural networks for imbalanced data.

Then they offered a few more recent approaches after the paper was submitted for review. Does that mean that this is not verified? Cost-sensitive loss function to replace traditional soft-max with a regression loss. It extends cost-functions in CNN. 

## Their boast
Their method automatically learns the balanced error function depending on the end problem. This is probably far enough to get a better feel of code writing with torhvision.

## Back to A systematic study of the class imbalance problem in convolutional neural networks- Main evaluation metric
Area under the receiver operating characteristic curve (ROC AUC) adjusted to multi-class tasks since overall accuracy metric is associated with notable difficulties in the context of imbalanced data.

## 5 Conclusions
* Effect of class imbalance on classification performance is detrimental.
* Oversampling was the prominent method to address class imbalance.
* Oversampling should be applied to the level that completely eliminates the imbalance, whereas the optimal undersampling ratio depends on the extent of imbalance.
* Oversampling does not cause overfitting of CNNs like some classic machine learning models
* Thresholding should be applied to compensate for prior class probabilities when overall number of properly classified cases is of interest

## 2 Categories of Methods for addressing imbalance
Methods for addressing class imbalance can be divided into two main categories. The first category is data level methods that operate on training set and change its class distribution. They aim to alter dataset in order to make standard training algorithms work. The other category covers classifier (algorithmic) level methods. These methods keep the training dataset unchanged and adjust training or inference algorithms. Moreover, methods that combine the two categories are available.
Undersampling and Oversampling for data level methods, and for classifier level methods include Thresholding, cost sensitive learning, and one class classification. Hybrid options exist with ensembling.

## Methods compared in this study
1. Random minority oversampling
2. Random majority undersampling
3. Two-phase training with pre-training on randomly oversampled dataset 
4. Two-phase training with pre-training on randomly undersampled dataset 
  * For the second phase, we keep the same hyperparameters and learn- ing rate decay policy as in the first phase. Only the base learning rate from the first phase is multiplied by the factor of 10−1
5. Thresholding with prior class probabilities
  * originally uses the im- balanced training set to train a neural network
6. Oversampling with thresholding
7. Undersampling with thresholding

## Oversampling, again
* Can be attributed to over-fitting.
* SMOTE came to be to deal with this
* You may preprocess the data to PERFORM more informed oversampling.
Cluster-based oversampling first clusters the dataset and then oversamples each cluster separately. This is the between calss AND within class imbalance issues discussed from the cost reasearch analysis. But Data-Boost-IM actually identifies difficult examples with boosting preprocessing and uses the realization to create synthetic data.
## Optimized Class Aware Sampling for Stochastic Gradient Descent
The main idea is to ensure uniform class distribution of each mini-batch and control the selection of examples from each class. This may be another angle towards the fashion-mnist to realize.

## Undersampling, again with a twist
Discarding good data can be frowned upon, but a rounded dataset may be sought after with/through this and oversampling. A more general approach than undersampling is data decontamination that can involve relabeling of some examples. Decontamination just sounds dirty.

## Classifier Level methods
### Thresholding
Also known as threshold moving or post scaling, adjusts the decision threshold of a classifier. It is applied in the test phase and involves changing the output class probabilities. There are many ways in which the network outputs can be adjusted. This one is to minimize arbitrary criterion using an optimization algorithm. Basic implementations just compensate for prior class probabilities. These are estimated for each class by its frequency in the imbalanced dataset before sampling is applied. Then another $10 dollar set of words Bayesian a posteriori probabilities. A posterior probability, in Bayesian statistics describe the realized or updated correction or probability for an outcome based off a recently realized set of outcomes or considering newly unfolded information. Its the probability that event 'A' will happen because of what happened in event 'B'. 

## Tie in Cost Sensitive learning now, again
One approach is threshold moving or post scaling that is applied in the inference phase after the classifier is already trained. Likewise it is to adapt the output of the network and use it as the backward pass of back propagation algorithm. Modifying a network to be cost sensitive is to also change the learning rate such that higher cost examples contribute more to the update of weights. Last we train the network by minimizing the misclassification cost instead of standard loss function. This is the research authors reasoning to omit implementation as it is their definition equivolent to oversampling.

## One-class classification
Novelty detection. Recognizes positive instances rather than discriminating between two classes. This is used as a means for anomaly detection where there exists extremely high imbalance. identity functions and auto encoders for associative mapping.

## Ensembling = Stir the paint
Mix of all the methods as a wrapper. Wrigleys, Juicy Fruit? EasyEnsemble and BalanceCascade. SMOTEBoost. Two phase training has been used in CNN's for brain tumor segmentation- even though the task was image segmentation, it was approached as a pixel level classification. The method involves network pre-training on balanced dataset and then fine-tuning the last output layer before softmax on the original, imbalanced data. I hope someone can elaborate on this. I hit it twice. 


