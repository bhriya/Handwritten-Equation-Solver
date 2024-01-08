# Handwritten-Equation-Solver
## Abstract
In this project a convolutional neural network approach is proposed
towards handwritten digit and basic operator recognition by extracting
features and converting patterns into categories. Further
implementing a basic calculator model, that can evaluate an expression in
a given image by first separating out numbers and operators using contour
plotting and then predicting their class using the model. Experiments are
performed on the 333,895 images belonging to 16 different classes. Major
libraries used for implementation are opencv, numpy and tensorflow.

## Working
**1. Image Pre-Processing**
The 333,895 images were divided in a 1:4 ratio between validation and
training set. These images were loaded at a target size of (40,40) and in
grayscale mode. These images were also rescaled by a factor of 1./255 so
that pixel values are not too high for the model to process.
**2. The Architecture**
The architecture of our neural network contains four learned layers, two
convolutional and two fully connected. The activation function we used for
the first three layers (two convolutional and one fully connected) is, rectified
linear unit (ReLu), as it trains much faster and helps us to deal with the
problem of vanishing gradients. We also used max pooling layers with
kernel size two for training our model in order to reduce overfitting. Our
network maximizes the multinomial logistic regression objective, which is
equivalent to maximizing the average across training cases of the
log-probability of the correct label under the prediction distribution.
The first convolutional layer is fed with 40×40×1 input images and consists
of 32 kernels of size 3x3 using a stride of 1 pixel (this is the distance
between the receptive field centers of neighboring neurons in a kernel
map). The second convolutional layer is fed with the output of the previous
layer and filters it with 64 kernels of size 3x3. Then we flatten the output
channels and pass it to the fully connected layer that has 128 neurons and
then finally to our output layer that uses a 16-way softmax which produces
a distribution over the 16 class labels.

**3. Detection**
Firstly the image was converted into grayscale using cvtColor function and
then to sharpen the image gaussian filter was applied on the image.This
was then followed by canny edge detection method for identifying the
edges of the numbers and symbols and then finally plotting contours.
All these were applied using python openCV library functions.

**4. Improvements**
The algorithm used in this project is a basic algorithm for calculating simple 
the value of simple equations. It can be further modified to solve for derivative and 
integration problems.

**Dataset**
The dataset can be found here .
