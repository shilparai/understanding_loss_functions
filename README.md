# Loss Functions and Optimization
Role of loss functions in Parameterized Learning

We know that, we don’t have control over the data **(xi,yi)** (it is fixed and given), but we do have control over these weights and we want to set them so that the predicted class scores are consistent with the ground truth labels in the training data and hence, minimizes the error.

The first component of this approach is to define the score function that maps the pixel values of an image to confidence scores for each class. Let’s assume that we have a training dataset of images, each associated with a label y which is class (cat,dog). 
Let's number of images in training dataset are, each with dimension (h x w x c). c stands for number of channels. Hence, size of input training dataset is going to be (N,h x w x c).  ***_D -> h x w x c_*** shows the number of pixels in nth image

We will now define the score function f:R<sup>D</sup>↦R<sup>K</sup> that maps the raw image pixels to class scores.

# Linear Classifier
Definition of linear classifier, which maps weights matrix **W** of size **C x D** with features vector **X** of size **D** (in our cases, it's total number of pixels)

<p align="center">
  <b>f(x<sub>i</sub>, W, b) = Wx<sub>i</sub> + b</b>
</p>

Linear classifier computes the score of a class as a weighted sum of all of its pixel values across all 3 of its color channels.

<img height = "204" src = "https://github.com/shilparai/understanding_loss_functions/blob/master/linear_classifier.jpg">


# Multiclass Support Vector Machine loss
SVM loss is the commonly used loss function. Simple idea behind SVM loss is that, it tends to have higher scores for the correct class of i<sup>th</sup> image and lower scores to the incorrect classes. This loss function is also called *Hinge Loss Function*. **multiclass_support_vector.py** shows the implementation of SVM loss function for cat-dog datasets.
