# A repo full of toy nets👨‍🦼

## CNN
### CNN on Cifar 10
**Basic Idea**:
* Several Conv layers
* Use stride 2 as down-sampling
* Adam/lr starts at 3e-3/Exponential 0.85
* Use Batch Normalization(slightly better than dropout)

**BaseLine**:
* Accuracy Average:73-75%
* Best Porformance:80.52%


**Note**:
* No multi-model
* No transfer learning
* No data augmentation

## RNN
### Vanilla RNN

#### Data: ShakesPeare Classical Works
**Basic Idea**:
* tanh as activation function
* DataSet size: about 10000 chars
* Backward all at once
* Batchsize about 200
  
**BaseLine**:
* Works Incredibly Bad.
* Most of the time speaks the last sentence, sometimes jibber jabber.
* Hard to Converge

### LSTM

**SOTA:like a modern poet** (Yet)

**Basic Idea**:
* classical LSTM with only 1-layer
* hiddensize=1024 at the moment
* Use a bit L1-loss to restrict the matrix from overfiting
* L1-loss doesn't really have significant impact.
* Trained under a poem dataset (about 20M)

**BaseLine**:
* Works like a poet.
* Can be trained to generate a poem with a given first sentence.
* Final loss about 1.3, but works pretty well.


### PCA
**Basic Idea**:
* Use a downward CNN to extract features,then upconv to reconstruct the image.
* Use MSE(now L1_Loss) as loss function
* After training the net, freeze the downconv part, and add a MLP to do classification.

**BaseLine**:
* works normally on cifar10
* PCA works great on cifar10, loss about 0.01. 
* MLP does some job on the down conv part, accuracy about 54%.
* still have space to improve.


### Few-Shot CIFAR10
**Basic Idea**:
* Use a PCA model pretrained on the whole cifar10(does not require any labeling)
* Add a MLP on PCA to do classification(size:2048)
* Cut the train set's size down by random select a subset of the train set of CIFAR10.
* Adam/lr=8e-3/StepLR:100 step;dacay=0.9/batchsize=512(which is almost the set size)
* Use pre-data augmentation to augment the whole CIFAR10 train set: VerticalFlip HorizentalFlip Sharpness(0.1) **before selecting the subset**

**BaseLine**:
* Accuracy deteriorates as the size of the train set shrinks in size
* Have an accuracy of about **43%** when train set size is 2000 with pre-data augmentation.
* Have an accuracy of about **37%** when train set size if 6000 with no data augmentation.

### Style-Transfer

**Basic Idea**:
* Use a Pretrained VGG16 as base model
* Use Gram-matrix to copy texture and feature inversion to copy content
* Almost the same with CS231n 2017 spring's class demo
* No FastCNN, just the basics
* Tried on PCA(or trained classifier on CIFAR10), but did not work well.
* Larger images usually have better performance
* use a decaying weight of $0.9^n$ on the gram-matrix of layer n
* use $R(x)=\sum\limits_{i=1}^{n}(x_i-x_{i+1})^2$ as regularization

**BaseLine**:
* Worked pretty well, especially on big pics.
* a demo is [here](Style_Transfer/output.jpg), trained with a 2k img of Vangogh's <Starry Night>.

