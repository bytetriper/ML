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



