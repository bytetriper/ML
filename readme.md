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

