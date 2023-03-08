import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init
from matplotlib import pyplot as plt
import os
from typing import Tuple
import numpy as np
inputsize=100
hiddensize=40
num_layers=2
class Encoder():
    def __init__(self) -> None:
        pass
    def encode(self,text:str,device:torch.device)->torch.Tensor:
        data=[]
        for t in text:
            y=[i==ord(t) for i in range(0,128)]
            data.append([y])
        data=torch.tensor(np.array(data)).to(device)
        return data
class RNN(nn.Module):
    def __init__(self,inputsize,hiddensize) -> None:
        super().__init__()
        self.Wx=nn.Linear(inputsize,hiddensize,False)
        self.Wh=nn.Linear(hiddensize,hiddensize,False)
        self.Wy=nn.Linear(hiddensize,inputsize,False)
        self.t=torch.zeros(hiddensize)
    def forward(self,s:torch.Tensor,label:torch.Tensor,initialState:torch.Tensor)->Tuple[torch.Tensor,torch.Tensor]:
        len,_=s.shape
        self.t=initialState
        F=nn.Softmax()
        lossF=nn.KLDivLoss(reduction="batchmean")
        loss=torch.tensor(0)
        for i in range(len):
            x=s[i]
            tmp=self.Wx(x)+self.Wh(self.t)
            y=F(self.Wy(tmp))
            loss+=lossF(torch.log(y),label[i])
            self.t=tmp
        return self.t,loss
class NNET_Wrapper():
    def __init__(self,inputsize,hiddensize) -> None:
        self.nnet=RNN(inputsize,hiddensize)
        self.optimizer=torch.optim.Adam(self.nnet.parameters(),lr=0.01)
    def train(self,data):
        pass
    ###FUCK! I'll Come Back!
if __name__=="__main__":
    test_input=torch.randn(20,1,inputsize)
    model=nn.RNN(inputsize,hiddensize,num_layers,False)
    out,h=model(test_input)
    print(out.shape,h.shape)
