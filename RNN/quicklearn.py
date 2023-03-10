import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init
from matplotlib import pyplot as plt
import os
from typing import Tuple
from typing import List
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
    def __init__(self,inputsize:int,hiddensize:int,endsign:int,startsign:int,device:torch.device) -> None:
        super().__init__()
        self.Wx=nn.Linear(inputsize,hiddensize,False)
        self.Wh=nn.Linear(hiddensize,hiddensize,False)
        self.Wy=nn.Linear(hiddensize,inputsize,False)
        self.device=device
        self.inputsize=inputsize
        self.hiddensize=hiddensize
        self.LossF=nn.KLDivLoss(reduction="batchmean")
        self.endsign=endsign
        self.startsign=startsign
        self.t=torch.zeros(hiddensize)
        self.Tanh=nn.Tanh()
        self.F=nn.Softmax()
        for m in self.modules():
            if isinstance(m,nn.Linear):
                init.xavier_uniform_(m.weight.data)
    def forward(self,s:torch.Tensor,initialState:torch.Tensor=None)->Tuple[torch.Tensor,List[torch.Tensor]]:
        len,_=s.shape
        if initialState!=None:
            self.t=initialState
        else:
            self.t=torch.zeros(self.hiddensize).to(self.device)
        output=[]
        for i in range(len):
            x=s[i]
            tmp=self.Tanh(self.Wx(x)+self.Wh(self.t))
            y=self.F(self.Wy(tmp))
            output.append(y)
            self.t=tmp
        return self.t,output
    def predict(self,initialState:torch.Tensor)->List[torch.Tensor]:
        self.t=initialState
        select=None
        ans=[]
        x=[(i==self.startsign) for i in range(self.inputsize)]
        x=torch.Tensor(np.array(x,dtype=float)).to(self.device)
        cnt=0
        maximum=10
        while not select or select.item() != self.endsign:
            self.t=self.Tanh(self.Wx(x)+self.Wh(self.t))
            y=self.F(self.Wy(self.t))
            select=torch.argmax(y)
            ans.append(select)
            cnt+=1
            if cnt > maximum:
                break
        return ans
class NNET_Wrapper():
    def __init__(self,inputsize:int,hiddensize:int,endsign:int,startsign:int,device:torch.device) -> None:
        self.nnet=RNN(inputsize,hiddensize,endsign,startsign,device)
        self.optimizer=torch.optim.Adam(self.nnet.parameters(),lr=0.01)
        self.device=device
        self.nnet.to(self.device)
    def lossF(self,input:List[torch.Tensor],label)->torch.Tensor:
        lossF=nn.KLDivLoss()
        loss=torch.Tensor([0]).to(self.device)
        seq_len=len(input)
        for i in range(seq_len):
            loss+=lossF(torch.log(input[i]),label[i])
        return loss
    def train(self,data:torch.Tensor)->torch.Tensor:
        self.nnet.train()
        torch.zeros
        h,output=self.nnet(data)
        loss=self.lossF(output[0:-2],data[1:-1])
        print(loss.shape)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss
    def predict(self,data:torch.Tensor)->torch.Tensor:
        self.nnet.eval()
        with torch.no_grad():
            output=self.nnet.predict(data)
        return output
    ###FUCK! I'll Come Back!
if __name__=="__main__":
    if torch.cuda.is_available():
        device=torch.device('cuda')
    else:
        device=torch.device('cpu')
    net=NNET_Wrapper(20,20,1,9,device)
    s=torch.rand(20).to(device)
    pi=torch.rand(10,20).to(device)
    st=torch.zeros(1,20).to(device)
    end=torch.zeros(1,20).to(device)
    st[0][1]=1
    end[0][9]=1
    pi=torch.cat((st,pi,end))
    print(net.predict(s))
    
