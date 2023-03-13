import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init
from matplotlib import pyplot as plt
import os
from typing import Tuple
from typing import List
from typing import Dict
import numpy as np
from Data_Manager import Data_Manager
inputsize=100
hiddensize=40
num_layers=2
class Encoder():
    def __init__(self,map:Dict[str,int],endsign:str) -> None:
        self.map=map
        self.endsign=endsign
        self.reversed_map={map[k]:k for k in map.keys()}
    def encode(self,text:str,device:torch.device)->torch.Tensor:
        data=torch.zeros(len(text),len(self.map)).to(device)
        for i in range(len(text)):
            s=text[i]
            if s not in self.map.keys():
                data[i][self.map[self.endsign]]=1
                break
            data[i][self.map[s]]=1.
        return data
    def decode(self,output:List[torch.Tensor])->str:
        ans=""
        for y in output:
            ans+=self.reversed_map[y.item()]
        return ans
class RNN(nn.Module):
    def __init__(self,inputsize:int,hiddensize:int,startsign:int,endsign:int,device:torch.device) -> None:
        super().__init__()
        self.Wx=nn.Linear(inputsize,hiddensize)
        self.Wh=nn.Linear(hiddensize,hiddensize)
        self.Wy=nn.Linear(hiddensize,inputsize)
        self.device=device
        self.inputsize=inputsize
        self.hiddensize=hiddensize
        self.endsign=endsign
        self.startsign=startsign
        self.t=torch.zeros(hiddensize)
        self.Tanh=nn.Tanh()
        self.F=nn.Softmax()
        for m in self.modules():
            if isinstance(m,nn.Linear):
                init.xavier_uniform_(m.weight.data)
    def forward(self,s:torch.Tensor,initialState:torch.Tensor=None)->Tuple[torch.Tensor,List[torch.Tensor]]:
        seq_len,_=s.shape
        if initialState!=None:
            self.t=initialState
        else:
            self.t=torch.zeros(self.hiddensize).to(self.device)
        output=[]
        for i in range(seq_len):
            x=s[i]
            tmp=self.Tanh(self.Wx(x)+self.Wh(self.t))#next state
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
        maximum=20
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
    def __init__(self,inputsize:int,hiddensize:int,startsign:int,endsign:int,device:torch.device) -> None:
        self.nnet=RNN(inputsize,hiddensize,startsign,endsign,device)
        self.optimizer=torch.optim.Adam(self.nnet.parameters(),lr=1e-5)
        self.device=device
        self.nnet.to(self.device)
    def lossF(self,input:List[torch.Tensor],label)->torch.Tensor:
        lossF=nn.KLDivLoss(reduction="batchmean")
        loss=torch.Tensor([0]).to(self.device)
        seq_len=len(input)
        for i in range(seq_len):
            loss+=lossF(torch.log(input[i]),label[i])
        return loss
    def train(self,data:torch.Tensor)->torch.Tensor:
        self.nnet.train()
        h,output=self.nnet(data)
        loss=self.lossF(output[0:-1],data[1:])
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
def test():
    map={}
    for i in range(26):
        map[chr(i+ord('a'))]=i
    map[' ']=26
    map[':']=27
    map['\n']=28 #end sign
    map[',']=29
    map['.']=30
    map['"']=31
    map["'"]=32
    map[';']=33
    map['*']=34# start sign
    inputsize=len(map)
    hiddensize=4000
    train_batch=200
    if torch.cuda.is_available():
        device=torch.device('cuda')
    else:
        device=torch.device('cpu')
    net=NNET_Wrapper(inputsize,hiddensize,map['*'],map['\n'],device)
    Enc=Encoder(map,'*')
    st=torch.zeros(1,inputsize).to(device)
    st[0][map['*']]=1.
    end=torch.zeros(1,inputsize).to(device)
    end[0][map['\n']]=1.
    loss_history=[]
    data=["i say:Hello","he say:goodbye","dope say:ahscui","lll say:aschsg uyu"]
    for i in range(len(data)*train_batch):
        s=Enc.encode(data[i%len(data)],device)
        train_epoch=torch.cat((st,s,end))
        #print(train_epoch)
        loss=net.train(train_epoch)
        loss_history.append(loss.item())
    plt.plot(range(len(data)*train_batch),loss_history)
    plt.savefig(r"/root/autodl-tmp/ML/RNN/loss.png")
    pi=torch.zeros(hiddensize).to(device)
    #print(net.predict(pi))
    print(Enc.decode(net.predict(pi)))

if __name__=="__main__":
    test()