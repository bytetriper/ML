import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init
from matplotlib import pyplot as plt
import os
from Data_Manager import Data_Manager
class NN_Net(nn.Module):
    def __init__(self,inputsize:int,outputsize:int) -> None:
        super().__init__()
        self.inputsize=inputsize
        self.outputsize=outputsize
        self.sequence=nn.Sequential(
            nn.BatchNorm1d(inputsize),
            nn.Linear(inputsize,outputsize),
        )
        for m in self.modules():
            if isinstance(m,nn.Linear):
                init.xavier_uniform_(m.weight.data)
    def forward(self,input:torch.Tensor)->torch.Tensor:
        input=input.view(-1,self.inputsize)
        pi=input
        pi=self.sequence(pi)
        return F.log_softmax(pi)
class NNET_Arch():
    def __init__(self,inputsize:int,outputsize:int) -> None:
        self.nnet=NN_Net(inputsize,outputsize)
        self.optimizer=torch.optim.Adam(self.nnet.parameters(),lr=0.01)
    def loss(self,output,y):
        lossf=nn.KLDivLoss(reduction="batchmean")
        return lossf(output,y)
    def train(self,batch,label):
        self.nnet.train()
        pi=self.nnet(batch)
        loss=self.loss(pi,label)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss
    def predict(self,s):
        self.nnet.eval()
        pi=self.nnet(s)
        return torch.argmax(pi,dim=1)
def test():
    dataM=Data_Manager("/root/autodl-tmp/train/pytorch_train")
    net=NNET_Arch(100,4)
    loss_his=[]
    for step in range(0,300):
        dataM.Make_Data(1000,100)
        batch,label=dataM.Return_Data()
        loss=net.train(batch,label)
        loss_his.append(loss.item())
    dataM.Make_Data(200,100)
    batch,label=dataM.Return_Data()
    v=dataM.Return_Ans()
    plt.plot(range(len(loss_his)),loss_his)
    plt.savefig(os.path.join(dataM.datapath,"tmp.jpg"))
    acc=torch.sum(net.predict(batch)==v)/200
    return acc
if __name__=="__main__":
    print(test())