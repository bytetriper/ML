from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init
from matplotlib import pyplot as plt
import os
import random
from typing import Tuple
from typing import List
from typing import Dict
from tqdm import tqdm
class nnet(nn.Module):
    def __init__(self,inputsize:int,ind:int,outputsize:int,device:torch.device) -> None:
        super(nnet,self).__init__()
        self.inputsize=inputsize
        self.dim=ind
        self.outputsize=outputsize
        self.net=nn.Sequential(
            nn.BatchNorm2d(self.dim),
            nn.Conv2d(self.dim,64,7,1,3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64,64,5,1,2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64,128,5,1,2),
            nn.ReLU()
            #nn.BatchNorm2d(128),
        )
        self.fc=nn.Sequential(
            nn.Linear(32*32*128,128,device=device),
            nn.ReLU()
        )
        self.fc2=nn.Sequential(
            nn.Linear(128,outputsize,device=device)
            #nn.ReLU()
        )
        for m in self.modules():
            if isinstance(m,nn.Linear):
                init.xavier_uniform_(m.weight.data)
        self.F=F.log_softmax
    def forward(self,batch:torch.Tensor)->torch.Tensor:
        pi=batch
        pi=self.net(pi)
        pi=pi.view(pi.shape[0],-1)
        pi=self.fc(pi)
        pi=self.fc2(pi)
        #print(pi.shape)
        return pi
class NNET_Wrapper():
    def __init__(self,device:torch.device,loadpath:str=None) -> None:
        if loadpath:
            self.nnet=torch.load(loadpath)
        else:
            self.nnet=nnet(32,3,10,device)
        self.device=device
        self.nnet.to(device)
        self.optimizer=torch.optim.Adam(self.nnet.parameters(),lr=3e-4)
        self.scheduler=torch.optim.lr_scheduler.ExponentialLR(self.optimizer,0.85)
        
    def loss(self,data:torch.Tensor,label:torch.Tensor)->torch.Tensor:
        F=nn.CrossEntropyLoss()
        #print(data.shape)
        #print(label.shape)
        return F(data,label)
    def train(self,dataloader,train_time:int)->torch.Tensor:
        loss_history=[]
        self.nnet.train()
        Interval=100
        for epoch in range(train_time):
            #batch_idx=random.sample(range(len(data)),batchsize)
            #batch_idx=range(len(data))
            #batch_data=data[batch_idx]
            #batch_label=label[batch_idx]
            with tqdm(total=len(dataloader)) as tbar:
                tbar.set_description(f'epoch:{epoch:d}')
                for i,d in enumerate(dataloader):
                    data,label=d
                    data,label=data.to(self.device),label.to(self.device)
                    #print(label)
                    pi=self.nnet(data)
                    loss=self.loss(pi,label)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    tbar.update(1)
                    tbar.set_postfix(loss=f'{loss.item():.4f}',lr=f'{self.scheduler.get_last_lr()[0]:.6f}')
                    #print(loss.item(),end=' '
                    #if i%Interval==0:
                    #    print(f"Epoch:{epoch:d} Iter:{i:d} Loss:{loss.item():.4f}")
                    loss_history.append(loss)
            self.scheduler.step()
        return loss_history
    def predict(self,data:torch.Tensor)->torch.Tensor:
        self.nnet.eval()
        return torch.argmax(self.nnet(data),dim=1)
    def Save(self,savepath:str)->None:
        #assert(os.path.exists(savepath))
        torch.save(self.nnet,savepath)
Params={
    "epoch":1,
    "train_time":10
}
def Test_Model(loadpath:str=None):
    Traindata=datasets.CIFAR10(
        root="/root/autodl-tmp/data",
        train=True,
        download=False,
        transform=ToTensor()
    )
    Loader=DataLoader(
        dataset=Traindata,
        batch_size=256,
        shuffle=True,
        num_workers=2
    )
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=NNET_Wrapper(device,loadpath)
    #print(data)
    loss=[]
    loss_h=model.train(Loader,Params['train_time'])
    [loss.append(i.item()) for i in loss_h]
    plt.plot(range(len(loss)),loss)
    plt.savefig(r'/root/autodl-tmp/ML/CNN/cifar10/loss.png')
    model.Save(r'/root/autodl-tmp/ML/CNN/cifar10/model.pth')
def Predict_Model(modelpath:str)->float:
    device=torch.device('cpu')
    model=NNET_Wrapper(device,modelpath)
    Testdata=datasets.CIFAR10(
        root="/root/autodl-tmp/data",
        train=False,
        download=False,
        transform=ToTensor()
    )
    data=[]
    label=[]
    for d in Testdata:
        data.append(d[0])
        label.append(d[1])
    #data=data[0:5000]
    #label=label[0:5000]
    data=torch.stack(data).to(device)
    label=torch.Tensor(label).to(device)
    pi=model.predict(data)
    acc=torch.sum(pi==label)/len(label)
    return acc
if __name__=="__main__":
    #Test_Model(r'/root/autodl-tmp/ML/CNN/cifar10/model.pth')
    Test_Model()
    acc=Predict_Model(r'/root/autodl-tmp/ML/CNN/cifar10/model.pth')
    print(acc)