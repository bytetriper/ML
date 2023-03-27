# use img as input
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import DataLoader
from torchvision import datasets
from matplotlib import pyplot as plt
from tqdm import tqdm
from typing import List
from typing import Iterator
from torchvision.transforms import ToTensor
import numpy as np
import sys
Params={
    "hiddensize":8,
    "device":torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "datapath":"/root/autodl-tmp/data",
    "modelpath":"./model.pth",
    "traintime":10,
    "epoch":5,
    "LossPath":"./loss.png",
    "FlossPath":"./floss.png",
    "batchsize":128,
    "imgpath":"./img.png",
    "imgpath_origin":"./img_origin.png",
}
class PCA(nn.Module):
    def __init__(self, input_channel,hiddensize):
        super(PCA, self).__init__()
        self.Down_Net=nn.Sequential(
            nn.Conv2d(input_channel,32,5,1,2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.Conv2d(32,64,5,1,2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.Conv2d(64,64,5,2,2), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.Conv2d(64,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.Conv2d(64,hiddensize,5,2,2),
            nn.BatchNorm2d(hiddensize),
            nn.ReLU(),
            nn.Conv2d(64,hiddensize,5,2,2),
            nn.BatchNorm2d(hiddensize),
            nn.ReLU(),
        )
        #self.fc_d=nn.Linear(64*4*4,hiddensize)# maybe be removed in the future
        self.Up_Net=nn.Sequential(
            nn.ConvTranspose2d(hiddensize,64,kernel_size=5,stride=2,padding=2,output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64,64,kernel_size=5,stride=2,padding=2,output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64,64,kernel_size=5,stride=2,padding=2,output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #nn.ConvTranspose2d(64,64,kernel_size=5,stride=2,padding=2,output_padding=1),
            #nn.BatchNorm2d(64),
            #nn.ReLU()
        )
        self.Conv2one=nn.Conv2d(64,input_channel,5,1,2)
        self.Normal=F.normalize
    def forward(self,s):
        s=self.Down_Net(s)
        #print(s.shape)
        s=self.Up_Net(s)
        #print(s.shape)
        s=self.Conv2one(s)
        #normalize s into 0-1
        return s
class NNET():
    def __init__(self,inputsize:int,hiddensize:int,device:torch.device) -> None:
        self.hiddensize=hiddensize
        self.inputsize=inputsize
        self.device=device
        self.net=PCA(inputsize,hiddensize)
        self.net.to(self.device)
        self.optimizer=torch.optim.Adam(self.net.parameters(),lr=0.01)
        self.scheduler=torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=100,gamma=0.9)
        self.cnter=self.train_cnt()
    def train_cnt(self)->Iterator[int]:
        cnt=1
        while True:
            yield cnt
            cnt+=1
    def train(self,datalist:DataLoader,epoch:int) -> List[float]:
        self.net.train()
        loss_his=[]
        for i in range(epoch):
            bar=tqdm(range(len(datalist)))
            bar.set_description_str("Epoch:{}".format(i))
            for batch_idx,(data,label) in enumerate(datalist):
                data,label=data.to(self.device),label.to(self.device)
                self.optimizer.zero_grad()
                output=self.net(data)
                loss=F.mse_loss(output,data)
                loss.backward()
                loss_his.append(loss.item())
                self.optimizer.step()
                bar.update(1)
                bar.set_postfix_str('SubEpoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                        batch_idx, batch_idx, len(datalist),
                        100.*batch_idx/len(datalist), loss.item()))
            self.scheduler.step()
        return loss_his
    def simple_train(self,data:torch.Tensor,epoch:int)->List[float]:
        self.net.train()
        data=data.to(self.device)
        loss_his=[]
        bar=tqdm(range(epoch))
        bar.set_description_str("Epoch:{}".format(self.cnter.__next__()))
        for i in range(epoch):
            self.optimizer.zero_grad()
            output=self.net(data)
            loss=F.mse_loss(output,data)
            loss.backward()
            loss_his.append(loss.item())
            self.optimizer.step()
            bar.update(1)
            bar.set_postfix_str('SubEpoch: {} \tLoss: {:.6f}'.format(
                    i,loss.item()))
            self.scheduler.step()
        return loss_his
    def predict(self,data:torch.Tensor) -> torch.Tensor:
        self.net.eval()
        with torch.no_grad():
            data=data.to(self.device)
            output=self.net(data)
            return output
# save model
def save_model(model:nn.Module,path:str):
    torch.save(model.state_dict(),path)
# load model
def load_model(model:nn.Module,path:str):
    model.load_state_dict(torch.load(path))
    return model
# create a NNET model
def create_model(inputsize:int,hiddensize:int,device:torch.device)->NNET:
    return NNET(inputsize,hiddensize,device)

def test_model(load:bool=False,save:bool=False)->NNET:
    torch.cuda.empty_cache()
    Dataset=datasets.CIFAR10(root=Params["datapath"],train=True,download=False,transform=ToTensor())
    trainloader=DataLoader(Dataset,batch_size=Params["batchsize"],shuffle=True)
    #move the data in dataset into GPU using for loop and .to method
    #for i in range(len(Dataset)):
    #    Dataset[i]=(Dataset[i][0].to(Params["device"]),Dataset[i][1])
    datalist=[]
    #for data,_ in trainloader:
    #    data=data.to(Params["device"])
    #    data=F.normalize(data)
    #    datalist.append(data)
    device = Params["device"]
    net=create_model(3,64,device)
    if load:
        load_model(net.net,Params["modelpath"])
    loss=net.train(trainloader,Params["epoch"])
    plt.plot(range(len(loss)),loss)
    plt.savefig(Params["LossPath"])
    if save:
        save_model(net.net,Params["modelpath"])
    return net
#show a img in CIFAR10 and prediction from a NNET model
def show_img(net:NNET,dataset:datasets.CIFAR10):
    # show a random img in CIFAR10
    img,label=dataset[np.random.randint(0,len(dataset))]
    Inputimg=img.unsqueeze(0)
    Inputimg=Inputimg.to(Params["device"])
    output=net.predict(Inputimg)
    output=output.squeeze(0)
    output=output.cpu()
    output=output.detach().numpy()
    #output=(output+1)/2
    output=np.transpose(output,(1,2,0))
    plt.imshow(output)
    plt.savefig(Params["imgpath"])
    plt.imshow(img.numpy().transpose(1,2,0))
    plt.savefig(Params["imgpath_origin"])
    return label
if __name__ == "__main__":
    # read from argv[1] to decide whether to train or not(if "train" then train,otherwise test only)
    if len(sys.argv)>1:
        if sys.argv[1]=="train":
            # read from argv[2] to decide whether to load model or not(if "load" then load,otherwise create a new model)
            if len(sys.argv)>2:
                if sys.argv[2]=="load":
                    net=test_model(load=True,save=True)
                else:
                    if sys.argv[2]=="new":
                        net=test_model(load=False,save=True)
                    else:
                        raise RuntimeError("Inappropriate argv[2]")
            else:
                raise RuntimeError("Please input argv[2] to decide whether to load model or not(if 'load' then load,otherwise create a new model)")
        else:# create class NNET and load model
            net=create_model(3,64,Params["device"])
            load_model(net.net,Params["modelpath"])
    else:
        raise RuntimeError("Please input argv[1] to decide whether to train or not(if 'train' then train,otherwise test only)")
    Dataset=datasets.CIFAR10(root=Params["datapath"],train=True,download=False,transform=ToTensor())
    # assure that net is not unbounded
    label=show_img(net,Dataset)
    print(label)

