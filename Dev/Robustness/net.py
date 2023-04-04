import torch
import torch.nn.functional as F
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torchvision import datasets
from typing import List,Iterator,Union,Tuple
from torch.utils.data import DataLoader
import numpy as np
import pickle as pkl
import torchvision
from torchvision.models.vgg import VGG16_Weights
import cv2
Params={
    "modelpath":"/root/autodl-tmp/ML/Models/PCA.pth",
    "classifierpath":"/root/autodl-tmp/ML/Dev/Robustness/Classifier.pth",
    "device":torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "lr":5e-3,
    "traintime":1000,
}

class PretrainedPCA(nn.Module):
    def __init__(self, path:str, input:torch.Tensor):
        super(PretrainedPCA, self).__init__()
        self.path = path
        self.load_Classifier()
        self.name_dict=dict(self.Classifier.named_modules())
        self.input=input.to(Params["device"])
        self.input.requires_grad=True
        #only perform gradient ascent on input
        self.optimizer=torch.optim.Adam([self.input],lr=Params["lr"])
        self.scheduler=torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=200,gamma=0.99)
        self.tcnt=self.train_cnt()
    #def a train_cnt that returns a Iterator to record the train time
    def train_cnt(self)->Iterator:
        cnt=0
        while True:
            yield cnt
            cnt+=1
    def test_run(self)->torch.Tensor:
        ### deprecated
        self.Classifier.eval()
        x=self.Classifier(self.input)
        loss=F.l1_loss(x,self.input)
        return loss
    def load_Classifier(self):
        ### Use pretrained Classifier to perform PCA
        #self.Classifier=Mini_Classifier(PCA(3,PCA_module.Params["hiddensize"]).Down_Net)
        #self.Classifier.load_state_dict(torch.load(Params["classifierpath"],map_location=Params["device"]))
        #self.Classifier.to(Params["device"])
        self.Classifier=torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT).to(Params["device"])
        self.layers=[]
        for layer in self.Classifier.children():
            if not isinstance(layer,nn.Sequential):
                if isinstance(layer,nn.Linear):
                    self.layers.append(nn.Flatten())
                self.layers.append(layer)
            else:
                [self.layers.append(sublayer) for sublayer in layer.children()]
        self.layers.append(nn.Linear(1000,10))

    def forward(self,s:torch.Tensor=None,compute_gram:bool=False,saved_activation:bool=False)->torch.Tensor:
        if s==None:
            s=self.input
        self.gram=[]
        self.activation=[]
        for layer in self.layers[:30]:
            s=layer(s)
            if isinstance(layer,nn.ReLU):
                if compute_gram:
                    b,c,h,w=s.shape
                    #print(b,c,h,w)
                    tmps=s.view(b,c,h*w)
                    #print(tmps.shape)
                    #print(s.shape)
                    gram_mat=tmps.bmm(tmps.transpose(1,2))
                    #print(s.shape)
                    gram_mat=gram_mat/(h*w)
                    self.gram.append(gram_mat)
                if saved_activation:
                    self.activation.append(s)
        return s
    def adjust_weight(self,rate:float,target_layer:int=-1):
        if target_layer==-1:
            # adjust all layers weight by (1-rate)
            for layer in self.layers:
                if isinstance(layer,nn.Conv2d):
                    fg=torch.rand(1)
                    if fg<0.5:
                        fg=-1
                    else:
                        fg=1
                    layer.weight.data*=(1-rate*fg)
                    if layer.bias is not None:
                        layer.bias.data*=(1-rate*fg)
        else:
            # adjust target layer weight by (1-rate)
            # needs to ensure target_layer is a Conv2d layer
            if not isinstance(self.layers[target_layer],nn.Conv2d):
                raise TypeError("target_layer must be a Conv2d layer")
            self.layers[target_layer].weight.data*=(1-rate)
            if self.layers[target_layer].bias is not None:
                self.layers[target_layer].bias.data*=(1-rate)
# use cifar10 to train the classifier
def train(PCA:PretrainedPCA)->torch.Tensor:
    dataset=datasets.CIFAR10(root="/root/autodl-tmp/data",train=True,download=True,transform=torchvision.transforms.ToTensor())
    dataloader=DataLoader(dataset,batch_size=128,shuffle=True)
    criterion=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(PCA.Classifier.parameters(),lr=1e-3)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.9)
    totalloss=0
    avgloss=0
    PCA.Classifier.train()
    for epoch in range(5):
        tbar=tqdm(range(len(dataloader)))
        tbar.set_description_str("Epoch:{}".format(epoch))
        totalloss=0
        for i,(data,target) in enumerate(dataloader):
            data,target=data.to(Params["device"]),target.to(Params["device"])
            optimizer.zero_grad()
            output=PCA.Classifier(data)
            loss=criterion(output,target)
            loss.backward()
            totalloss+=loss.item()
            avgloss=totalloss/(i+1)
            tbar.update(1)
            tbar.set_postfix_str("Loss:{}".format(avgloss))
            optimizer.step()
        scheduler.step()
#test the model on CIFAR10
def test(PCA:PretrainedPCA)->float:
    dataset=datasets.CIFAR10(root="/root/autodl-tmp/data",train=False,download=True,transform=torchvision.transforms.ToTensor())
    dataloader=DataLoader(dataset,batch_size=128,shuffle=True)
    PCA.Classifier.eval()
    correct=0
    total=0
    with torch.no_grad():
        for data,target in dataloader:
            data,target=data.to(Params["device"]),target.to(Params["device"])
            output=PCA.Classifier(data)
            _,predicted=torch.max(output.data,1)
            total+=target.size(0)
            correct+=(predicted==target).sum().item()
    return correct/total
#save the model
def save_model(PCA:PretrainedPCA):
    torch.save(PCA.Classifier.state_dict(),Params["classifierpath"])
#load the model
def load_model(PCA:PretrainedPCA):
    PCA.Classifier.load_state_dict(torch.load(Params["classifierpath"],map_location=Params["device"]))
    PCA.Classifier.to(Params["device"])
    PCA.Classifier.eval()


if __name__!="__main__":
    model=PretrainedPCA(Params["modelpath"],torch.randn(1,3,32,32))
    train(model)
    save_model(model)
    acc=test(model)
    print("Accuracy:{}".format(acc))
else:
    #load the model and test
    model=PretrainedPCA(Params["modelpath"],torch.randn(1,3,32,32))
    load_model(model)
    model.adjust_weight(0.2)
    acc=test(model)
    print("Accuracy:{}".format(acc))
