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
from typing import Union
from torchvision.transforms import ToTensor
import numpy as np
import sys
import pickle as pkl
#import ../ML/PCA net.py as PCA
sys.path.append("/root/autodl-tmp/")
from Headers.Nets import PCA
from Headers.Utils import ImgP
Params={
    "modelpath":"/root/autodl-tmp/ML/PCA/model.pth",
    "device":torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "lr":1e-3,
    "Input_imgpath":'/root/autodl-tmp/ML/Style_Transfer/source.jpg',
    "Output_imgpath":'/root/autodl-tmp/ML/Style_Transfer/output.jpg',
    'losspath':'/root/autodl-tmp/ML/Style_Transfer/loss.pkl'
}
class PretrainedPCA(nn.Module):
    def __init__(self, path:str, input:torch.Tensor):
        super(PretrainedPCA, self).__init__()
        self.path = path
        self.load_pca()
        self.name_dict=dict(self.PCA.named_modules())
        self.input=input.to(Params["device"])
        self.input.requires_grad=True
        #only perform gradient ascent on input
        self.optimizer=torch.optim.Adam([self.input],lr=Params["lr"])
        self.scheduler=torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=300,gamma=0.9)
        self.tcnt=self.train_cnt()
    #def a train_cnt that returns a Iterator to record the train time
    def train_cnt(self)->Iterator:
        cnt=0
        while True:
            yield cnt
            cnt+=1
    def test_run(self)->torch.Tensor:
        self.PCA.eval()
        x=self.PCA(self.input)
        loss=F.l1_loss(x,self.input)
        return loss
    def load_pca(self):
        self.PCA=PCA.PCA(3,PCA.Params["hiddensize"])
        self.PCA.to(Params["device"])
        self.PCA.load_state_dict(torch.load(self.path,map_location=Params["device"]))
    def forward(self,target_layer:int)->torch.Tensor:
        s=self.input
        self.layer=self.name_dict["Down_Net."+str(target_layer)]
        for layer in self.PCA.Down_Net[:target_layer+1]:
            s=layer(s)
        #use negative loss to perform gradient ascent
        return -torch.square(s).sum()/s.numel()
    def train(self,target_layer:int, epochs:int=100)->List[float]:
        self.PCA.train()
        tbar=tqdm(range(epochs))
        #print the source image path
        print("Source Image:"+self.path)
        #record the loss
        loss_list=[]
        #record the average loss
        average_loss=0
        total_loss=0
        tbar.set_description_str(f"Fitting layer_{target_layer:d}")
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            loss=self.forward(target_layer)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            total_loss+=loss.item()
            average_loss=total_loss/(epoch+1)
            tbar.update(1)
            tbar.set_description_str("Epoch: %d, Activation: %.6f"%(epoch,-average_loss))
            #print("Epoch: %d, Activation: %.6f"%(epoch,-average_loss))
            loss_list.append(-loss.item())
        return loss_list
def read_img(path:str)->torch.Tensor:
    #use plt to read the img from path in [0-1]
    img=plt.imread(path)
    img=img.transpose((2,0,1))/255.
    #print(img)
    img=img[np.newaxis,:,:,:]
    img=torch.from_numpy(img.copy()).float()
    return img
def save_img(img:torch.Tensor, path:str):
    Saveimg=img.cpu().detach().numpy()
    Saveimg=Saveimg.transpose((0,2,3,1))
    Saveimg=Saveimg[0]
    print(Saveimg.max())
    print(Saveimg.min())
    #print(Saveimg)
    #Threshold=1.
    #Saveimg[Saveimg>Threshold]=Threshold
    #print(Saveimg)
    Saveimg=Saveimg.clip(0,1)
    plt.imshow(Saveimg)
    plt.savefig(path)
if __name__=="__main__":
    layer=int(input("Layer:"))
    img=read_img(Params["Input_imgpath"])
    #img,_=ImgP.get_img("/root/autodl-tmp/data")
    #save_img(img,Params["Input_imgpath"])
    net=PretrainedPCA(Params["modelpath"],img)
    #use test_run and print the loss
    loss=net.test_run()
    print("Predict Loss:",loss.item())
    loss=net.train(layer,500)
    print(net.layer)
    pkl.dump(loss,open(Params["losspath"],"wb"))
    save_img(net.input,Params["Output_imgpath"])

    
