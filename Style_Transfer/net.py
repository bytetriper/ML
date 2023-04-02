import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import DataLoader
from torchvision import datasets
from matplotlib import pyplot as plt
from tqdm import tqdm
from typing import List,Iterator,Union,Tuple
from torchvision.transforms import ToTensor
import numpy as np
import sys
import pickle as pkl
#import ../ML/PCA net.py as PCA
sys.path.append("/root/autodl-tmp/")
from Headers.Nets import PCA
from Headers.Utils import ImgP
Params={
    "modelpath":"/root/autodl-tmp/ML/Models/PCA.pth",
    "device":torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "lr":8e-3,
    "Input_imgpath":'/root/autodl-tmp/ML/Style_Transfer/source.jpg',
    "Style_imgpath":'/root/autodl-tmp/ML/Style_Transfer/style.jpg',
    "Content_imgpath":'/root/autodl-tmp/ML/Style_Transfer/content.jpg',
    "Output_imgpath":'/root/autodl-tmp/ML/Style_Transfer/output.jpg',
    'losspath':'/root/autodl-tmp/ML/Style_Transfer/loss.pkl',
    "traintime":5000,
    "penalty":1e-4,
    "style_weight":1e3,
    "content_weight":1e-1,
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
        self.scheduler=torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=200,gamma=0.95)
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
    def forward_layer(self,target_layer:int)->torch.Tensor:
        s=self.input
        self.layer=self.name_dict["Down_Net."+str(target_layer)]
        for layer in self.PCA.Down_Net[:target_layer+1]:
            s=layer(s)
        #use negative loss to perform gradient ascent
        return -torch.square(s).sum()/s.numel()
    def forward(self,s:torch.Tensor=None,compute_gram:bool=False,saved_activation:bool=False)->torch.Tensor:
        if s==None:
            s=self.input
        self.gram=[]
        self.activation=[]
        for layer in self.PCA.Down_Net:
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
    def L2_Regularization(self,s:torch.Tensor)->torch.Tensor:
        return (torch.square(s)).mean()
    def gram_loss(self,gram:torch.Tensor,target_gram:torch.Tensor)->torch.Tensor:
        return F.mse_loss(gram,target_gram)
    def gram_train(self,target:torch.Tensor,target_layer:int, epochs:int=100)->Tuple[torch.Tensor,List[float]]:
        self.PCA.eval()
        #print the source image path
        print("Source Image:"+self.path)
        #run forward with self.input and set compute_gram to True
        self.forward(target,compute_gram=True)
        #save the gram matrix of target layer and detach it
        target_gram=self.gram[target_layer].detach()
        #record the loss
        loss_list=[]
        #record the average loss
        average_loss=0
        total_loss=0
        tbar=tqdm(range(epochs))
        tbar.set_description_str(f"Fitting layer_{target_layer:d}")
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            self.forward(compute_gram=True)
            loss=self.gram_loss(self.gram[target_layer],target_gram)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            total_loss+=loss.item()
            average_loss=total_loss/(epoch+1)
            tbar.update(1)
            tbar.set_postfix_str("Epoch: %d, Avgloss: %.6f"%(epoch,average_loss))
            #print("Epoch: %d, Activation: %.6f"%(epoch,-average_loss))
            loss_list.append(loss.item())
        return self.input,loss_list
    def feature_train(self,target:torch.Tensor,target_layer:int, epochs:int=100)->Tuple[torch.Tensor,List[float]]:
        self.PCA.eval()
        tbar=tqdm(range(epochs))
        #print the source image path
        print("Source Image:"+self.path)
        #run forward with self.input and set compute_gram to True
        self.forward(target,saved_activation=True)
        target_feature=self.activation[target_layer].detach()
        #record the loss
        loss_list=[]
        #record the average loss
        average_loss=0
        total_loss=0
        tbar.set_description_str(f"Fitting layer_{target_layer:d}")
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            self.forward(saved_activation=True)
            loss=self.gram_loss(self.activation[target_layer],target_feature)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            total_loss+=loss.item()
            average_loss=total_loss/(epoch+1)
            tbar.update(1)
            tbar.set_postfix_str("Epoch: %d, Avgloss: %.6f"%(epoch,average_loss))
            #print("Epoch: %d, Activation: %.6f"%(epoch,-average_loss))
            loss_list.append(loss.item())
        return self.input,loss_list 
    def integrated_train(self,gram_target:torch.Tensor,feature_target:torch.Tensor,gram_layer:int,feature_layer:int, epochs:int=100)->Tuple[torch.Tensor,List[float]]: 
        self.PCA.eval()
        tbar=tqdm(range(epochs))
        #print the source image path
        print("Source Image:"+self.path)
        #run forward with self.input and set compute_gram to True
        self.forward(gram_target,compute_gram=True)
        #save the gram matrix of target layer and detach it
        gram_target=self.gram[gram_layer].detach()
        self.forward(feature_target,saved_activation=True)
        feature_target=self.activation[feature_layer].detach()
        #record the loss
        loss_list=[]
        #record the average loss
        average_loss=0
        total_loss=0
        tbar.set_description_str(f"Fitting layer_{gram_layer:d} and layer_{feature_layer:d}")
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            self.forward(compute_gram=True,saved_activation=True)
            gram_loss=self.gram_loss(self.gram[gram_layer],gram_target)
            feature_loss=self.gram_loss(self.activation[feature_layer],feature_target)
            loss=Params["style_weight"] *gram_loss+feature_loss*Params["content_weight"]
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            total_loss+=loss.item()
            average_loss=total_loss/(epoch+1)
            tbar.update(1)
            tbar.set_postfix_str("Epoch: %d, Avgloss: %.6f"%(epoch,average_loss))
            #print("Epoch: %d, Activation: %.6f"%(epoch,-average_loss))
            loss_list.append(loss.item())
        return self.input,loss_list      
    def train(self,target_layer:int, epochs:int=100)->List[float]:#try to maximize the activation of target_layer
        self.PCA.eval()
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
            loss=self.forward_layer(target_layer)+Params["penalty"]*self.L2_Regularization(self.input)
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
def read_img(path:str,device:torch.device=Params["device"])->torch.Tensor:
    ### read image from path and return a tensor at the device
    #use plt to read the img from path in [0-1]
    img=plt.imread(path)
    img=img.transpose((2,0,1))/255.
    img=img[np.newaxis,:,:,:]
    img=torch.from_numpy(img.copy()).float().to(device)
    return img
def save_img(img:torch.Tensor, path:str):
    Saveimg=img.cpu().detach().numpy()
    Saveimg=Saveimg.transpose((0,2,3,1))
    Saveimg=Saveimg[0]
    print(Saveimg.max())
    print(Saveimg.min())
    #print(Saveimg)
    #Threshold=1.0
    #Saveimg[Saveimg>Threshold]=Threshold
    #print(Saveimg)
    Saveimg=Saveimg.clip(0,1)
    plt.imshow(Saveimg)
    plt.savefig(path)
if __name__=="__main__":
    Style_layer=int(input("Style Layer:"))
    Content_layer=int(input("Content Layer:"))
    #img=read_img(Params["Input_imgpath"])
    #img,_=ImgP.get_img("/root/autodl-tmp/data")
    #save_img(img,Params["Input_imgpath"])
    #read a content img and a style img using read_img
    Content_img=read_img(Params["Content_imgpath"])
    Style_img=read_img(Params["Style_imgpath"])
    x=torch.rand_like(Content_img,device=Params["device"])
    net=PretrainedPCA(Params["modelpath"],x)
    #use and print the loss
    output,loss=net.integrated_train(Style_img,Content_img,Style_layer,Content_layer,epochs=Params["traintime"])
    pkl.dump(loss,open(Params["losspath"],"wb"))
    save_img(output,Params["Output_imgpath"])

    
