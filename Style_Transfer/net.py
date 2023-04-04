import torch
import torch.nn.functional as F
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from typing import List,Iterator,Union,Tuple
import numpy as np
import pickle as pkl
import torchvision
from torchvision.models.vgg import VGG16_Weights
import cv2
Params={
    "modelpath":"/root/autodl-tmp/ML/Models/PCA.pth",
    "classifierpath":"/root/autodl-tmp/ML/Models/Mini_Classifier.pth",
    "device":torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "lr":5e-3,
    "Input_imgpath":'/root/autodl-tmp/ML/Style_Transfer/source.jpg',
    "Style_imgpath":'/root/autodl-tmp/ML/Style_Transfer/imgs/night.jpg',
    "Content_imgpath":'/root/autodl-tmp/ML/Style_Transfer/imgs/R-C.jpg',
    "Output_imgpath":'/root/autodl-tmp/ML/Style_Transfer/output.jpg',
    'losspath':'/root/autodl-tmp/ML/Style_Transfer/loss.pkl',
    "traintime":1000,
    "penalty":.1,
    "style_weight":1.,
    "content_weight":1.6,
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
                self.layers.append(layer)
            else:
                [self.layers.append(sublayer) for sublayer in layer.children()]
    def forward_layer(self,target_layer:int)->torch.Tensor:
        s=self.input
        self.layer=self.name_dict["Down_Net."+str(target_layer)]
        for layer in self.layers:
            s=layer(s)
        #use negative loss to perform gradient ascent
        return -torch.square(s).sum()/s.numel()
    def Weight(self,index:int)->float:
        return .95**(index)
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
    def L2_Regularization(self,s:torch.Tensor)->torch.Tensor:
        return (torch.square(s)).mean()
    def Feature_Regularization(self,s:torch.Tensor)->torch.Tensor:
        #print(s.shape)
        return (torch.square(s[:,1:]-s[:,:-1])).mean()
    def gram_loss(self,gram:Union[torch.Tensor,List[torch.Tensor]],target_gram:Union[torch.Tensor,List[torch.Tensor]])->torch.Tensor:
        if isinstance(gram,list):
            return sum([F.l1_loss(g,tg)*self.Weight(i) for i,(g,tg) in enumerate(zip(gram,target_gram))])
        elif isinstance(gram,torch.Tensor):
            return F.l1_loss(gram,target_gram)
        raise TypeError("gram must be a list or a tensor")
    def gram_train(self,target:torch.Tensor, epochs:int=100)->Tuple[torch.Tensor,List[float]]:
        self.Classifier.eval()
        #print the source image path
        print("Source Image:"+self.path)
        #run forward with self.input and set compute_gram to True
        self.forward(target,compute_gram=True)
        #save the gram matrix of target  layer and detach it
        target_gram=[layer.detach() for layer in self.gram]
        #record the loss
        loss_list=[]
        #record the average loss
        average_loss=0
        total_loss=0
        tbar=tqdm(range(epochs))
        tbar.set_description_str(f"Gram Train:")
        #self.input=target
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            self.forward(compute_gram=True)
            loss=.1*self.Feature_Regularization(self.input)
            loss+=self.gram_loss(self.gram,target_gram)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            total_loss+=loss.item()
            average_loss=total_loss/(epoch+1)
            tbar.update(1)
            tbar.set_postfix_str("Epoch: %d, Avgloss: %.6f,Lr:%.3e"%(epoch,average_loss,self.scheduler.get_last_lr()[0]))
            #print("Epoch: %d, Activation: %.6f"%(epoch,-average_loss))
            loss_list.append(loss.item())
        return self.input,loss_list
    def feature_train(self,target:torch.Tensor,target_layer:int, epochs:int=100)->Tuple[torch.Tensor,List[float]]:
        self.Classifier.eval()
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
            loss=self.gram_loss(self.activation[target_layer],target_feature)+.1*self.Feature_Regularization(self.input)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            total_loss+=loss.item()
            average_loss=total_loss/(epoch+1)
            tbar.update(1)
            tbar.set_postfix_str("Epoch: %d, Avgloss: %.6f,Lr:%.3e"%(epoch,average_loss,self.scheduler.get_last_lr()[0]))
            #print("Epoch: %d, Activation: %.6f"%(epoch,-average_loss))
            loss_list.append(loss.item())
        return self.input,loss_list 
    def integrated_train(self,gram_target:torch.Tensor,feature_target:torch.Tensor,feature_layer:int, epochs:int=100)->Tuple[torch.Tensor,List[float]]: 
        self.Classifier.eval()
        tbar=tqdm(range(epochs))
        #print the source image path
        print("Source Image:"+self.path)
        #run forward with self.input and set compute_gram to True
        self.forward(gram_target,compute_gram=True)
        #save the gram matrix of target layer and detach it
        gram_target=[layer.detach() for layer in self.gram]
        self.forward(feature_target,saved_activation=True)
        feature_target=self.activation[feature_layer].detach()
        #record the loss
        loss_list=[]
        #record the average loss
        average_loss=0
        total_loss=0
        tbar.set_description_str(f"Fitting layer_{feature_layer:d}")
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            self.forward(compute_gram=True,saved_activation=True)
            gram_loss=self.gram_loss(self.gram,gram_target)
            feature_loss=self.gram_loss(self.activation[feature_layer],feature_target)
            loss=Params["style_weight"] *gram_loss+feature_loss*Params["content_weight"]+Params["penalty"]*self.Feature_Regularization(self.input)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            total_loss+=loss.item()
            average_loss=total_loss/(epoch+1)
            tbar.update(1)
            tbar.set_postfix_str("Epoch: %d, Avgloss: %.6f,Lr:%.3e"%(epoch,average_loss,self.scheduler.get_last_lr()[0]))
            #print("Epoch: %d, Activation: %.6f"%(epoch,-average_loss))
            loss_list.append(loss.item())
        return self.input,loss_list      
    def train(self,target_layer:int, epochs:int=100)->List[float]:#try to maximize the activation of target_layer
        self.Classifier.eval()
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
# use cv2 to replace read_img with a new function with same functionality
def read_img_cv2(path:str,device:torch.device=Params["device"])->torch.Tensor:
    ### read image from path and return a tensor at the device
    #use cv2 to read the img from path in [0-1]
    #remember that cv2 reads in BGR and needs to be converted to RGB
    img=cv2.imread(path)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img=img.transpose((2,0,1))/255.
    img=img[np.newaxis,:,:,:]
    img=torch.from_numpy(img.copy()).float().to(device)
    return img
# use cv2 to replace save_img with a new function with same functionality
def save_img_cv2(img:torch.Tensor, path:str):
    #remember that function takes a RGB img and needs to be converted to BGR
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
    #brighten the image
    #Saveimg+=0.5
    Saveimg=Saveimg*255
    Saveimg=Saveimg.astype(np.uint8)
    Saveimg=cv2.cvtColor(Saveimg,cv2.COLOR_RGB2BGR)
    cv2.imwrite(path,Saveimg)

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
    #brighten the image
    #Saveimg+=0.5
    plt.imshow(Saveimg)
    plt.savefig(path)
if __name__=="__main__":
    Content_layer=int(input("Content Layer:"))
    Content_img=read_img_cv2(Params["Content_imgpath"])
    Style_img=read_img_cv2(Params["Style_imgpath"])
    x=torch.rand_like(Content_img,device=Params["device"])
    #generate a 2k rand x
    #x=torch.rand(1,3,256,256,device=Params["device"])
    net=PretrainedPCA(Params["classifierpath"],x)
    #use and print the loss
    #output,loss=net.feature_train(Content_img,Content_layer,epochs=Params["traintime"])
    #output,loss=net.gram_train(Style_img,epochs=Params["traintime"])
    output,loss=net.integrated_train(Style_img,Content_img,Content_layer,epochs=Params["traintime"])
    pkl.dump(loss,open(Params["losspath"],"wb"))
    save_img_cv2(output,Params["Output_imgpath"])

    
