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
from typing import Union
from torchvision.transforms import ToTensor,RandomHorizontalFlip,RandomAdjustSharpness,RandomVerticalFlip,Compose
import numpy as np
import sys
import PIL.Image as Image
Params={
    "hiddensize":128,
    "device":torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "datapath":"/root/autodl-tmp/data",
    "modelpath":"./model.pth",
    "classifierpath":"./classifier.pth",
    "traintime":10,
    "epoch":40,
    "LossPath":"./loss.png",
    "ClassifylossPath":"./floss.png",
    "batchsize":512,
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
            #nn.Dropout2d(0.2),
            nn.Conv2d(32,64,7,1,3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #nn.Dropout2d(0.2),
            nn.Conv2d(64,64,5,2,2), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #nn.Dropout2d(0.2),
            nn.Conv2d(64,64,5,1,2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #nn.Dropout2d(0.2),
            nn.Conv2d(64,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #nn.Dropout2d(0.2),
            nn.Conv2d(64,64,5,2,2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #nn.Dropout2d(0.2),
            nn.Conv2d(64,hiddensize,5,2,2),
            nn.BatchNorm2d(hiddensize),
            nn.ReLU(),
        )
        #self.fc_d=nn.Linear(64*4*4,hiddensize)# maybe be removed in the future
        self.Up_Net=nn.Sequential(
            nn.ConvTranspose2d(hiddensize,64,kernel_size=5,stride=2,padding=2,output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64,64,kernel_size=5,stride=2,padding=2,output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,64,5,1,2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64,64,kernel_size=5,stride=2,padding=2,output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,64,7,1,3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #nn.ConvTranspose2d(64,64,kernel_size=5,stride=2,padding=2,output_padding=1),
            #nn.BatchNorm2d(64),
            #nn.ReLU()
        )
        #self.BN=nn.BatchNorm2d(64)
        self.Conv2one=nn.Conv2d(64,input_channel,7,1,3)
    def forward(self,s):
        #print(s.shape)
        s=self.Down_Net(s)
        #print(s.shape)
        s=self.Up_Net(s)
        #print(s.shape)
        s=self.Conv2one(s)
        return s
class Mini_Classifier(nn.Module):
    def __init__(self,Feature_Extractor:nn.Sequential) -> None:
        super(Mini_Classifier,self).__init__()
        self.Feature_Extractor=Feature_Extractor
        # save the parameters of the feature extractor
        self.saved_Params=nn.ParameterList([self.Feature_Extractor])
        self.fc=nn.Linear(2048,2048)
        self.activation=nn.ReLU()
        self.fc2=nn.Linear(2048,10)
    def forward(self,s):
        s=self.Feature_Extractor(s)
        # to detach and flatten the tensor(prevent gradient flow back to the feature extractor)
        
        s=s.view(s.size(0),-1)
        #s=s.detach()
        s=self.fc(s)
        s=self.activation(s)
        s=self.fc2(s)
        return s
class NNET():
    def __init__(self,inputsize:int,hiddensize:int,device:torch.device) -> None:
        self.hiddensize=hiddensize
        self.inputsize=inputsize
        self.device=device
        self.net=PCA(inputsize,hiddensize)
        self.net.to(self.device)
        self.optimizer=torch.optim.Adam(self.net.parameters(),lr=8e-5)
        self.scheduler=torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=300,gamma=0.9)
        self.cnter=self.train_cnt()
    def train_cnt(self)->Iterator[int]:
        cnt=1
        while True:
            yield cnt
            cnt+=1
    def train(self,datalist:DataLoader,epoch:int) -> List[float]:
        self.net.train()
        loss_his=[]
        # keep track of average loss during training
        total_loss=0
        avg_loss=0
        for i in range(epoch):
            bar=tqdm(range(1,len(datalist)))
            bar.set_description_str("Epoch:{}".format(i))
            total_loss=0
            for batch_idx,(data,label) in enumerate(datalist):
                data,label=data.to(self.device),label.to(self.device)
                self.optimizer.zero_grad()
                output=self.net(data)
                loss=F.l1_loss(output,data)
                loss.backward()
                loss_his.append(min(loss.item(),2.))
                total_loss+=loss.item()
                avg_loss=total_loss/(batch_idx+1)
                self.optimizer.step()
                bar.update(1)
                bar.set_postfix_str('SubEpoch: {} [{}/{} ({:.0f}%)] Loss: {:.3f}| lr: {:.3e}'.format(
                        batch_idx, batch_idx, len(datalist),
                        100.*batch_idx/len(datalist), avg_loss,self.scheduler.get_last_lr()[0]))
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
class Classifier_Model():
    def __init__(self,model:Union[NNET,Mini_Classifier],device:torch.device) -> None:
        if isinstance(model,NNET):
            self.model=Mini_Classifier(model.net.Down_Net)
        elif isinstance(model,Mini_Classifier):
            self.model=model
        else:
            raise TypeError("model must be NNET or Mini_Classifier")
        self.device=device
        # may encounter error when using different device on Down_net and Mini_Classifier
        self.model.to(device)
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=8e-3)
        self.scheduler=torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=300,gamma=0.9)
    def train(self,datalist:DataLoader,epoch:int) -> List[float]:
        self.model.train()
        loss_his=[]
        # keep track of average loss during training
        total_loss=0
        avg_loss=0
        for i in range(epoch):
            bar=tqdm(range(len(datalist)))
            bar.set_description_str("Epoch:{}".format(i))
            total_loss=0
            for batch_idx,(data,label) in enumerate(datalist):
                data,label=data.to(self.device),label.to(self.device)
                self.optimizer.zero_grad()
                output=self.model(data)
                loss=F.cross_entropy(output,label)
                loss.backward()
                loss_his.append(loss.item())
                total_loss+=loss.item()
                avg_loss=total_loss/(batch_idx+1)
                self.optimizer.step()
                bar.update(1)
                bar.set_postfix_str('SubEpoch: {} [{}/{} ({:.0f}%)] Loss: {:.3f}| lr: {:.3e}'.format(
                        batch_idx, batch_idx, len(datalist),
                        100.*batch_idx/len(datalist), avg_loss,self.scheduler.get_last_lr()[0]))
                self.scheduler.step()
        return loss_his
    def predict(self,data:torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            data=data.to(self.device)
            output=self.model(data)
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

def train_model(load:bool=False,save:bool=False)->NNET:
    torch.cuda.empty_cache()
    Dataset=datasets.CIFAR10(root=Params["datapath"],train=True,download=False,transform=Compose([
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomAdjustSharpness(0.1),
        ToTensor()
    ])
    )
    trainloader=DataLoader(Dataset,batch_size=Params["batchsize"],shuffle=True)
    device = Params["device"]
    net=create_model(3,128,device)
    if load:
        load_model(net.net,Params["modelpath"])
    loss=net.train(trainloader,Params["epoch"])
    plt.plot(range(len(loss)),loss)
    plt.savefig(Params["LossPath"])
    plt.clf()
    if save:
        save_model(net.net,Params["modelpath"])
    return net
#create a classifier model using the pretrained NNET model
def create_classifier(net:NNET,device:torch.device)->Classifier_Model:
    return Classifier_Model(net,device)
#load a classifier model for testing
def load_classifier(path:str,device:torch.device)->Classifier_Model:
    tmpmodel=create_model(3,Params["hiddensize"],device)
    model=Mini_Classifier(tmpmodel.net.Down_Net)
    #print(torch.load(path))
    model.load_state_dict(torch.load(path))
    classifier=Classifier_Model(model,device)
    return classifier
#test the accuracy of Classifier_Model on cifar10 testset, and return the accuracy
def test_classfier(classifier:Classifier_Model,save:bool=False)->float:
    torch.cuda.empty_cache()
    Dataset=datasets.CIFAR10(root=Params["datapath"],train=False,download=False,transform=ToTensor())
    testloader=DataLoader(Dataset,batch_size=Params["batchsize"],shuffle=True)
    correct=0
    total=0
    with torch.no_grad():
        for data,label in testloader:
            data,label=data.to(Params["device"]),label.to(Params["device"])
            output=classifier.predict(data)
            _,predicted=torch.max(output.data,1)
            total+=label.size(0)
            correct+=(predicted==label).sum().item()
    acc=correct/total
    if save:
        save_model(classifier.model,Params["classifierpath"])
    return acc
rand_idx=np.random.choice(50000,2000)
def train_classifier(classifier:Classifier_Model,save:bool=False)->None:
    Dataset=datasets.CIFAR10(root=Params["datapath"],train=True,download=False,transform=Compose([
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomAdjustSharpness(0.1),
        ToTensor()
    ]))
    #random choice 5000 idx from cifar10 trainset
    #Dataset.data=[Dataset.data[i] for i in rand_idx]
    #Dataset.targets=[Dataset.targets[i] for i in rand_idx]
    trainloader=DataLoader(Dataset,batch_size=Params["batchsize"],shuffle=True)
    loss=classifier.train(trainloader,Params["epoch"])
    plt.plot(range(len(loss)),loss)
    plt.savefig(Params["ClassifylossPath"])
    plt.clf()
    if save:
        torch.save(classifier.model.state_dict(),Params["classifierpath"])
    return
#show a img in CIFAR10 and prediction from a NNET model
def show_img(net:NNET,dataset:datasets.CIFAR10,imgpath:str=''):
    # show a random img in CIFAR10
    img,label=dataset[np.random.randint(0,len(dataset))]
    #otherwise show the img in imgpath as a numpy array
    if imgpath!='':
        img=Image.open(imgpath)
        img=ToTensor()(img)
        print(img)
    Inputimg=img.unsqueeze(0)
    Inputimg=Inputimg.to(Params["device"])
    #print(Inputimg)
    output=net.predict(Inputimg)
    output=output.squeeze(0)
    output=output.cpu()
    output=output.detach().numpy()
    #output=(output+1)/2
    print(np.abs(output-img.numpy()).mean())
    output=np.transpose(output,(1,2,0))
    plt.imshow(output)
    plt.savefig(Params["imgpath"])
    plt.clf()
    plt.imshow(img.numpy().transpose(1,2,0))
    plt.savefig(Params["imgpath_origin"])
    plt.clf()
    return label
if __name__ != "__main__":
    # read from argv[1] to decide whether to train or not(if "train" then train,otherwise test only)
    if len(sys.argv)>1:
        #if argv[2] is "PCA" then train NNET, otherwise train Classifier_Model
        #if sys.argv[2]=="PCA", then use sys.argv[3] to decide whether to load model or to train
        #if sys.argv[2]=="classify",then load a NNET from modelpath to init a classifier,and train it using train_classifier
        if sys.argv[1]=="train":
            if sys.argv[2]=="PCA":
                if sys.argv[3]=="load":
                    net=train_model(True,True)
                else:
                    net=train_model(False,True)
                # use show_img to show a img in CIFAR10 and prediction from a NNET model
                Dataset=datasets.CIFAR10(root=Params["datapath"],train=True,download=False,transform=ToTensor())
                label=show_img(net,Dataset)
            elif sys.argv[2]=="classify":
                #use sys.argv[3] to decide whether to load a classifier or to create a new one
                if sys.argv[3]=="load":
                    classifier=load_classifier(Params["classifierpath"],Params["device"])
                elif sys.argv[3]=="new":
                    net=create_model(3,128,Params["device"])
                    load_model(net.net,Params["modelpath"])
                    classifier=create_classifier(net,Params["device"])
                else:
                    raise KeyError("please input a valid command")
                train_classifier(classifier,True)
            #otherwise raise a keyerror
            else:
                raise KeyError("please input a valid command")
        elif sys.argv[1]=="test":
            if sys.argv[2]=="PCA":
                net=create_model(3,Params["hiddensize"],Params["device"])
                load_model(net.net,Params["modelpath"])
                Dataset=datasets.CIFAR10(root=Params["datapath"],train=True,download=False,transform=ToTensor())
                label=show_img(net,Dataset)
            elif sys.argv[2]=="classify":
                # if sys.argv[3]=="load", then load a classifier from classifierpath and test it,otherwise create a classifier and test it
                if sys.argv[3]=="load":
                    classifier=load_classifier(Params["classifierpath"],Params["device"])
                elif sys.argv[3]=="new":
                    net=create_model(3,Params["hiddensize"],Params["device"])
                    load_model(net.net,Params["modelpath"])
                    classifier=create_classifier(net,Params["device"])
                else:
                    raise KeyError("please input a valid command")
                #test the classfier and print the accuracy in a pretty way
                acc=test_classfier(classifier)
                print("the accuracy of the classifier is: {:.2f}%".format(acc*100))
            else:
                raise KeyError("please input a valid command")
        else:
            raise KeyError("please input a valid command")
else:
    print(torch.load(Params["classifierpath"]).keys())

