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
from typing import Union
from tqdm import tqdm
from Char_Enc import Encoder
from typing import Iterator
import sys


Params={
    "hiddensize":1024,
    "device":torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "datapath":"/root/autodl-tmp/data/txt/anaphora",
    "modelpath":"./model.pth",
    "traintime":10,
    "epoch":40,
    "LossPath":"./loss.png",
    "FlossPath":"./floss.png",
    "startsign":"#",
    "endsign":"\\",
    "Norm":1.
}
os.chdir(os.path.dirname(os.path.abspath(__file__)))
class nnet(nn.Module):
    def __init__(self,inputsize:int,hiddensize:int,startsign:int,endsign:int,device:torch.device) -> None:
        super().__init__()
        # net contains Wi,Wf,Wo,Wg
        self.startsign=startsign
        self.endsign=endsign
        self.device=device
        self.inputsize=inputsize
        self.hiddensize=hiddensize
        self.concatsize=hiddensize*2
        self.Enc=nn.Linear(inputsize,hiddensize)
        self.Wi=nn.Linear(self.concatsize,hiddensize)
        self.Wf=nn.Linear(self.concatsize,hiddensize)
        self.Wo=nn.Linear(self.concatsize,hiddensize)
        self.Wg=nn.Linear(self.concatsize,hiddensize)
        self.Dec=nn.Linear(hiddensize,inputsize)
        for m in self.modules():
            if isinstance(m,nn.Linear):
                init.xavier_uniform_(m.weight.data)
                init.zeros_(m.bias.data)
    def forward(self,s:List[torch.Tensor],h:torch.Tensor,c:torch.Tensor) -> Tuple[List[torch.Tensor],torch.Tensor,torch.Tensor]:
        # assert s starts with self.startsign and ends with self.endsign 
        lens=len(s)
        pi=[]
        h.requires_grad=True #Gradients pass through h 
        for i in range(lens):
            xt=self.Enc(s[i])
            Instate=torch.concat((h,xt))
            i=torch.sigmoid(self.Wi(Instate))
            f=torch.sigmoid(self.Wf(Instate))
            o=torch.sigmoid(self.Wo(Instate))
            g=torch.tanh(self.Wg(Instate))
            c=f*c+i*g
            #f:float(0:1) c:Tensor i:float(0:1) g:float(-1:1)
            h=o*torch.tanh(c)
            pi.append(self.Dec(h))
        return pi,h,c
    def predict(self,startvec:torch.Tensor,h:torch.Tensor,c:torch.Tensor) -> List[torch.Tensor]:
        # h:Tensor c:Tensor
        # return a tensor with size of self.inputsize
        select=self.inputsize
        xt=startvec
        cnt=0
        maxinum=200
        ans=[]
        while cnt<=maxinum:
            cnt+=1
            Instate=torch.concat((h,self.Enc(xt)))
            i=torch.sigmoid(self.Wi(Instate))
            f=torch.sigmoid(self.Wf(Instate))
            o=torch.sigmoid(self.Wo(Instate))
            g=torch.tanh(self.Wg(Instate))
            c=f*c+i*g
            #f:float(0:1) c:Tensor i:float(0:1) g:float(-1:1)
            h=o*torch.tanh(c)
            y=F.softmax(self.Dec(h),dim=0)
            select=int(torch.argmax(y).item())
            xt=torch.zeros(self.inputsize,device=self.device)
            xt[select]=1
            ans.append(xt)
        return ans
class LSTM():
    def __init__(self,hiddensize:int,device:torch.device) -> None:
        self.hiddensize=hiddensize
        self.Enc=Encoder(None,"~",".",device)
        self.inputsize=len(self.Enc.map)
        self.startsign=self.Enc.map[Params["startsign"]]
        self.endsign=self.Enc.map[Params["endsign"]]
        self.startvec=torch.zeros(self.inputsize,device=device)
        self.startvec[self.startsign]=1
        self.nnet=nnet(self.inputsize,self.hiddensize,self.startsign,self.endsign,device)
        self.nnet.to(device)
        self.device=device
        self.optimizer=torch.optim.Adam(self.nnet.parameters(),lr=5e-4)
        self.scheduler=torch.optim.lr_scheduler.ExponentialLR(self.optimizer,gamma=0.9)
        self.cnter=self.train_cnt()
    def loss(self,pi:List[torch.Tensor],s:List[torch.Tensor]) -> torch.Tensor:
        # pi:List[torch.Tensor] s:List[torch.Tensor]
        # return a tensor
        lens=len(s)
        loss=torch.Tensor([0.]).to(device=self.device)
        for i in range(lens):
            #print(pi[i].device)
            #print(s[i].device)
            loss+=F.cross_entropy(pi[i],s[i])
        return loss
    def train_cnt(self)->Iterator[int]:
        cnt=1
        while True:
            yield cnt
            cnt+=1
        return 0
    def train(self,data:Union[str,List[str]],epoch:int) -> Tuple[List[float],List[float]]:
        # data:str
        # epoch:int
        # train the net
        # data is a string which needs to split by "."
        # epoch is the number of epoch
        #context=data.split(".")[101:140] # remove the last empty string
        loss_history=[]
        floss_history=[]
        if isinstance(data,list):
            context=[self.Enc.encode(i) for i in data]
        else:
            context=self.Enc.encode(data)
        #print(context)
        self.nnet.train()
        #print(context[0])
        #print(len(context))
        with tqdm(total=epoch) as pbar:
            pbar.set_description("epoch {}/{}".format(self.cnter.__next__(),Params["traintime"]))
            for i in range(epoch):
                self.optimizer.zero_grad()
                #print("Input:{}".format(len(s)))
                seq_len=random.randint(min(len(context),10),max(len(context),100))
                if isinstance(data,list):#random select a context
                    context_slice=random.choice(context)
                    context_slice=context_slice[0:seq_len]
                else:#use the whole context
                    context_slice=context
                    context_slice=context_slice[0:seq_len]
                #print(len(context_slice))
                #assert(isinstance(context_slice,list))
                h=torch.zeros(self.hiddensize,device=self.device)
                c=torch.zeros(self.hiddensize,device=self.device)
                pi,h,c=self.nnet(context_slice,h,c)
                #print("Output:{}".format(len(pi)))
                loss=self.loss(pi[:-1],context_slice[1:])/len(context_slice)
                floss=F.l1_loss(self.nnet.Wf.weight,torch.zeros_like(self.nnet.Wf.weight))*Params["Norm"]
                floss+=F.l1_loss(self.nnet.Wi.weight,torch.zeros_like(self.nnet.Wi.weight))*Params["Norm"]
                floss+=F.l1_loss(self.nnet.Wo.weight,torch.zeros_like(self.nnet.Wo.weight))*Params["Norm"]
                floss+=F.l1_loss(self.nnet.Wg.weight,torch.zeros_like(self.nnet.Wg.weight))*Params["Norm"]
                loss=loss+floss
                #print("Loss:{}".format(loss))
                loss.backward()
                loss_history.append(loss.item())
                floss_history.append(floss.item())
                pbar.update(1)
                pbar.set_postfix(loss=loss.item(),lr=self.scheduler.get_last_lr()[-1])
                self.optimizer.step()
        self.scheduler.step()
        return loss_history,floss_history
    def predict(self,previous_message:str)->str:
        self.nnet.eval()
        previous_input=self.Enc.encode(previous_message,False)
        h=torch.zeros(self.hiddensize,device=self.device)
        c=torch.zeros(self.hiddensize,device=self.device)
        if len(previous_input)>1:
            pi,h,c=self.nnet(previous_input[:-1],h,c)
        ans=self.nnet.predict(previous_input[-1],h,c)
        ans=self.Enc.decode(ans)
        return previous_message+ans
def save_model(nnet:LSTM,path:str) -> None:
    # nnet:LSTM path:str
    # save the model
    torch.save(nnet.nnet.state_dict(),path)

def load_model(nnet:LSTM,path:str) -> None:
    # nnet:LSTM path:str
    # load the model
    nnet.nnet.load_state_dict(torch.load(path))
    nnet.nnet.eval()
def test_model(load:bool,save:bool)-> Tuple[LSTM,List[float],List[float]]:
    Single_Load=False
    if Single_Load:
        with open(Params["datapath"],'r',encoding='utf-8') as f:
            data=f.readlines()
    else:
        # walk through the directory Params["datapath"] and load all the files
        data=[]
        for root,dirs,files in os.walk(Params["datapath"]):
            #load the files
            for file in files:
                #load the files to data
                with open(os.path.join(root,file),'r',encoding='utf-8') as f:
                    data.append("".join(f.readlines()))
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    nnet=LSTM(Params["hiddensize"],device)
    if load:
        load_model(nnet,Params["modelpath"])
    loss_his=[]
    floss_his=[]
    for time in range(Params["traintime"]):
        loss_history,floss_history=nnet.train(data,Params["epoch"])
        [loss_his.append(i) for i in loss_history]
        [floss_his.append(i) for i in floss_history]
    if save:
        save_model(nnet,Params["modelpath"])
    return nnet,loss_his,floss_his
def load_LSTM(path:str) -> LSTM:
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    nnet=LSTM(Params["hiddensize"],device)
    load_model(nnet,path)
    return nnet
if __name__ == "__main__":
    if sys.argv[1]=='test':
        model=load_LSTM(Params["modelpath"])
        print(model.predict(input("predict for:")+'\n'))
    if sys.argv[1]=='train':
        Loaded=input("Load model?(y/n)")
        if Loaded=='y':
            load=True
        else:
            load=False
        model,loss_his,floss_his=test_model(load,True)
        # nnet,loss_history=test_model(True,False)
        plt.plot(range(len(loss_his)),loss_his,label="loss")
        plt.legend()
        plt.savefig(Params["LossPath"])
        plt.clf()
        plt.plot(range(len(floss_his)),floss_his,label="floss")
        plt.legend()
        plt.savefig(Params["FlossPath"])
        print(model.predict(""))
    
    
    