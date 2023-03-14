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
inputsize=100
hiddensize=40
num_layers=2
Available_Charset="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890!@#$%^&*()_+-=;'\":.,<>?/\\|~` \n"
map={}
for i in range(len(Available_Charset)):
    map[Available_Charset[i]]=i
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
            if self.reversed_map[y.item()]==self.endsign:
                break
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
        self.Interval=200# Only backward [Interval] steps
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
            #if i%self.Interval==0:
            #    self.t.detach()
        return self.t,output
    def predict(self,initialState:torch.Tensor,x:torch.Tensor)->List[torch.Tensor]:
        self.t=initialState
        select=None
        ans=[]
        cnt=0
        maximum=100
        while True:
            with torch.no_grad():
                self.t=self.Tanh(self.Wx(x)+self.Wh(self.t))
                y=self.F(self.Wy(self.t))
                select=torch.argmax(y)
                x=torch.zeros_like(y)
                x[0][select.item()]=1
                ans.append(select)
                cnt+=1
                if cnt > maximum:
                    break
        return ans
def Ones(size,pos,device)->torch.Tensor:
    t=torch.zeros(1,size,device=device)
    t[0][pos]=1
    return t
class NNET_Wrapper():
    def __init__(self,inputsize:int,hiddensize:int,map:Dict[str,int],startsign:str,endsign:str,device:torch.device,pretrained=None) -> None:
        self.nnet=RNN(inputsize,hiddensize,startsign,endsign,device)
        if pretrained:
            self.nnet.load_state_dict(pretrained)
        self.optimizer=torch.optim.Adam(self.nnet.parameters(),lr=1e-5)
        #self.scheduler=torch.optim.lr_scheduler.ExponentialLR(self.optimizer,0.997)
        self.device=device
        self.map=map
        self.startsign=startsign
        self.endsign=endsign
        self.separatesign=map[' ']
        self.seperateVec=torch.zeros(1,inputsize).to(device)
        #self.seperateVec[0][self.separatesign]=1
        self.startVec=torch.zeros(1,inputsize).to(device)
        self.startVec[0][self.map[self.startsign]]=1
        self.endVec=torch.zeros(1,inputsize).to(device)
        self.endVec[0][self.map[self.endsign]]=1
        self.nnet.to(self.device)
        self.Encoder=Encoder(self.map,endsign)
    def lossF(self,input:List[torch.Tensor],label)->torch.Tensor:
        lossF=nn.KLDivLoss(reduction="batchmean")
        loss=torch.Tensor([0]).to(self.device)
        seq_len=len(input)
        for i in range(seq_len):
            loss+=lossF(torch.log(input[i]),label[i])
        return loss
    def train(self,data:str)->torch.Tensor:
        self.nnet.train()
        data=torch.cat((self.startVec,self.Encoder.encode(data,self.device),self.endVec))
        h,output=self.nnet(data)
        loss=self.lossF(output[0:-1],data[1:])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        #self.scheduler.step()
        return loss
    def predict(self,prefix:str)->str:
        self.nnet.eval()
        if prefix==None:
            prefix=''
        prefix=self.startsign+prefix
        lastChar=prefix[-1]
        CharVec=Ones(self.nnet.inputsize,self.map[lastChar],self.device)
        if len(prefix)>1:
            Input=prefix[0:-1]
            Input=self.Encoder.encode(Input,self.device)
            h,_=self.nnet(Input)
            output=self.nnet.predict(h,CharVec)
            output=self.Encoder.decode(output)
        else:
            h=torch.zeros(self.nnet.hiddensize).to(self.device)
            output=self.nnet.predict(h,CharVec)
            output=self.Encoder.decode(output)
        return prefix+output
    ###FUCK! I'll Come Back!
Params={
    'lr':1e-6,
    'hiddensize':256,
    'train_epoch':512,
    'batch_epoch':16,
    'batch_size':12,
    'inputsize':len(map),
    'map':map,
    'startsign':'~',
    'endsign':'`',
    'device':torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}
def model_train(nnet:NNET_Wrapper,datasource:str,LossPath:str,Savepath:str):
    assert(os.path.exists(datasource))
    endsign=';'
    startsign='*'
    display_interval=4
    with open(datasource,"r",encoding='utf8') as f:
        txt=f.readlines()[3:]
    content=''
    for t in txt:
        content+=t
    loss_history=[]
    content=content.split(',')
    #print(content)
    content_len=len(content)
    for i in range(Params['train_epoch']):
        st=random.randint(0,content_len-Params['batch_size'])
        end=st+Params['batch_size']
        data=''
        for j in range(st,end):
            data+=content[j]
        #print(data)
        #print(data)
        if len(data)<10:
            continue
        for epoch in range(Params['batch_epoch']):
            loss=nnet.train(data)/len(data)
            if epoch%display_interval==0:
                print(f"train_epoch:{i:d} sub_epoch:{epoch:d} Loss:{loss.item():.5f}")
            loss_history.append(loss.item())
        save_model(nnet,os.path.join(Savepath,f"model{i:d}.pth"))
    plt.plot(range(len(loss_history)),loss_history)
    plt.savefig(os.path.join(LossPath,"Loss.png"))
    #save_model(nnet,SavePath)
    return
def create_model()->NNET_Wrapper:
    model=NNET_Wrapper(Params['inputsize'],Params['hiddensize'],Params['map'],Params['startsign'],Params['endsign'],Params['device'],None)
    return model
def save_model(model:NNET_Wrapper,savepath:str)->None:
    torch.save({'model':model.nnet.state_dict()},savepath)
def load_model(modelpath:str)->NNET_Wrapper:
    state_dict=torch.load(modelpath)['model']
    model=NNET_Wrapper(Params['inputsize'],Params['hiddensize'],Params['map'],Params['startsign'],Params['endsign'],Params['device'],state_dict)
    return model
if __name__=="__main__":
    model=load_model(r'/root/ML/RNN/Model_final.pth')
    #model=create_model()
    #model_train(model,"/root/ML/RNN/data/data.txt","/root/ML/RNN","/root/ML/RNN/Model")
    #save_model(model,"/root/ML/RNN/Model_final.pth")
    print(model.predict(""))