import torch
import torch.nn.functional as F
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from typing import List,Iterator,Union,Tuple,Optional
import random
import string
import os
import tiktoken
from Char_Enc import Encoder
Params={
    "device":torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "lr":3e-4,
    "traintime":5,
    "batchsize":1,
    "modelpath":"/root/autodl-tmp/ML/Models/Transformer.pth",
    "datasize":48,
    "datapath":"/root/autodl-tmp/data/poems/abc"
}
class Test_Net(nn.Module):
    def __init__(self,enc:tiktoken.Encoding,device:torch.device=Params["device"]):
        super(Test_Net, self).__init__()
        self.transformer=nn.Transformer(num_encoder_layers=6,num_decoder_layers=6,batch_first=True)
        self.enc=enc
        self.EncW=nn.Linear(self.enc.n_vocab+1,self.transformer.d_model)
        self.DecW=nn.Linear(self.transformer.d_model,self.enc.n_vocab)
        self.maximum_length=100
        self.device=device
        self.end_selection=self.enc._special_tokens["<im_end>"]
        for p in self.parameters():
            if p.dim()>1:
                nn.init.xavier_uniform_(p)
    def make_temporal_signal(self, x:torch.Tensor,st_time:int=0)->torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        position = torch.arange(st_time,st_time+seq_len, device=x.device).unsqueeze(0).unsqueeze(-1)
        position=position.repeat(batch_size,1,1)
        return position
    def forward(self, x:torch.Tensor,y:torch.Tensor)->torch.Tensor:
        x=self.EncW(torch.cat([x,self.make_temporal_signal(x)],dim=-1))
        y=self.EncW(torch.cat([y,self.make_temporal_signal(y)],dim=-1))
        output=self.transformer(x,y)
        output=self.DecW(output)
        return output
    def predict(self, x:torch.Tensor,prompt:torch.Tensor)->torch.Tensor:
        x=torch.cat([x,self.make_temporal_signal(x)],dim=-1)
        x=self.EncW(x)
        ans=prompt
        prompt=self.EncW(torch.cat([prompt,self.make_temporal_signal(prompt)],dim=-1))
        output=self.transformer(x,prompt)
        for i in range(self.maximum_length):
            predict_token=output[:,-1,:]
            predict_token=self.DecW(predict_token)
            selected=torch.argmax(predict_token,dim=-1).item()
            if selected==self.end_selection:
                break
            next_token=torch.zeros_like(predict_token).unsqueeze(1)
            next_token[0,0,selected]=1
            ans=torch.cat([ans,next_token],dim=1)
            next_token=torch.cat([next_token,self.make_temporal_signal(next_token,len(ans[0,:,0])-1)],dim=-1)
            next_token=self.EncW(next_token)
            prompt=torch.cat([prompt,next_token],dim=1)
            output=self.transformer(x,prompt)
        return ans

class Wrapper:
    def __init__(self, net:Optional[Test_Net]=None, enc:tiktoken.Encoding=tiktoken.get_encoding('p50k_base'),device:torch.device=Params["device"]):
        if net!=None:
            self.net=net
        else:
            self.net=Test_Net(enc)
        self.net.to(device)
        self.enc=enc
        self.device=device
        self.start_token=torch.zeros(1,1,self.enc.n_vocab,device=device)
        self.end_token=torch.zeros(1,1,self.enc.n_vocab,device=device)
        self.start_token[0,0,self.enc._special_tokens["<im_start>"]]=1
        self.end_token[0,0,self.enc._special_tokens["<im_end>"]]=1
        self.end_selection=self.enc._special_tokens["<im_end>"]
        self.optimizer=torch.optim.Adam(self.net.parameters(),Params["lr"])
        self.scheduler=torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=200,gamma=0.97)
        self.batchifier=self.batchify
    def loss(self, x:torch.Tensor,y:torch.Tensor)->torch.Tensor:
        loss=nn.CrossEntropyLoss()
        return loss(x,y)
    def encode(self, s:str)->Optional[torch.Tensor]:
        seq=self.enc.encode(s)
        if len(seq)>0:
            seq=F.one_hot(torch.tensor(seq),self.enc.max_token_value+1).float().unsqueeze(0).to(self.device)
            return seq
        else:
            return None
    def decode(self, seq:torch.Tensor)->str:
        outseq=seq.argmax(dim=-1).tolist()
        return self.enc.decode(outseq)
    def seq_normalize(self,seq:Optional[torch.Tensor],only_start:bool=False)->torch.Tensor:
        if seq==None:
            if only_start:
                return self.start_token
            return torch.cat([self.start_token,self.end_token],dim=1)
        else:
            if only_start:
                return torch.cat([self.start_token.repeat(seq.shape[0],1,1),seq],dim=1)
            return torch.cat([self.start_token.repeat(seq.shape[0],1,1),seq,self.end_token.repeat(seq.shape[0],1,1)],dim=1)
    def predict(self, s:str,prompt:str)->str:
        self.net.eval()
        seq=self.encode(s)
        promptseq=self.encode(prompt)
        #add start and end token to seq,only add start token to prompt
        #notice that self.start_token,self.end_token has a dimension 0 of 1,while seq has a dimension 0 of batchsize
        #so we need to unsqueeze the start and end token to make them have the same dimension as seq
        # we use repeat to make the start and end token have the same dimension as seq
        seq=self.seq_normalize(seq)
        promptseq=self.seq_normalize(promptseq,only_start=True)
        outseq=self.net.predict(seq,promptseq)
        outseq=outseq.squeeze(0)
        return self.decode(outseq)
    def batchify(self, data:List[Tuple[str,str]],batchsize:int=64)->Iterator[Tuple[torch.Tensor,torch.Tensor]]:
        for i in range(0,len(data),batchsize):
            batch=data[i:i+batchsize]
            x=[self.encode(s) for s,_ in batch if self.encode(s)!=None]
            y=[self.encode(s) for _,s in batch if self.encode(s)!=None]
            if len(x)>0:
                x=torch.cat(x,dim=0)
            else:
                x=None
            if len(y)>0:
                y=torch.cat(y,dim=0)
            else:
                y=None
            x=self.seq_normalize(x)
            y=self.seq_normalize(y)
            yield x,y
    #train the model,use tqdm to show progress
    def train(self, traindata:List[Tuple[str,str]],traintime:int=10,batchsize:int=64):
        self.net.train()
        for epoch in (range(traintime)):
            tbar=tqdm(range(len(traindata)//batchsize))
            tbar.set_description_str(f"Epoch {epoch+1}/{traintime}")
            totalloss=0
            avgloss=0
            for i,(x,y) in enumerate(self.batchifier(traindata,batchsize=batchsize)):
                out=self.net(x,y)
                loss=self.loss(out.transpose(2,1)[:,:,:-1],y.transpose(2,1)[:,:,1:])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                totalloss+=loss.item()
                avgloss=totalloss/(i+1)
                tbar.update(1)
                tbar.set_postfix(loss=avgloss,lr=self.scheduler.get_last_lr()[0])
    def save(self,path:str):
        torch.save(self.net.state_dict(),path)
    def load(self,path:str):
        self.net.load_state_dict(torch.load(path,map_location=self.device))
def generate_random_data(n:int=10000)->List[Tuple[str,str]]:
    poems=[]
    for i in range(n):
        poem=""
        return_poem="I have a "
        for j in range(random.randint(1,1)):
            s=random.choice(string.ascii_letters)
            poem+=s
            return_poem+=s
        poems.append((poem,return_poem))
    return poems
def generate_poems_data(n:int=10000)->List[Tuple[str,str]]:
    poems=[]
    #os.walk through every file in the datapath and open them
    for root,dirs,files in os.walk(Params["datapath"]):
        for file in files:
            #open the file
            targetdir=root.split(os.sep)[-1]
            with open(os.path.join(root,file),'r') as f:
                #read the file
                data=f.read()
                #split the file by newline
                poems.append((targetdir,data))
    return poems

if __name__ == "__main__":
    p50k=tiktoken.get_encoding('p50k_base')
    enc=tiktoken.Encoding(name='p50k_base',pat_str=p50k._pat_str,mergeable_ranks=p50k._mergeable_ranks,special_tokens={**p50k._special_tokens,"<im_start>":p50k.max_token_value+1,"<im_end>":p50k.max_token_value+2})
    #enc=Encoder(map=None,startsign="{",endsign="}",device=Params["device"],warning=False)
    net=Wrapper(None,enc,Params["device"])
    #net.load(Params["modelpath"])
    traindata=generate_poems_data(Params["datasize"])
    #print(traindata[:10])
    #traindata=generate_random_data(Params["datasize"])
    #print(traindata[0])
    net.train(traindata,Params["traintime"],Params["batchsize"])
    net.save(Params["modelpath"])
    print(net.predict("2 ABC of H.k. and China revised vision.","tears"))