import os
import torch
import numpy as np
from typing import Tuple
class Data_Manager():
    def __init__(self,path:str) -> None:
        self.datapath=path
        self.data=None
        self.label=None
        ## label is a one-hot distribution in [1,x,x^2,x^3]
    def Make_Data(self,batchsize,inputsize):
        self.data=[]
        self.label=[]
        self.ans=[]
        for i in range(batchsize):
            target=np.random.randint(0,4)
            x=np.linspace(-4,4,inputsize)
            self.label.append([i==target for i in range(0,4)])
            y=(10*np.random.rand())*x**target+np.random.rand()
            self.data.append(y)
            self.ans.append(target)
        self.data=torch.tensor(self.data,dtype=torch.float)
        self.ans=torch.tensor(self.ans,dtype=torch.float)
        self.label=torch.tensor(self.label,dtype=torch.float)
    def Return_Data(self)->Tuple[torch.Tensor,torch.Tensor]:
        return self.data,self.label
    def Return_Ans(self)->torch.Tensor:
        return self.ans