import os
import torch
import numpy as np
from typing import Tuple,List
class Data_Manager():
    def __init__(self,path:str) -> None:
        self.datapath=path
        self.data=None
        self.label=None
        ## label is a one-hot distribution in [1,x,x^2,x^3]
    def Make_Data(self,batchsize,inputsize,device):
        self.data=[]
        for i in range(batchsize):
            target=np.random.randint(0,4)
            Hello=[7,4,11,14]
            y=torch.eye(inputsize,inputsize,dtype=torch.float)
            y=y[Hello].to(device)
            self.data.append(y)
    def Return_Data(self)->List[torch.Tensor]:
        return self.data
   