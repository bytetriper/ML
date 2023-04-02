import torch
from typing import Tuple
from typing import List
from typing import Dict
from typing import Union

map={}
class Encoder():
    def __init__(self,map:Union[Dict[str,int],None],startsign:str,endsign:str,device:torch.device) -> None:
        self.map={}
        self.Available_Charset="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890{}!@#$%^&*()_+-=;'\":.,<>?/\\[]|~` \n\t\r"
        for i in range(len(self.Available_Charset)):
            self.map[self.Available_Charset[i]]=i
        if map!=None:
            self.map=map
        self.startsign=startsign
        self.device=device
        self.endsign=endsign
        self.startvec=torch.zeros(len(self.map),device=device)
        self.startvec[self.map[self.startsign]]=1
        self.endvec=torch.zeros(len(self.map),device=device)
        self.endvec[self.map[self.endsign]]=1
        self.reversed_map={self.map[k]:k for k in self.map.keys()}
    def encode(self,text:str,closed:bool=True)->List[torch.Tensor]:
        data=[]
        data.append(self.startvec)
        for i in range(len(text)):
            s=text[i]
            if s not in self.map.keys():
                #output an error message
                print("Error: Char{} not in charset".format(ord(s)))
                break
            tmp=torch.zeros(len(self.map),device=self.device)
            tmp[self.map[s]]=1.
            data.append(tmp)
        if closed:
            data.append(self.endvec)
        return data
    def decode(self,output:List[torch.Tensor])->str:
        ans=""
        for y in output:
            select=int(torch.argmax(y).item())
            ans+=self.reversed_map[select]
            #if self.reversed_map[select]==self.endsign:
            #    break
        return ans