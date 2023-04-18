import torch
from typing import Tuple
from typing import List
from typing import Dict
from typing import Union

map={}
class Encoder():
    def __init__(self,map:Union[Dict[str,int],None],startsign:str,endsign:str,device:torch.device,warning:bool=True) -> None:
        self.map={}
        self.Available_Charset="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890{}!@#$%^&*()_+-=;'\":.,<>?/\\[]|~` \n\t\r"
        for i in range(len(self.Available_Charset)):
            self.map[self.Available_Charset[i]]=i
        if map!=None:
            self.map=map
        self.startsign=startsign
        self.device=device
        self.endsign=endsign
        if self.startsign not in self.map.keys():
            self.map[self.startsign]=len(self.map)
        if self.endsign not in self.map.keys():
            self.map[self.endsign]=len(self.map)
        self.n_vocab=len(self.map)
        self.max_token_value=max(self.map.values())
        self._special_tokens={
            "<im_start>":self.map[self.startsign],
            "<im_end>":self.map[self.endsign]
        }
        self.warning=warning
        self.reversed_map={self.map[k]:k for k in self.map.keys()}
    def encode(self,text:str,closed:bool=True)->List[int]:
        data=[]
        for i in range(len(text)):
            if text[i] not in self.map.keys():
                #output an error message
                if self.warning:
                    print("Encode Error: Char{} not in charset".format(ord(text[i])))
                break
            data.append(self.map[text[i]])
        return data
    def decode(self,output:List[int])->str:
        ans=""
        for y in output:
            if y not in self.reversed_map.keys():
                #output an error message
                if self.warning:
                    print("Decode Error: Char{} not in charset".format(y))
                break
            ans+=self.reversed_map[y]
        return ans