import torch
import torch.nn.functional as F
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from typing import List, Tuple, Optional
import numpy as np
import pickle as pkl
import torchvision
import random
import tiktoken
import os
Params = {
    "modelpath": "/root/autodl-tmp/ML/Models/decoder.pth",
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "lr": 8e-5,
    "traintime": 10,
    "batch_size": 64,
    "maxlen": 50,
    "datapath": "/root/autodl-tmp/data/poems",
    "datasize": 1024,

}


class TransDecode(nn.Module):
    def __init__(self, enc: tiktoken.Encoding, device: torch.device) -> None:
        super().__init__()
        self.layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        self.transformer = nn.TransformerDecoder(self.layer, num_layers=6)
        print(enc.n_vocab)
        self.EncW = nn.Linear(enc.n_vocab+1, 512)
        self.DecW = nn.Linear(512, enc.n_vocab)
        self.end_selection_sign = enc._special_tokens['<im_end>']
        self.device = device
        self.enc = enc

    def forward(self, inp: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        inp = self.add_position_encoding(inp)
        inp = self.EncW(inp)
        out = self.transformer(inp, memory=torch.zeros_like(inp))
        out = self.DecW(out)
        return out

    def add_position_encoding(self, inp: torch.Tensor, st: int = 0) -> torch.Tensor:
        position = torch.arange(
            st, st+inp.shape[1], device=self.device).unsqueeze(0).repeat(inp.shape[0], 1).unsqueeze(-1)
        return torch.cat([inp, position], dim=-1)

    def generate(self, inp: torch.Tensor, mask: Optional[torch.Tensor] = None, max_len: int = 100) -> torch.Tensor:
        ans = inp
        inp = self.add_position_encoding(inp)
        inp = self.EncW(inp)
        for i in range(inp.shape[0], inp.shape[0]+max_len):
            out = self.transformer(inp, memory=torch.zeros_like(inp))
            out = self.DecW(out)
            out = out[:, -1, :]
            out = out.argmax(dim=-1)
            if out == self.end_selection_sign:
                break
            out = F.one_hot(out, self.enc.n_vocab).float().to(
                self.device).unsqueeze(0)
            ans = torch.cat([ans, out], dim=1)
            out = self.add_position_encoding(out, st=i)
            out = self.EncW(out)
            inp = torch.cat([inp, out], dim=1)
        return ans


class Wrapper():
    def __init__(self, enc: tiktoken.Encoding, device: torch.device) -> None:
        self.enc = enc
        self.device = device
        self.model = TransDecode(enc, device)
        self.model = self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=Params["lr"])
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1, gamma=0.9)
        self.startvec = F.one_hot(torch.tensor(
            [enc._special_tokens['<im_start>']]), enc.n_vocab).float().to(device)
        self.endvec = F.one_hot(torch.tensor(
            [enc._special_tokens['<im_end>']]), enc.n_vocab).float().to(device)

    def encode(self, inp: str, normalize: bool = True, only_st: bool = False) -> torch.Tensor:
        # generate one-hot vector
        seq = self.enc.encode(inp)
        if len(seq) == 0:
            return self.startvec
        seq = F.one_hot(torch.tensor(seq), self.enc.n_vocab
                        ).float().to(self.device)
        if normalize:
            return torch.cat([self.startvec, seq, self.endvec], dim=0) if not only_st else torch.cat([self.startvec, seq], dim=0)
        else:
            return seq

    def decode(self, inp: torch.Tensor) -> str:
        # generate one-hot vector
        seq = inp.argmax(dim=-1).squeeze(0)
        seq = self.enc.decode(seq.tolist())
        return seq

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path))

    def add_padding(self, inp: List[torch.Tensor], max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # add padding or cut down maxlength to make all sequences in seq of same length:max_length

        # inp: List[(seq_len, n_vocab+1)]
        # out: List[(max_len, n_vocab+1)]
        out = []
        mask = []
        for seq in inp:
            if seq.shape[0] > max_len:
                out.append(seq[:max_len])
                mask.append(torch.ones(max_len, device=self.device))
            else:
                # pad
                pad = torch.zeros(
                    max_len-seq.shape[0], seq.shape[1], device=self.device)
                out.append(torch.cat([seq, pad], dim=0))
                mask.append(torch.cat([torch.ones(seq.shape[0], device=self.device), torch.zeros(
                    max_len-seq.shape[0], device=self.device)]))
        return torch.stack(out, dim=0), torch.stack(mask, dim=0)

    def loss(self, output: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # output: (batch_size, seq_len, n_vocab+1)
        # target: (batch_size, seq_len)
        # mask: (batch_size, seq_len)
        if mask is None:
            mask = torch.ones_like(target)
        loss = F.cross_entropy(
            output.reshape(-1, output.shape[-1]), target.reshape(-1), reduction='none')
        loss = loss.reshape(target.shape)
        loss = loss * mask
        loss = loss.sum() / mask.sum()
        return loss

    def encode_data(self, inp: List[str], normalize: bool = True, only_st: bool = False) -> List[torch.Tensor]:
        # inp: List[str]
        # out: List[(seq_len, n_vocab+1)]
        out = []
        for seq in inp:
            out.append(self.encode(seq, normalize, only_st))
        return out

    def encode_label(self, inp: List[str], normalize: bool = True, only_st: bool = False) -> List[List[int]]:
        # inp: List[str]
        # out: List[List[int]]
        out = []
        for seq in inp:
            trans = []
            if normalize:
                trans.append(self.enc._special_tokens['<im_start>'])
            trans.extend(self.enc.encode(seq))
            if normalize and not only_st:
                trans.append(self.enc._special_tokens['<im_end>'])
            out.append(trans)
        return out

    def add_label_padding(self, labels: List[List[int]], max_len: int) -> torch.Tensor:
        # labels: List[List[int]]
        # out: (batch_size, max_len)
        out = []
        for label in labels:
            if len(label) > max_len:
                out.append(label[:max_len])
            else:
                # pad
                pad = [0] * (max_len-len(label))
                out.append(label+pad)
        return torch.tensor(out, device=self.device)

    def train(self, data: List[str], epochs: int = 100, batch_size: int = 32):
        self.model.train()
        train_data = self.encode_data(data)
        labels = self.encode_label(data)
        for epoch in range(epochs):
            tbar = tqdm(range(0, len(train_data)//batch_size))
            tbar.set_description_str(f'Epoch {epoch}')
            total_loss = 0
            avg_loss = 0
            for subepoch in range(len(train_data)//batch_size):
                # random choice a batchidx of batchsize and use that to list the batch and label
                batchidx = np.random.choice(
                    len(train_data), batch_size, replace=False)
                tmpbatch = [train_data[i] for i in batchidx]
                tmplabel = [labels[i] for i in batchidx]
                batch, mask = self.add_padding(
                    tmpbatch, max_len=Params["maxlen"])
                label = self.add_label_padding(
                    tmplabel, max_len=Params["maxlen"])
                # train
                self.optimizer.zero_grad()
                output = self.model(batch)
                loss = self.loss(output[:, :-1, :], label[:, 1:], mask[:, :-1])
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                avg_loss = total_loss / (subepoch+1)
                tbar.update(1)
                tbar.set_postfix(
                    loss=avg_loss, lr=self.scheduler.get_last_lr()[0])
            self.scheduler.step()

    def predict(self, inp: str, max_len: int = 100) -> str:
        self.model.eval()
        seq = self.encode(inp, only_st=True)
        seq = seq.unsqueeze(0)
        seq = self.model.generate(seq, max_len=max_len)
        print(seq.shape)
        seq = self.decode(seq)
        return seq


def generate_poems_data(n: int = 10000) -> List[str]:
    poems = []
    # os.walk through every file in the datapath and open them
    for root, dirs, files in os.walk(Params["datapath"]):
        for file in files:
            # open the file
            targetdir = root.split(os.sep)[-1]
            with open(os.path.join(root, file), 'r') as f:
                # read the file
                data = f.read()
                # split the file by newline
                data = data.split('\n')
                # remove the empty string
                data = [i for i in data if i != '']
                # remove the first line
                data = data[1:]
                # remove the last line
                data = data[:-1]
                # add every data in the file to the poems list
                poems.extend(data)
                if len(poems) > n:
                    return poems
    return poems


if __name__ == "__main__":
    # test TransDecode
    stdenc = tiktoken.get_encoding('p50k_base')
    custom_enc = tiktoken.Encoding(
        name='p50k_base',
        pat_str=stdenc._pat_str, mergeable_ranks=stdenc._mergeable_ranks, special_tokens={**stdenc
                                                                                          ._special_tokens, "<im_start>": stdenc.max_token_value+1, "<im_end>": stdenc.max_token_value+2, "<im_pad>": stdenc.max_token_value+3})
    model = Wrapper(custom_enc, Params["device"])
    data = generate_poems_data(Params["datasize"])
    # print("data size:",len(data))
    model.load(Params["modelpath"])
    #model.train(data,epochs=Params["traintime"],batch_size=Params["batch_size"])
    #model.save(Params["modelpath"])
    print(model.predict("Patterns of Gods are", max_len=Params["maxlen"]))
