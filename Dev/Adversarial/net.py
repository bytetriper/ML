import torch
import torch.nn.functional as F
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from typing import List, Tuple, Optional
import numpy as np
import pickle as pkl
from torchvision.models import VGG16_Weights
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.nn.init import kaiming_normal_
import cv2
Params = {
    "batch_size": 256,
    "lr": 1e-5,
    "epoch": 2,
    "train_time": 1,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "net_save_path": "/root/autodl-tmp/ML/Dev/Adversarial/params.pth",
    "net_load_path": "/root/autodl-tmp/ML/Dev/Adversarial/params.pth",
    "gen_save_path": "/root/autodl-tmp/ML/Dev/Adversarial/gen.pth",
    "gen_load_path": "/root/autodl-tmp/ML/Dev/Adversarial/gen.pth",
    "data_path": "/root/autodl-tmp/data",
    "img_path": "/root/autodl-tmp/ML/Dev/Adversarial/adv.png",
    "img_trick_path": "/root/autodl-tmp/ML/Dev/Adversarial/adv_trick.png",
    "loss_path": "/root/autodl-tmp/ML/Dev/Adversarial/loss.png",
}


class VGG_For_CIFAR10(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = models.vgg16(weights=VGG16_Weights.DEFAULT)
        self.fc = nn.Linear(1000, 10)
        self.fc.weight.data.normal_(0, 0.01)
        self.fc.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        x = self.fc(x)
        return x

class Img_Generator(nn.Module):
    def __init__(self,in_size:int) -> None:
        #in_channel=3=out_channel
        #in_size=out_size
        super().__init__()
        self.downblock=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2,padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=5,stride=2,padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        self.upblock=nn.Sequential(
            nn.ConvTranspose2d(in_channels=64,out_channels=64,kernel_size=5,stride=2,padding=2,output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64,out_channels=64,kernel_size=5,stride=2,padding=2,output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.ConvTo3=nn.Conv2d(in_channels=64,out_channels=3,kernel_size=3,stride=1,padding=1)
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode="fan_out",nonlinearity="relu")
                m.weight.data*=1e-4
            elif isinstance(m,nn.BatchNorm2d) and m.affine:
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
    def forward(self,x:torch.Tensor)->torch.Tensor:
        s=self.downblock(x)
        s=self.upblock(s)
        s=self.ConvTo3(s)
        s=x+s
        s=s.clip(0,1)
        return s


class Adversarial():
    def __init__(self, device: torch.device, input: torch.Tensor, label: int, load: bool = False) -> None:
        self.net = VGG_For_CIFAR10().to(device)
        self.gen = Img_Generator(32).to(device)
        self.device = device
        self.input = input
        self.original_input = input.clone()
        self.input.requires_grad_()
        self.label = torch.tensor([label]).to(device)
        self.false_label = torch.tensor([9-label]).to(device)
        print(self.false_label)
        print(self.label)
        if load:
            self.net.load_state_dict(torch.load(Params["net_load_path"]))
        self.optimizer = torch.optim.Adam(
            self.gen.parameters(), lr=Params["lr"])
        self.input_optimizer = torch.optim.Adam([self.input], lr=Params["lr"])
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=50, gamma=0.9)
        self.input_scheduler = torch.optim.lr_scheduler.StepLR(
            self.input_optimizer, step_size=50, gamma=1.1)
        self.loss = nn.CrossEntropyLoss(reduction='mean')
    
    def maximum_loss(self,origin:torch.Tensor,inp:torch.Tensor,threshold: float = .07):
        loss=torch.abs(origin-inp)
        #l2_loss=torch.sqrt(loss.sum(dim=1))
        l1_loss=nn.L1Loss(reduction='mean')
        return l1_loss(origin,inp)*8
        return loss.square().mean()
        #if torch.max(loss) < threshold:
        #    return torch.zeros(1,device=self.device)
        #else:
        #    return ((loss>= threshold)*1.e1).mean()
    def trick_input(self, epoch: int, trick: bool = True, threshold: float = 9) -> None:
        tbar = tqdm(range(epoch))
        tbar.set_description_str("tricking input")
        for i in range(epoch):
            self.optimizer.zero_grad()
            output = self.net(self.input)
            if trick:
                label_loss = self.loss(output, self.false_label)
                maxloss= self.maximum_loss(self.original_input,self.input)
                loss = label_loss+maxloss
                if abs(label_loss.item()) < threshold:
                    print("last loss:", loss.item())
                    break
            else:
                label_loss = -self.loss(output, self.label)
                maxloss= self.maximum_loss(self.original_input,self.input)
                loss = label_loss+maxloss
                if label_loss.item() < -threshold:
                    print("last loss:", loss.item())
                    break
            loss.backward()
            tbar.update(1)
            tbar.set_postfix_str(f"l2_loss:{maxloss.item():.4e} label_loss:{label_loss.item():.4e}")
            self.input_optimizer.step()
            self.input_scheduler.step()
        return
    def train(self, train_loader: DataLoader, epoch: int) -> Tuple[List[float], List[float], List[float]]:
        self.net.train()
        
        loss_list = []
        label_loss_list = []
        l2_loss_list = []
        for time_epoch in range(Params["train_time"]):
            tbar = tqdm(range(len(train_loader)*epoch))
            tbar.set_description_str("training")
            total_loss = 0
            avg_loss = 0
            total_label_loss = 0
            avg_label_loss = 0
            total_l2_loss = 0
            avg_l2_loss = 0
            for i, (x, y) in enumerate(train_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                for e in range(epoch):
                    self.optimizer.zero_grad()
                    gen_output = self.gen(x)
                    if gen_output.max() > 1:
                        print("gen_output.max:",gen_output.max())
                    assert gen_output.max() <= 1
                    assert gen_output.min() >= 0
                    output = self.net(gen_output)
                    label_loss = -self.loss(output, y)
                    l2_loss = self.maximum_loss(x,gen_output)
                    
                    total_l2_loss += l2_loss.item()
                    total_label_loss += label_loss.item()
                    avg_l2_loss = total_l2_loss/(i+1)
                    avg_label_loss = total_label_loss/(i+1)
                    loss = label_loss+l2_loss
                    loss_list.append(loss.item())
                    label_loss_list.append(label_loss.item())
                    l2_loss_list.append(l2_loss.item())
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()
                    avg_loss = total_loss/(i+1)
                    tbar.update(1)
                    tbar.set_postfix_str(f"loss:{avg_loss:.3e},label_loss:{avg_label_loss:.3e},l2_loss:{avg_l2_loss:.3e},max_loss:{torch.max(torch.abs(x-gen_output)).item():.3e},mean_loss:{torch.mean(torch.abs(x-gen_output)).item():.3e}")
                self.scheduler.step()
        return loss_list, label_loss_list, l2_loss_list
    def save(self) -> None:
        torch.save(self.net.state_dict(), Params["net_save_path"])
        torch.save(self.gen.state_dict(), Params["gen_save_path"])

    def predict_input(self) -> int:
        self.net.eval()
        with torch.no_grad():
            output = self.net(self.input)
            return output.argmax().item()

    def predict(self, s: torch.Tensor) -> int:
        self.net.eval()
        with torch.no_grad():
            output = self.net(s)
            return output.argmax().item()


def save_img(img: torch.Tensor, path: str) -> None:
    # use cv2 to save the image
    # remember img are in order of (batch_size,channel,height,width) and RGB
    # but cv2 need (height,width,channel) and BGR
    sav_img = img.cpu().detach().numpy()
    sav_img = sav_img.squeeze(0)
    sav_img = np.transpose(sav_img, (1, 2, 0))
    sav_img = sav_img*255
    sav_img = sav_img.astype(np.uint8)
    sav_img = cv2.cvtColor(sav_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, sav_img)


def train_model():
    train_set = CIFAR10(
        root=Params["data_path"], train=True, download=True, transform=ToTensor())
    train_loader = DataLoader(
        train_set, batch_size=Params["batch_size"], shuffle=True)
    # get a random image in the dataset
    input = train_loader.dataset[0][0].unsqueeze(0)
    input = input.to(Params["device"])
    device = Params["device"]
    adv = Adversarial(device, input, True)
    loss,l_loss,t_loss =adv.train(train_loader, Params["epoch"])
    adv.save()
    plt.plot(loss, label="loss")
    plt.plot(l_loss, label="label_loss")
    plt.plot(t_loss, label="l2_loss")
    plt.legend()
    plt.savefig(Params["loss_path"])
    return adv


def test_model_acc(model: Adversarial):
    test_set = CIFAR10(
        root=Params["data_path"], train=False, download=True, transform=ToTensor())
    test_loader = DataLoader(
        test_set, batch_size=Params["batch_size"], shuffle=True)
    model.net.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for i, (x, y) in enumerate(test_loader):
            x = x.to(Params["device"])
            y = y.to(Params["device"])
            output = model.net(x)
            _, pred = torch.max(output.data, 1)
            total += y.size(0)
            correct += (pred == y).sum().item()
        print(f"acc:{correct/total:.4f}")
    rand_idx = np.random.randint(0, 1000)
    rand_img=test_loader.dataset[rand_idx][0].unsqueeze(0)
    rand_img = rand_img.to(Params["device"])
    model.input = rand_img
    #print(f"predict:{model.predict_input()}")
    save_img(model.gen(model.input), Params["img_path"])


def trick_input(model: Adversarial, epoch: int, trick: bool = True):
    model.trick_input(epoch, trick)
    if trick:
        save_img(model.input, Params["img_trick_path"])
    else:
        save_img(model.input, Params["img_path"])
    print(f"predict:{model.predict_input()}")


def get_model_withCIFAR10Input():
    train_set = CIFAR10(
        root=Params["data_path"], train=False, download=True, transform=ToTensor())
    train_loader = DataLoader(
        train_set, batch_size=Params["batch_size"], shuffle=True)
    # get a random image in the dataset
    rand_idx = np.random.randint(0, 1000)
    #rand_idx = 33
    rand_idx = 943
    print("rand_idx:", rand_idx)
    input = train_loader.dataset[rand_idx][0].unsqueeze(0)
    label = train_loader.dataset[rand_idx][1]
    input = input.to(Params["device"])
    device = Params["device"]
    adv = Adversarial(device, input, label, True)
    return adv

def give_exampleimg_with_model():
    model=get_model_withCIFAR10Input()
    model.gen.load_state_dict(torch.load(Params["gen_save_path"]))
    model.net.eval()
    with torch.no_grad():
        output = model.gen(model.input)
        final_fc= model.net(output)
        print(f"predict:{final_fc.argmax().item()}")
        print(f"label:{model.label}")
        save_img(model.input, Params["img_path"])
        save_img(output, Params["img_trick_path"])
if __name__ == "__main__":
    #give_exampleimg_with_model()
    model=train_model()
    test_model_acc(model)
    
    #train_model()

    
    

