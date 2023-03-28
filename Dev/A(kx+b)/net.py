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
from torchvision.transforms import ToTensor
import numpy as np
import sys
import pickle as pkl
# def a dict named Params that contains the essential parameters for training

Params = {
    "input_size": 1,
    "output_size": 10,
    "hidden_size": 32,
    "num_layers": 3,
    "lr": 1e-1,
    "batch_size": 512,
    "train_time": 10,
    "NetDimension": "2d",
    "num_epochs": 100,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "losspath": "./loss.png",
    "modelpath": "./model.pth",
    "enable_W": True,
    "loss_threshold": 3.0,
    "loss_listpath": "./loss_list.pkl",
}
Params["input_size"] = 32*32*3 if Params["NetDimension"] == "1d" else 3
Params["losspath"] = "./loss.png" if Params["enable_W"] else "./loss_noW.png"
Params["loss_listpath"] = "./loss_list.pkl" if Params["enable_W"] else "./loss_list_noW.pkl"


class DevLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        # create a ParameterList with a single random tensor, store [0] in self.W
        self.W = nn.ParameterList([torch.tensor(1., requires_grad=True)])[0]
        self.batchnorm = nn.BatchNorm1d(output_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.init_weights()

    def init_weights(self):
        init.xavier_uniform_(self.linear.weight)
        init.zeros_(self.linear.bias)

    def forward(self, x):
        x = self.linear(x)
        #x = self.W*x
        x = self.batchnorm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class DevLayer2D(nn.Module):
    def __init__(self, input_size, output_size, stride, enable_W: bool = True):
        super().__init__()
        self.linear = nn.Conv2d(input_size, output_size,
                                kernel_size=5, stride=stride, padding=2)
        # create a ParameterList with a single random tensor, store [0] in self.W
        if enable_W:
            self.W=torch.ones((1,output_size,1,1),device=Params["device"],requires_grad=True)
            self.W=nn.ParameterList([self.W])
        else:
            self.W=torch.ones((1,output_size,1,1),device=Params["device"],requires_grad=False)
            self.W=nn.ParameterList([self.W])
        self.batchnorm = nn.BatchNorm2d(output_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.enable_W = enable_W

    def forward(self, x):
        x = self.linear(x)
        if self.enable_W:
            x = self.W[0]*x
        x = self.batchnorm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class DevNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, dimension: str = "1d",enable_W: bool = Params["enable_W"]):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        if dimension == "1d":
            self.layers = nn.ModuleList([DevLayer(input_size, hidden_size)])
            self.layers.extend([DevLayer(hidden_size, hidden_size)
                               for _ in range(num_layers-2)])
            self.layers.append(DevLayer(hidden_size, output_size))
        elif dimension == "2d":
            self.layers = nn.ModuleList(
                [DevLayer2D(input_size, 32, stride=1, enable_W=enable_W)])
            self.layers.extend([DevLayer2D(
                32, 32, stride=2, enable_W=enable_W) for _ in range(num_layers-2)])
            self.layers.append(DevLayer2D(
                32, 32, stride=1, enable_W=enable_W))
            # add a fully connected layer to convert 2d to 1d
            self.fc = DevLayer(8192, output_size)
        else:
            raise ValueError("dimension must be 1d or 2d")
        self.printcnt=self.print_time()
    def print_time(self)->Iterator[int]:
        for i in range(10000+Params["train_time"]):
            yield i
    def forward(self, x):
        if Params["NetDimension"] == "1d":
            x = x.view(x.shape[0], -1)
            assert x.shape[1] == self.input_size
        x = x.to(Params["device"])
        for layer in self.layers:
            x = layer(x)
        if Params["NetDimension"] == "2d":
            x = x.view(x.shape[0], -1)
            x = self.fc(x)
        return x
    # print self.model structure, especially W

    def print(self):
        # print all the layer and it's size in torchsummary format
        #print(self)

        # print all the layer's W in a clean way if enable_W is True
        #if Params["enable_W"]:
        #    for i, layer in enumerate(self.layers):
        #        print(f"layer {i} W: {layer.W}")
        #    print(f"fc W: {self.fc.W}")
        # plot all the layer's W in a 2D graph if enable_W is True, and save it in W.png
        if Params["enable_W"]:
            img=torch.stack([layer.W[0].view(-1) for layer in self.layers],dim=0)
            print(img)
            plt.imshow(img.detach().cpu().numpy())
            plt.savefig(f"W-{next(self.printcnt)}.png")
            plt.clf()


# def a class to create a NNET instance containing a DevNet model and the optimizer and the scheduler
# use parameters defined in dict Param, do not transfer much parameters into __init__
# use cross entropy loss function
class NNET:
    def __init__(self,enable_W:bool):
        self.model = DevNet(Params["input_size"], Params["output_size"],
                            Params["hidden_size"], Params["num_layers"], Params["NetDimension"],enable_W=enable_W)
        self.model.to(Params["device"])
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=Params["lr"])
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=200, gamma=0.8)
        self.loss_fn = nn.CrossEntropyLoss()
        self.tcnt = self.train_cnt()

    def train_cnt(self) -> Iterator[int]:
        # self.model.train()
        for i in range(100000+Params["train_time"]):
            yield i

    def train(self, train_loader: DataLoader) -> List[float]:
        self.model.train()
        losses = []
        # use tqdm to show the process of training, containing lr and avgloss
        tbar = tqdm(range(len(train_loader)))
        # set the title of tbar, use train_cnt to illustrate the train epoch
        tbar.set_description(f"Train Epoch: {next(self.tcnt)}")
        total_loss = 0
        avg_loss = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(Params["device"])
            y = y.to(Params["device"])
            self.optimizer.zero_grad()
            y_pred = self.model(x)
            loss = self.loss_fn(y_pred, y)
            loss.backward()
            self.optimizer.step()
            losses.append(min(Params["loss_threshold"] , loss.item()))
            total_loss += loss.item()
            avg_loss = total_loss/(batch_idx+1)
            tbar.set_postfix(lr=self.scheduler.get_last_lr()
                             [0], avgloss=avg_loss)
            tbar.update(1)
            self.scheduler.step()
        return losses

    def predict(self, test_loader: DataLoader) -> Iterator[torch.Tensor]:
        self.model.eval()
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(Params["device"])
                y_pred = self.model(x)
                yield y_pred

# now we can create a NNET instance and train it
# use the MNIST dataset


def get_data_loader(batch_size: int, train: bool = True) -> DataLoader:
    dataset = datasets.CIFAR10(
        root="/root/autodl-tmp/data",
        train=train,
        transform=ToTensor(),
        download=True,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train(model: NNET, train_loader: DataLoader, num_epochs: int) -> None:
    train_losses = []
    test_losses = []
    for epoch in range(Params["train_time"]):
        train_loss = model.train(train_loader)
        train_losses.extend(train_loss)
        model.model.print()
    plt.plot(train_losses, label="training loss")
    plt.legend()
    # save the plot in Params['losspath']
    plt.savefig(Params['losspath'])
    plt.clf()
    #use pickle as pkl to save the train_losses in Params['loss_listpath']
    with open(Params['loss_listpath'], 'wb') as f:
        pkl.dump(train_losses, f)
# test the accuracy of the model on MNIST


def test(model: NNET, test_loader: DataLoader) -> None:
    model.model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(Params["device"])
            y = y.to(Params["device"])
            y_pred = model.model(x)
            pred = y_pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(y.view_as(pred)).sum().item()
    print(f"Accuracy: {correct / len(test_loader.dataset):.4f}")
# create an DevNet instance and save it in Params['modelpath']
def init_saveWeights():
    model = DevNet(Params["input_size"], Params["output_size"],
                   Params["hidden_size"], Params["num_layers"], Params["NetDimension"],enable_W=True)
    torch.save(model.state_dict(), Params['modelpath'])
    print("save model successfully")


if __name__ == "__main__":
    #init_saveWeights()
    #exit()
    train_loader = get_data_loader(Params["batch_size"])
    model = NNET(Params["enable_W"])
    #load the model from Params['modelpath']
    model.model.load_state_dict(torch.load(Params['modelpath']))
    model.model.print()
    train(model, train_loader, Params["num_epochs"])
    # print the model structure
    #model.model.print()
    # test the model
    test_loader = get_data_loader(Params["batch_size"], train=False)
    test(model, test_loader)
    # save the model
    #torch.save(model.model.state_dict(), Params['modelpath'])
else:
    #init_saveWeights()
    pass