import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np

with open("./loss_list.pkl", "rb") as f:
    loss_list = pkl.load(f)
with open("./loss_list_noW.pkl", "rb") as f:
    loss_list_noW = pkl.load(f)

#draw the delta loss
loss_list=np.array(loss_list)
loss_list_noW=np.array(loss_list_noW)
delta=loss_list_noW-loss_list
print(np.mean(delta))
print(np.std(delta))
plt.plot(np.arange(len(loss_list)),delta, label="with W")
plt.legend()
plt.savefig("./loss_cmp.png")
