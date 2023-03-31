import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np

with open("./loss.pkl", "rb") as f:
    loss_list = pkl.load(f)

#draw the delta loss
loss_list=np.array(loss_list)
plt.plot(np.arange(len(loss_list)),loss_list, label="loss")
plt.legend()
plt.savefig("./loss.png")
