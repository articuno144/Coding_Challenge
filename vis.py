import numpy as np
from matplotlib import pyplot as plt

wr = np.load("save/winrate.npy")
avgl = np.load("save/avglen.npy")

plt.plot(wr)
plt.show()
