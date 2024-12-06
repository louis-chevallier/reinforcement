import pickle
import matplotlib.pyplot as plt
from utillc import *
with open("loss.pickle","rb") as fd :
    l = pickle.load(fd)
EKOX(len(l))
plt.plot(l)
plt.show()
