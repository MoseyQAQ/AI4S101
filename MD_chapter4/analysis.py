import numpy as np
import matplotlib.pyplot as plt

a = np.load('a.npy')
n = np.load('n.npy')
print(a)
plt.plot(n-a,'o')
plt.show()