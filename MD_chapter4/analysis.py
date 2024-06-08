import numpy as np
import matplotlib.pyplot as plt

my_force = np.load('force.npy')
ref_force = np.loadtxt('ref_force.out')[:64]
plt.plot(my_force[64:]-my_force[:64], 'o')
print(np.max(np.abs(my_force[:64]- ref_force)))
plt.savefig('tersoff-validation.pdf')