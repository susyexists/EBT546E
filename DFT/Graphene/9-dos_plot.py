import matplotlib.pyplot as plt
from matplotlib import rcParamsDefault
import numpy as np

# load data
energy, dos, idos = np.loadtxt('./dos.dat', unpack=True)
e_fermi = 0.2539


# make plot
plt.figure(figsize = (12, 6))
plt.plot(energy, dos, linewidth=0.75, color='red')
plt.yticks([])
plt.xlabel('Energy (eV)')
plt.ylabel('DOS')
plt.axvline(x=e_fermi, linewidth=0.5, color='k', linestyle=(0, (8, 10)))
plt.xlim(-5, 5)
plt.ylim(0, )
plt.fill_between(energy, 0, dos, where=(energy < e_fermi), facecolor='red', alpha=0.25)
plt.text(-0.5, 1.7, 'Fermi energy',  rotation=90)
plt.savefig("dos.png")
plt.close()