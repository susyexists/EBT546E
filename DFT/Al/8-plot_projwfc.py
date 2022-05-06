import matplotlib.pyplot as plt
from matplotlib import rcParamsDefault
import numpy as np

# load data
def data_loader(fname):
    fid = open(fname, "r")
    data = fid.readlines()
    fid.close()

    energy = []
    pdos = []

    for row in range(len(data)):
        data_row = data[row]
        if (data_row[0][0] != '#'):
            data_row = data_row[:-1].split('  ')
            energy.append(float(data_row[1]))
            pdos.append(float(data_row[3]))

    energy = np.asarray(energy)
    pdos = np.asarray(pdos)

    return energy, pdos

energy, pdos_s = data_loader('pdos.dat.pdos_atm#1(Al)_wfc#1(s)')
_, pdos_p = data_loader('pdos.dat.pdos_atm#1(Al)_wfc#2(p)')
_, pdos_tot = data_loader('pdos.dat.pdos_tot')

# make plots
plt.figure(figsize = (8, 4))
plt.plot(energy, pdos_s, linewidth=0.75, color='#006699', label='s-orbital')
plt.plot(energy, pdos_p, linewidth=0.75, color='r', label='p-orbital')
plt.plot(energy, pdos_tot, linewidth=0.75, color='k', label='total')
plt.yticks([])
plt.xlabel('Energy (eV)')
plt.ylabel('DOS')
plt.axvline(x= 7.9303, linewidth=0.5, color='k', linestyle=(0, (8, 10)))
plt.xlim(-5, 27)
plt.ylim(0, )
plt.fill_between(energy, 0, pdos_s, where=(energy < 7.9421), facecolor='#006699', alpha=0.25)
plt.fill_between(energy, 0, pdos_p, where=(energy < 7.9421), facecolor='r', alpha=0.25)
plt.fill_between(energy, 0, pdos_tot, where=(energy < 7.9421), facecolor='k', alpha=0.25)
# plt.text(6.5, 0.52, 'Fermi energy', fontsize= small, rotation=90)
plt.legend(frameon=False)
plt.show()