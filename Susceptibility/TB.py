# Numerical tools
import scipy.optimize
from scipy.ndimage.filters import gaussian_filter1d
from numpy import *
from numpy.linalg import *

# Plotting
import matplotlib.pyplot as plt

# Data analysis
import pandas as pd

# Parallel computation
from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()


class TB:

    def __init__(self, path, nscf, wout, hr, super_cell=1):
        self.path = path
        self.fermi_energy = self.read_efermi(path+nscf)
        self.g_vec = self.read_gvec(path+wout)
        self.g_length = 1
        data = self.read_hr(path+hr)
        self.hopping = data[0]
        self.nbnd = int(sqrt(len(data[0])/len(data[2])))
        self.points = len(data[2])

        self.super_cell = super_cell
        self.sym = data[2]
        self.h = self.hopping.reshape(self.points, self.nbnd*self.nbnd)
        self.x = data[1].reshape(2, self.points, self.nbnd*self.nbnd)

    def read_hr(self, path):
        lines = open(path, 'r').readlines()
        sym_line = int(ceil(float(lines[2].split()[0])/15))+3
        sym = array([int(lines[i].split()[j]) for i in range(3, sym_line)
                    for j in range(len(lines[i].split()))])
        hr_temp = array([float(lines[i].split()[j]) for i in range(
            sym_line, len(lines)) for j in range(len(lines[i].split()))])
        hr = hr_temp.reshape(-1, 7).T
        wannier = hr
        x = wannier[0:2]
        hopping = wannier[5]+1j*wannier[6]
        return (hopping, x, sym)

    def read_efermi(self, path):
        lines = open(path, 'r').readlines()
        e_fermi = 0
        for i in lines:
            if "the Fermi energy is" in i:
                e_fermi = float(i.split()[-2])
                return e_fermi

    def read_gvec(self, path):
        lines = open(path, 'r').readlines()
        g_vec = zeros(shape=(3, 3))
        count = 0
        for i in lines:
            if "b_" in i:
                if count == 3:
                    continue
                else:
                    g_vec[count] = i.split()[1:]
                    count += 1
        g_vec = g_vec.T[:2].T[:2]
        return (g_vec)

    def phselfen(self, λ, λ_real, γ, γ_real, ω):
        λ = loadtxt(self.path+"lambda.dat").reshape(-1, 9).T
        λ_real = loadtxt(self.path+"lambda_re.dat").reshape(-1, 9).T
        γ = loadtxt(self.path+"gamma.dat").reshape(-1, 9).T
        γ_real = loadtxt(self.path+"gamma_re.dat").reshape(-1, 9).T
        ω = loadtxt(self.path+"omega.dat").reshape(-1, 9).T
        return λ, λ_real, γ, γ_real, ω

    def fourier(self, k):
        kx = tensordot(k, self.x, axes=(0, 0))
        transform = dot(self.sym, exp(-1j*self.super_cell*kx)
                        * self.h).reshape(self.nbnd, self.nbnd)
        return(transform)

    def mesh_2d(self, N):
        # mesh = model.mesh_2d(100)*norm(model.g_vec[0])*pi/2
        one_dim = linspace(-1, 1, N)
        two_dim = array([[i, j] for i in one_dim for j in one_dim])
        return (two_dim)

    def eig(self, k):
        val = []
        vec = []
        for i in range(len(k)):
            sol = linalg.eigh(self.fourier(k[i]))
            val.append(sol[0])
            vec.append(sol[1])
        return (val, vec)

    def solver(self, k):
        kx = tensordot(k, self.x, axes=(0, 0))
        transform = dot(self.sym, exp(-1j*self.super_cell*kx*2*pi)
                        * self.h).reshape(self.nbnd, self.nbnd)
        val, vec = linalg.eigh(transform)
        return(val)

    def parallel_solver(self, path):
        results = Parallel(n_jobs=num_cores)(
            delayed(self.solver)(i) for i in path)
        res = array(results).T-self.fermi_energy
        return (res)

    def fermi(self, e, T=0.01):
        return 1/(1+exp(e/T))

    def suscep(self, point, mesh, mesh_energy, mesh_fermi):
        shifted_energy = self.parallel_solver(point+mesh)[6]
        shifted_fermi = self.fermi(shifted_energy)
        num = mesh_fermi-shifted_fermi
        den = mesh_energy-shifted_energy
        res = -average(num/den)
        return(res)

    def suscep_epw(self, point, mesh, mesh_energy, mesh_fermi, epw1, epw2=[]):
        if epw2 == []:
            epw2 = copy(epw1)
        shifted_energy = self.parallel_solver(point+mesh)[6]
        shifted_fermi = self.fermi(shifted_energy)
        num = mesh_fermi-shifted_fermi
        den = mesh_energy-shifted_energy
        mult = -epw1*conj(epw2)*num/den*10**-3
        res = average(mult)
        return(res)

    def hexagon():
        a = array([[[-1/sqrt(3), 1/sqrt(3)], [1, 1]],
                   [[1/sqrt(3), 2/sqrt(3)], [1, 0]],
                   [[2/sqrt(3), 1/sqrt(3)], [0, -1]],
                   [[1/sqrt(3), -1/sqrt(3)], [-1, -1]],
                   [[-1/sqrt(3), -2/sqrt(3)], [-1, 0]],
                   [[-2/sqrt(3), -1/sqrt(3)], [0, 1]],
                   ])
        return (a)

    def plot_electron_path(self, band, sym, label, temp=None, ylim=None, save=None):

        # Plot band
        plt.figure(figsize=(7, 6))
        for i in band:
            plt.plot(i, c="blue")
        plt.xticks(ticks=sym, labels=label, fontsize=15)
        plt.xlim(sym[0], len(band.T))
        plt.axvline(sym[1], c="black", linestyle="--")
        plt.axvline(sym[2], c="black", linestyle="--")
        plt.axhline(0, linestyle="--", color="red")
        if ylim == None:
            plt.ylim(-0.6, 0.8)
        else:
            plt.ylim(ylim)
        if temp == None:
            plt.title(r"Electron Band", fontsize=15)
        else:
            plt.title(r"Electron Band" "\n" f"σ = {temp}", fontsize=15)
        plt.ylabel("Energy (eV)", fontsize=15)
        return plt.show()

    def t_mesh(self, mesh):
        t_mesh = dot(self.g_vec.T, mesh.T)
        return t_mesh

    def plot_electron_mesh(self, mesh, band, temp=None, save=None):
        t_mesh = self.t_mesh(mesh)
        plt.figure(figsize=(5, 7))
        plt.scatter(t_mesh[0], t_mesh[1], c=band, cmap='jet')
        plt.colorbar()
        plt.xlim(min(t_mesh[0]), max(t_mesh[0]))
        plt.ylim(min(t_mesh[1]), max(t_mesh[1]))
        plt.axis("equal")
        plt.axis("off")
        if temp == None:
            plt.title("")
        else:
            plt.title(f"σ = {temp}")
        return plt.show()


def GMKG(num_points, g_length=1):
    mult = 0.75*num_points/1.00786
    GM = int(norm(array([[0, 0.5]])-array([[0.0001, 0.0001]]))*mult)
    MK = int(norm(array([[0.33333, 0.333333]])-array([[0, 0.5]]))*mult)
    KG = int(norm(array([[0.0001, 0.0001]])-array([[0.33333, 0.333333]]))*mult)
    path1 = array([zeros(GM), linspace(0.0001, g_length/2, GM)]).T
    path2 = array([linspace(0, g_length/3, MK), -1/2.1 *
                  linspace(0, g_length/3, MK)+g_length/2]).T
    path3 = array([linspace(g_length/3, 0.0001, KG),
                  linspace(g_length/3, 0.0001, KG)]).T
    path = concatenate([path1, path2, path3])
    print("Length of the path is ", len(path))
    sym = [0, GM, GM+MK, GM+MK+KG]
    label = ['Γ', 'M', 'K', 'Γ']
    return path, sym, label


def Symmetries(fstring):
    f = open(fstring, 'r')
    x = zeros(0)
    for i in f:
        x = append(x, float(i.split()[-1]))
    f.close()
    return x
