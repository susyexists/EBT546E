#!/bin/bash 
#SBATCH -p bansil
#SBATCH -n 64
#SBATCH -N 1
#SBATCH -J EBT

source /shared/centos7/intel/oneapi/2022.1.0/setvars.sh
QE=/work/bansil/s.sevim/software/q-e/bin

mpirun -np 30 $QE/pw.x  < scf.in > scf.out
mpirun -np 30 $QE/ph.x  < ph.in > ph.out
$QE/q2r.x  < q2r.in > q2r.out
$QE/matdyn.x  < matdyn.in > matdyn.out
$QE/plotband.x < plotband.in > plotband.out

module load anaconda3