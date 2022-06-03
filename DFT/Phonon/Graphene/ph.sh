#!/bin/bash 


mpirun -np 30 pw.x  < scf.in > scf.out
mpirun -np 30 ph.x  < ph.in > ph.out
q2r.x  < q2r.in > q2r.out
matdyn.x  < matdyn.in > matdyn.out
plotband.x < plotband.in > plotband.out

python ph_plot.py