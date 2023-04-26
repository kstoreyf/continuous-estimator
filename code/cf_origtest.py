import numpy as np
from Corrfunc.theory.DD import DD

L = 500.0
N = 1000
x = np.random.rand(N)*float(L)
y = np.random.rand(N)*float(L)
z = np.random.rand(N)*float(L)

binwidth = 10.0
rmin = 40.0
rmax = 150.0
r_edges = np.arange(rmin, rmax+binwidth, binwidth)

nthreads = 1
periodic = False

dd_res = DD(1, nthreads, r_edges, x, y, z, periodic=periodic)
print(dd_res)
