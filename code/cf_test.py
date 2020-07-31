import numpy as np
import Corrfunc
from Corrfunc.io import read_lognormal_catalog
from Corrfunc.theory import DDsmu

print("smoothcorrfunc file:", Corrfunc.__file__)
print("smoothcorrfunc version:", Corrfunc.__version__)
print("smoothcorrfunc author:", Corrfunc.__author__)

x, y, z = read_lognormal_catalog(n='1e-4')
boxsize = 750.0
nd = len(x)
print("Read in lognormal catalog, number of data points:",nd)

rmin = 40.0
rmax = 150.0
nbins = 11
r_edges = np.linspace(rmin, rmax, nbins+1)

periodic = True
nthreads = 1
proj_type = 'tophat'
nprojbins = nbins
dd_res, dd_proj, _ = DDsmu(1, nthreads, r_edges, mumax, nmubins, x, y, z, 
                           boxsize=boxsize, periodic=periodic, proj_type=proj_type, nprojbins=nprojbins)
print("Corrfunc result:", dd_res)
print("Smoothcorrfunc result:", dd_proj)
