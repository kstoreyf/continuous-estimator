import numpy as np
import Corrfunc
from Corrfunc.io import read_lognormal_catalog
from Corrfunc.theory import DDsmu
from Corrfunc.utils import evaluate_xi
from Corrfunc.utils import trr_analytic

print("corrfunc file:", Corrfunc.__file__)
print("corrfunc version:", Corrfunc.__version__)
print("corrfunc author:", Corrfunc.__author__)

x, y, z = read_lognormal_catalog(n='5e-5')
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
ncomponents = nbins
nmubins = 1
mumax = 1.0
dd_res, dd_proj, _ = DDsmu(1, nthreads, r_edges, mumax, nmubins, x, y, z, 
                           boxsize=boxsize, periodic=periodic, proj_type=proj_type, ncomponents=ncomponents)
print("Corrfunc result:", dd_res)
print("Smoothcorrfunc result:", dd_proj)

volume = boxsize**3
rr_ana, trr_ana = trr_analytic(rmin, rmax, nd, volume, ncomponents, proj_type, rbins=r_edges)

numerator = dd_proj - rr_ana
amps_ana, *_ = np.linalg.lstsq(trr_ana, numerator, rcond=None) # Use linalg.lstsq instead of actually computing inverse!
r_fine = np.linspace(rmin, rmax, 200)
xi_ana = evaluate_xi(amps_ana, r_fine, proj_type, rbins=r_edges)
print(xi_ana)
print("Successfully ran suave on lognormal mock!")
