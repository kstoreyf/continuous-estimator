import numpy as np
import suave
from suave.io import read_lognormal_catalog
from suave.theory import DDsmu
from suave.utils import evaluate_xi
from suave.utils import qq_analytic

print("suave file:", suave.__file__)
print("suave version:", suave.__version__)
print("suave author:", suave.__author__)

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
nmubins = 1
mumax = 1.0
dd_res, dd_proj, _ = DDsmu(1, nthreads, r_edges, mumax, nmubins, x, y, z, 
                           boxsize=boxsize, periodic=periodic, proj_type=proj_type, nprojbins=nprojbins)
print("Corrfunc result:", dd_res)
print("Suave result:", dd_proj)

volume = boxsize**3
rr_ana, qq_ana = qq_analytic(rmin, rmax, nd, volume, nprojbins, proj_type, rbins=r_edges)

numerator = dd_proj - rr_ana
amps_ana, *_ = np.linalg.lstsq(qq_ana, numerator, rcond=None) # Use linalg.lstsq instead of actually computing inverse!
r_fine = np.linspace(rmin, rmax, 200)
xi_ana = evaluate_xi(amps_ana, r_fine, proj_type, rbins=r_edges)
print("Successfully ran suave on lognormal mock!")
