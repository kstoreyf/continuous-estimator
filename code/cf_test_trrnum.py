import Corrfunc
from Corrfunc import bases, theory, utils
import numpy as np

#proj = 'tophat'
proj = 'spline'
print(proj)

boxsize = 200.0
nd = 20
nr = 3*nd

x = np.random.uniform(0, boxsize, nd)
y = np.random.uniform(0, boxsize, nd)
z = np.random.uniform(0, boxsize, nd)

x_rand = np.random.uniform(0, boxsize, nr)
y_rand = np.random.uniform(0, boxsize, nr)
z_rand = np.random.uniform(0, boxsize, nr)

rmin, rmax, ncomponents = 40.0, 60.0, 8
r_edges = np.linspace(rmin, rmax, ncomponents+1)
nmubins = 1
mumax = 1.0
periodic = True
nthreads = 1

if proj=='tophat':
    proj_type = 'tophat'
    projfn = None
elif proj=='spline':
    proj_type = 'generalr'
    kwargs = {'order': 3}
    projfn = 'cubic_spline.dat'
    bases = bases.spline_bases(rmin, rmax, projfn, ncomponents, ncont=2000, **kwargs)
else:
    print(f"Proj {proj} not setup here!")
    exit(1)

dd_res, dd_proj, _ = theory.DDsmu(1, nthreads, r_edges, mumax, nmubins, x, y, z, 
                                  boxsize=boxsize, periodic=periodic, proj_type=proj_type, 
                                  ncomponents=ncomponents, projfn=projfn)
dr_res, dr_proj, _ = theory.DDsmu(0, nthreads, r_edges, mumax, nmubins, x, y, z, 
                                  X2=x_rand, Y2=y_rand, Z2=z_rand, 
                                  boxsize=boxsize, periodic=periodic, proj_type=proj_type, 
                                  ncomponents=ncomponents, projfn=projfn)
rr_res, rr_proj, trr_proj = theory.DDsmu(1, nthreads, r_edges, mumax, nmubins, 
                                         x_rand, y_rand, z_rand, boxsize=boxsize,
                                         periodic=periodic, proj_type=proj_type,
                                         ncomponents=ncomponents, projfn=projfn)

amps = utils.compute_amps(ncomponents, nd, nd, nr, nr, dd_proj, dr_proj, dr_proj, rr_proj, trr_proj)
print('amps:', amps)

if proj=='spline':
    r_edges = None
r_fine = np.linspace(rmin, rmax, 10)
print(amps, r_fine, proj_type, r_edges, projfn)
xi_proj = utils.evaluate_xi(amps, r_fine, proj_type, rbins=r_edges, projfn=projfn)
print('xi_proj:', xi_proj)
