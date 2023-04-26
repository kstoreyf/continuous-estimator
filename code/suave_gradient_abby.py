import numpy as np
import scipy
from scipy.interpolate import BSpline
from scipy.interpolate import _bspl
import matplotlib
import matplotlib.pyplot as plt

import Corrfunc
from Corrfunc.bases import bao_bases
from Corrfunc.io import read_lognormal_catalog
from Corrfunc.theory.DDsmu import DDsmu
from Corrfunc.utils import compute_amps
from Corrfunc.utils import evaluate_xi
from Corrfunc.utils import convert_3d_counts_to_cf
from colossus.cosmology import cosmology

import globals

globals.initialize_path()
path_to_dir = globals.path_to_dir

# define cosmo_bases function
def cosmo_bases(rmin, rmax, projfn, cosmo_base=None, ncont=2000, 
              redshift=0.0, bias=1.0):
    if cosmo_base is None:
        print("cosmo_base not provided, defaulting to Planck 2015 cosmology ('planck15')")
        cosmo_base = cosmology.setCosmology('planck15')

    cf = cosmo_base.correlationFunction

    def cf_model(r):
        return bias * cf(r, z=redshift)

    rcont = np.linspace(rmin, rmax, ncont)
    bs = cf_model(rcont)

    nbases = 1
    bases = np.empty((ncont, nbases+1))
    bases[:,0] = rcont
    bases[:,1] = np.array(bs)

    np.savetxt(projfn, bases)
    ncomponents = bases.shape[1]-1
    return bases

# load in data
m = 0.75
b = 0.5
grad_dim = 1
boxsize = np.load(f"{path_to_dir}boxsize.npy")

mock_data = np.load(f"{path_to_dir}gradient_mocks/{grad_dim}D/mocks/grad_mock_m-{m}-L_b-{b}.npy")
mock_data += boxsize/2

x = mock_data[:,0]
y = mock_data[:,1]
z = mock_data[:,2]

nd = len(x)

# random set
nr = 1*nd
xr = np.random.rand(nr)*float(boxsize)
yr = np.random.rand(nr)*float(boxsize)
zr = np.random.rand(nr)*float(boxsize)

# parameters for suave
rmin = 40.0
rmax = 140.0
ntopbins = 10
r_edges = np.linspace(rmin, rmax, ntopbins+1) 
ncont = 2000
r_fine = np.linspace(rmin, rmax, ncont)

nmubins = 1
mumax = 1.0
periodic = False
nthreads = 1

proj_type = 'gradient'
weight_type = 'pair_product_gradient'
projfn = 'cosmo_basis.dat'

# set basis
bases = cosmo_bases(rmin, rmax, projfn)
ncomponents = 4*(bases.shape[1]-1)
r = bases[:,0]
base_vals = bases[:,1]

# weights
loc_pivot = [boxsize/2., boxsize/2., boxsize/2.]
weights = np.array([np.ones(len(x)), x-loc_pivot[0], y-loc_pivot[1], z-loc_pivot[2]])
weights_r = np.array([np.ones(len(xr)), xr-loc_pivot[0], yr-loc_pivot[1], zr-loc_pivot[2]])

# run the pair counts
dd_res, dd_proj, _ = DDsmu(1, nthreads, r_edges, mumax, nmubins, x, y, z, weights1=weights, 
                           proj_type=proj_type, ncomponents=ncomponents, projfn=projfn, 
                           periodic=periodic, weight_type=weight_type)
print("DD:", np.array(dd_proj))

dr_res, dr_proj, _ = DDsmu(0, nthreads, r_edges, mumax, nmubins, x, y, z, weights1=weights, 
                           X2=xr, Y2=yr, Z2=zr, weights2=weights_r, 
                           proj_type=proj_type, ncomponents=ncomponents, projfn=projfn, 
                           periodic=periodic, weight_type=weight_type)
print("DR:", np.array(dr_proj))

rr_res, rr_proj, qq_proj = DDsmu(1, nthreads, r_edges, mumax, nmubins, xr, yr, zr, weights1=weights_r, 
                                 proj_type=proj_type, ncomponents=ncomponents, projfn=projfn, 
                                 periodic=periodic, weight_type=weight_type)
print("RR:", np.array(rr_proj))

amps = compute_amps(ncomponents, nd, nd, nr, nr, dd_proj, dr_proj, dr_proj, rr_proj, qq_proj)
xi_proj = evaluate_xi(amps, r_fine, proj_type, rbins=r_edges, projfn=projfn)

# extract the standard binned values
dd = np.array([x['npairs'] for x in dd_res], dtype=float)
dr = np.array([x['npairs'] for x in dr_res], dtype=float)
rr = np.array([x['npairs'] for x in rr_res], dtype=float)
xi_standard = convert_3d_counts_to_cf(nd, nd, nr, nr, dd, dr, dr, rr)
r_avg = 0.5*(r_edges[:-1] + r_edges[1:])

# recovered gradient
print("amps = ", amps)
w_cont = amps[1:]/amps[0]
w_cont_norm = np.linalg.norm(w_cont)
w_cont_hat = w_cont/w_cont_norm
print("w_cont = ", w_cont)
print(f"||w_cont|| = {w_cont_norm:.6f}")
b_guess = 0.5
m_recovered_perL = w_cont_norm*b_guess*boxsize

# expected gradient (only in x direction)
grad_expected = np.array([m/(b*L),0,0])
print("expected gradient (m_input/b_input)w_hat =", grad_expected)

# save recovered values!
recovered_arr = [w_cont, b_guess, m_recovered_perL]
np.save(recovered_arr, f"{path_to_dir}gradient_mocks/{grad_dim}D/suave/recovered_grad_m-{m}-L_b-{b}_suave")

print(f"If we assume an initial b={b_guess}, this gives m = {m_recovered_perL:.4f}/L") 

# plot correlation functions along the gradient axis
fig = plt.figure(figsize=(5,4))
ax = plt.gca()

v_min = -boxsize/2.
v_max = boxsize/2.
vs_norm = matplotlib.colors.Normalize(vmin=v_min, vmax=v_max)
cmap = matplotlib.cm.get_cmap('cool')
nvs = 10
vs = np.linspace(v_min, v_max, nvs)

for i, v in enumerate(vs):
    
    loc = loc_pivot + v*w_cont_hat
    if i==len(vs)-1:
        print(loc)
    weights1 = np.array(np.concatenate(([1.0], loc-loc_pivot)))
    weights2 = weights1 #because we just take the average of these and want to get this back
    
    xi_loc = evaluate_xi(amps, r_fine, proj_type, projfn=projfn, 
                     weights1=weights1, weights2=weights2, weight_type=weight_type)    
    
    p = plt.plot(r_fine, xi_loc, color=cmap(vs_norm(v)), lw=0.5)
    
    
sm = plt.cm.ScalarMappable(cmap=cmap, norm=vs_norm)
cbar = plt.colorbar(sm)
cbar.set_label(r'$v \,\, (\mathbf{x} = v\hat{e}_\mathrm{gradient} + \mathbf{x}_\mathrm{pivot})$', rotation=270, labelpad=12)
ax.axhline(0, color='grey', lw=0.5)
ax.set_xlabel(r'separation $r$ ($h^{-1}\,$Mpc)')
ax.set_ylabel(r'$\xi(r)$')
ax.set_title(f"Recovered Gradient, m={m}, b={b}")

#fig.savefig(f"{path_to_dir}gradient_mocks/{grad_dim}D/suave/recovered_grad_m-{m}-L_b-{b}_suave.png")
fig.savefig(f"recovered_grad_m-{m}-L_b-{b}_suave.png")

plt.show()
