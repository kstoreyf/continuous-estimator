import os
import numpy as np
import time

import Corrfunc
from Corrfunc.theory.DDsmu import DDsmu
from Corrfunc.utils import compute_amps
from Corrfunc.utils import evaluate_xi
from Corrfunc.utils import qq_analytic

import read_lognormal as reader


def main():
    cat_tag = '_L750_N125k'
    #cat_tag = '_L750_n1e-4'
    proj_type = 'tophat'
    cf_tag = f'_{proj_type}_bw5'
    nbins = None
    binwidth = 5
    Nrealizations = 100
    overwrite = False
    
    result_dir = '../results/results_lognormal{}'.format(cat_tag)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    cat_dir = '../catalogs/lognormal'
    for Nr in range(Nrealizations):
    #for Nr in range(16, Nrealizations):
        print(f"Realization {Nr}")
        save_fn = f'{result_dir}/cf{cf_tag}{cat_tag}_rlz{Nr}.npy'
        if os.path.exists(save_fn) and not overwrite:
            continue
    
        print("Computing xi")
        fn = f'{cat_dir}/cat{cat_tag}_lognormal_rlz{Nr}.bin'
        Lx, Ly, Lz, N, data = reader.read(fn)
        x, y, z, vx, vy, vz = data.T
        print(N)
    
        rmin = 40.
        rmax = 200.
        #binwidth = 5
        assert bool(nbins) ^ bool(binwidth), "Can't set both nbins and binwidth!"
        if nbins:   
            binwidth = (rmax-rmin)/float(nbins) 
        r_edges = np.arange(rmin, rmax+binwidth, binwidth)
        print(r_edges)
        r_avg = 0.5*(r_edges[1:]+r_edges[:-1])
        nthreads = 24
        
        start = time.time()
        #res = Corrfunc.theory.xi(Lx, nthreads, r_edges, x, y, z)
        #res = np.array(res)
        #xi = [rr[3] for rr in res]
        r, xi = xi_proj_periodic(x, y, z, Lx, r_edges, proj_type)
        end = time.time()
        print("Time:", end-start, "s")
    
        #np.save(save_fn, [r_avg, xi, 'standard'])
        np.save(save_fn, [r, xi, proj_type])

def xi_proj_periodic(x, y, z, L, rbins, proj_type, projfn=None, nprojbins=None):
    mumax = 1
    nmubins = 1
    verbose = True
    periodic = True
    nthreads = 24
    nprojbins = len(rbins) - 1
    _, dd_proj, _ = DDsmu(1, nthreads, rbins, mumax, nmubins, x, y, z,
                proj_type=proj_type, nprojbins=nprojbins, projfn=projfn,
                verbose=verbose, boxsize=L, periodic=periodic)
    
    rmin = min(rbins)
    rmax = max(rbins)
    volume = L**3
    nrbins = len(rbins)-1
    nd = len(x)
    rr_ana, qq_ana = qq_analytic(rmin, rmax, nd, volume, nprojbins, nrbins, rbins, proj_type)

    rcont = np.linspace(rmin, rmax, 1000)
    amps_periodic_ana = np.matmul(np.linalg.inv(qq_ana), dd_proj) - 1
    xi_periodic_ana = evaluate_xi(nrbins, amps_periodic_ana, len(rcont), rcont, len(rbins)-1, rbins, proj_type, projfn=projfn)

    return rcont, xi_periodic_ana


if __name__=='__main__':
    main()
