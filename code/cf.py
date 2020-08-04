#!/usr/bin/env python
import os
import numpy as np
import time

import Corrfunc
from Corrfunc.theory.DDsmu import DDsmu
from Corrfunc.utils import compute_amps
from Corrfunc.utils import evaluate_xi
from Corrfunc.utils import qq_analytic
from Corrfunc.bases import spline
from Corrfunc.bases import bao

import read_lognormal as reader

from Corrfunc.theory.DD import DD
from Corrfunc.mocks.DDsmu_mocks import DDsmu_mocks
from nbodykit.lab import *
import nbodykit




def main():

    print("Corrfunc version:", Corrfunc.__version__, Corrfunc.__file__)

    L = 750
    cat_tag = f'_L{L}_n1e-4'
    kwargs = {}
    
    #proj = 'theory'
    #binwidth = 3
    #cf_tag = f"_{proj}_bw{binwidth}"

    proj = 'tophat'
    binwidth = 5
    cf_tag = f"_{proj}_bw{binwidth}_anatest"

    # proj = 'piecewise'
    # binwidth = 10
    # cf_tag = f"_{proj}_bw{binwidth}_hangstill"

    #proj = 'spline'
    #kwargs = {'order': 3}
    #binwidth = 10
    #cf_tag = f"_{proj}{kwargs['order']}_bw{binwidth}_xlim40-140"
    
    nbins = None
    Nrealizations = 1
    overwrite = True
    qq_analytic = True
    nthreads = 12

    result_dir = '../results/results_lognormal{}'.format(cat_tag)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    # define bins
    #rmin = 0.
    #rmax = 240. #!! 
    rmin = 36.0
    rmax = 156.0
    #rmin = 40.0
    #rmax = 140.0
    #rmax = 240.
    assert bool(nbins) ^ bool(binwidth), "Can't set both nbins and binwidth!"
    if nbins:   
        binwidth = (rmax-rmin)/float(nbins) 
    r_edges = np.arange(rmin, rmax+binwidth, binwidth)
    r_avg = 0.5*(r_edges[1:]+r_edges[:-1])

    proj_type, nprojbins, projfn = get_proj_parameters(proj, r_edges=r_edges, cf_tag=cf_tag, **kwargs)

    if not qq_analytic:
        # Load in randoms, will need for DR at least
        nx = 10
        rand_dir = '../catalogs/randoms'
        rand_fn = '{}/rand{}_{}x.dat'.format(rand_dir, cat_tag, nx)
        random = np.loadtxt(rand_fn)
        # check if already have RR, compute if not
        rr_qq_fn = f'{result_dir}/rr_qq{cf_tag}{cat_tag}.npy'
        if os.path.exists(rr_qq_fn):
            print(f"RR & QQ already computed, loading in from {rr_qq_fn}")
            rr_proj, qq_proj, _ = np.load(rr_qq_fn, allow_pickle=True)
        else:
            rr_proj, qq_proj = compute_rr_qq_numeric(rr_qq_fn,random, L, r_edges, nprojbins, proj, proj_type, projfn=projfn)

    cat_dir = '../catalogs/lognormal'
    for Nr in range(Nrealizations):
        print(f"Realization {Nr}")
        qq_tag = '' if qq_analytic else '_qqnum'
        save_fn = f'{result_dir}/cf{cf_tag}{qq_tag}{cat_tag}_rlz{Nr}.npy'
        if os.path.exists(save_fn) and not overwrite:
            print(f"Already exists! ({save_fn}) continuing")
            continue
    
        print("Computing xi")
        fn = f'{cat_dir}/cat{cat_tag}_lognormal_rlz{Nr}.bin'
        Lx, Ly, Lz, N, data = reader.read(fn)
        assert L==Lx and L==Ly and L==Lz, f"Box sizes don't align! L: {L}, Lx: {Lx}, Ly: {Ly}, Lz: {Lz}"
        x, y, z, vx, vy, vz = data.T
        print("N =",N)
        
        start = time.time()
        if "theory" in proj:
            xi = xi_theory_periodic(x, y, z, L, r_edges)
            r = r_avg
        else:
            if qq_analytic:
                r, xi = xi_proj_periodic_analytic(x, y, z, L, r_edges,
                 nprojbins, proj_type, projfn=projfn)
            else:
                rx, ry, rz = random.T
                r, xi = xi_proj_periodic_numeric(x, y, z, 
                rx, ry, rz, rr_proj, qq_proj, L, r_edges,
                 nprojbins, proj_type, projfn=projfn)
        end = time.time()

        #print("Done but not saved", save_fn)
        print(f"Saved result to {save_fn}")
        np.save(save_fn, [r, xi, proj])
        print("Time:", end-start, "s")
            

def xi_theory_periodic(x, y, z, L, r_edges, nthreads=24):
    res = Corrfunc.theory.xi(L, nthreads, r_edges, x, y, z)
    res = np.array(res)
    xi = [rr[3] for rr in res]
    return xi


def xi_proj_periodic_analytic(x, y, z, L, r_edges, nprojbins, proj_type, projfn=None, nthreads=24):
    mumax = 1.0
    nmubins = 1
    verbose = False # MAKE SURE THIS IS FALSE otherwise breaks
    periodic = True
    _, dd_proj, _ = DDsmu(1, nthreads, r_edges, mumax, nmubins, x, y, z,
                proj_type=proj_type, nprojbins=nprojbins, projfn=projfn,
                boxsize=L, periodic=periodic)

    rmin = min(r_edges)
    rmax = max(r_edges)
    volume = L**3
    nrbins = len(r_edges)-1
    nd = len(x)

    # works up to 100 thru here
    # hangs up to 15 when through next line
    print(rmin, rmax, nd, volume, nprojbins, nrbins, r_edges, proj_type, projfn)
    rr_ana, qq_ana = qq_analytic(rmin, rmax, nd, volume, nprojbins, nrbins, r_edges, proj_type, projfn=projfn)
    print(rr_ana)

    rcont = np.linspace(rmin, rmax, 1000)
    numerator = dd_proj - rr_ana
    amps_periodic_ana = np.linalg.solve(qq_ana, numerator)
    xi_periodic_ana = evaluate_xi(nrbins, amps_periodic_ana, len(rcont), rcont, nrbins, r_edges, proj_type, projfn=projfn)

    print("DD proj:", dd_proj)
    print(numerator)
    print(amps_periodic_ana)
    return rcont, xi_periodic_ana


def xi_proj_periodic_numeric(x, y, z, rx, ry, rz, rr_proj, qq_proj, L, r_edges, nprojbins, proj_type, projfn=None, nthreads=24):
    mumax = 1.0
    nmubins = 1
    verbose = False # MAKE SURE THIS IS FALSE otherwise breaks
    periodic = True

    print("computing dd...")
    _, dd_proj, _ = DDsmu(1, nthreads, r_edges, mumax, nmubins, x, y, z,
                proj_type=proj_type, nprojbins=nprojbins, projfn=projfn,
                boxsize=L, periodic=periodic)
    print("DD proj:", dd_proj)


    print("computing dr...")
    _, dr_proj, _ = DDsmu(0, nthreads, r_edges, mumax,               nmubins, x, y, z, X2=rx, Y2=ry, Z2=rz,
            proj_type=proj_type, nprojbins=nprojbins, projfn=projfn,
            boxsize=L, periodic=periodic)
    print("DR proj:", dr_proj)

    nd = len(x)
    nr = len(rx)
    rmin = min(r_edges)
    rmax = max(r_edges)
    r_cont = np.linspace(rmin, rmax, 1000)
    amps = compute_amps(nprojbins, nd, nd, nr, nr, dd_proj, dr_proj, dr_proj, rr_proj, qq_proj)
    xi_proj = evaluate_xi(nprojbins, amps, len(r_cont), r_cont, len(r_edges)-1, r_edges, proj_type, projfn=projfn)

    return r_cont, xi_proj



def compute_rr_qq_numeric(rr_qq_fn, random, L, r_edges, nprojbins, proj, proj_type, projfn=None, nthreads=24):
    print("computing rr...")

    rx, ry, rz = random.T

    mumax = 1.0
    nmubins = 1
    verbose = False # MAKE SURE THIS IS FALSE otherwise breaks
    periodic = True
    _, rr_proj, qq_proj = DDsmu(1, nthreads, r_edges, mumax,             nmubins, rx, ry, rz,
            proj_type=proj_type, nprojbins=nprojbins, projfn=projfn,
            boxsize=L, periodic=periodic)

    print("RR proj:", rr_proj)
    print("QQ proj:", qq_proj)   
    print(f"Saving RR and QQ to {rr_qq_fn}")
    np.save(rr_qq_fn, [rr_proj, qq_proj, proj])

    return rr_proj, qq_proj


def get_proj_parameters(proj, r_edges=None, cf_tag=None, **kwargs):
    proj_type = proj
    projfn = None
    if proj=="tophat" or proj_type=="piecewise":
        nprojbins = len(r_edges)-1
    elif proj=='spline':
        nprojbins = len(r_edges)-1
        proj_type = 'generalr'
        projfn = f'../tables/bases{cf_tag}.dat'
        spline.write_bases(r_edges[0], r_edges[-1], len(r_edges)-1, projfn, **kwargs)
    elif proj=='bao':
        proj_type = 'generalr'
        projfn = f'../tables/bases{cf_tag}.dat'
        nprojbins, _ = bao.write_bases(r_edges[0], r_edges[-1], projfn, **kwargs) 
    elif proj=='theory':
        proj_type = None
        nprojbins = None
    else:
      raise ValueError("Proj type {} not recognized".format(proj_type))
    return proj_type, nprojbins, projfn


if __name__=='__main__':
    main()
