#!/usr/bin/env python
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import os

from nbodykit.lab import *
import nbodykit
import Corrfunc
from Corrfunc.theory.DD import DD
from Corrfunc.theory.DDsmu import DDsmu
from Corrfunc.mocks.DDsmu_mocks import DDsmu_mocks
from Corrfunc.utils import evaluate_xi
from Corrfunc.utils import qq_analytic
from Corrfunc.bases import spline
from Corrfunc.bases import bao
from Corrfunc.utils import compute_amps

import read_lognormal as reader


def main():
    print("Go!")
    nrealizations = 1
    #L = 750
    L = 450.0
    N = 10000 # number of points in mock

    #proj = 'spline'
    #kwargs = {'order': 3}
    #binwidth = 10
    #cf_tag = f"_{proj}{kwargs['order']}_bw{binwidth}"

    proj = 'tophat'
    kwargs = {}
    binwidth = 5
    cf_tag = f"_{proj}_bw{binwidth}_anatest"

    #proj = None
    #kwargs = {}
    #binwidth = 10
    #cf_tag = f"_{proj}_bw{binwidth}"

    rmin = 40.0
    rmax = 150.0
    r_edges = np.arange(rmin, rmax+binwidth, binwidth)
    rmax = max(r_edges)
    r_avg = 0.5*(r_edges[1:]+r_edges[:-1])
    
    proj_type, nprojbins, projfn = get_proj_parameters(proj, r_edges=r_edges, cf_tag=cf_tag, **kwargs)

    nthreads = 24

    for Nr in range(nrealizations):
        print(f"Realization {Nr}")

        # Generate cubic mock
        x = np.random.rand(N)*float(L)
        y = np.random.rand(N)*float(L)
        z = np.random.rand(N)*float(L)
        pos = np.array([x, y, z]).T

        print("Run DDsmu, {} basis".format(proj))
        nmubins = 1
        mumax = 1.0
        periodic=True
        dd_res, dd_proj, _ = DDsmu(1, nthreads, r_edges, mumax, nmubins, x, y, z, proj_type=proj_type, nprojbins=nprojbins, projfn=projfn, periodic=periodic, boxsize=L, verbose=True)
        print("DD:", dd_proj)

        if proj_type is None:
            print(dd_res)
            continue

        print("Run qq_analytic")
        rmin = min(r_edges)
        rmax = max(r_edges)
        nd = len(x)
        volume = float(L**3)
        nrbins = len(r_edges)-1 
        rr_ana, qq_ana = qq_analytic(rmin, rmax, nd, volume, nprojbins, nrbins, r_edges, proj_type, projfn=projfn)
        print("RR_ana:", rr_ana)

        print("Run evaluate_xi")
        rcont = np.linspace(rmin, rmax, 1000)
        numerator = dd_proj - rr_ana
        amps_periodic_ana = np.linalg.solve(qq_ana, numerator)
        print("amplitudes:", amps_periodic_ana)
        xi_periodic_ana = evaluate_xi(nprojbins, amps_periodic_ana, len(rcont), rcont, nrbins, r_edges, proj_type, projfn=projfn)
        print("evaluate_xi done")





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
        print(kwargs)
        nprojbins, _ = bao.write_bases(r_edges[0], r_edges[-1], projfn, **kwargs) 
    elif proj==None:
        nprojbins = None
    else:
      raise ValueError("Proj type {} not recognized".format(proj_type))
    return proj_type, nprojbins, projfn


if __name__=='__main__':
    main()
