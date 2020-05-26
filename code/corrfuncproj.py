import numpy as np
import pandas as pd
import time

from Corrfunc.mocks.DDrppi_mocks import DDrppi_mocks
from Corrfunc.mocks.DDsmu_mocks import DDsmu_mocks
from Corrfunc.utils import convert_rp_pi_counts_to_wp
from Corrfunc.utils import compute_amps
from Corrfunc.utils import evaluate_xi

#from Corrfunc.mocks.DDsmu_mocks import convert_3d_proj_counts_to_amplitude
#from Corrfunc._countpairs_mocks import countpairs_s_mu_mocks as s_mu_mocks
#from Corrfunc._countpairs_mocks import convert_3d_proj_counts_to_amplitude as compute_amplitude

from astropy.cosmology import LambdaCDM



def counts_smu(ra_data, dec_data, z_data, ra_rand, dec_rand, z_rand, rpbins,
            losmax, cosmo, nproc=1, weights_data=None, weights_rand=None,
            comoving=False, proj_type=None, nprojbins=None, projfn=None):

    assert(len(ra_data)==len(dec_data) and len(ra_data)==len(z_data))
    assert(len(ra_rand)==len(dec_rand) and len(ra_rand)==len(z_rand))

    nd1 = len(ra_data)
    nr1 = len(ra_rand)
    nd2 = nd1
    nr2 = nr1

    if nprojbins is None:
        nprojbins = len(rpbins)

    if weights_data is None:
        weights_data = np.ones(nd1)
    if weights_rand is None:
        weights_rand = np.ones(nr1)

    cosmology = 1
    nthreads = nproc
    verbose = False
    weight_type = 'pair_product'
    isa = 'fallback'

    mumax = losmax
    nmubins = 1

    print('Computing DD pairs')
    start = time.time()
    dd_res_corrfunc, dd_proj, dd_projt = DDsmu_mocks(1, cosmology, nthreads, mumax, nmubins, rpbins, ra_data, dec_data, z_data,
                                   weights1=weights_data, is_comoving_dist=comoving, verbose=verbose,
                                   weight_type=weight_type, isa=isa, proj_type=proj_type, nprojbins=nprojbins, projfn=projfn)
    end = time.time()
    print("Time DD pairs:", end - start)
    print("DD:", dd_proj)
    #print(dd_res_corrfunc)

    #TODO: allow cross-correlations
    print('Computing DR pairs')
    start = time.time()
    dr_res_corrfunc, dr_proj, dr_projt = DDsmu_mocks(0, cosmology, nthreads, mumax, nmubins, rpbins, ra_data, dec_data, z_data,
                                        RA2=ra_rand, DEC2=dec_rand, CZ2=z_rand, weights1=weights_data,
                                   weights2=weights_rand, is_comoving_dist=comoving, verbose=verbose,
                                   weight_type=weight_type, isa=isa, proj_type=proj_type, nprojbins=nprojbins, projfn=projfn)
    end = time.time()
    print("Time DR pairs:", end - start)
    print("DR:", dr_proj)
    #print(dr_res_corrfunc)

    print('Computing RR pairs')
    start = time.time()
    rr_res_corrfunc, rr_proj, rr_projt = DDsmu_mocks(1, cosmology, nthreads, mumax, nmubins, rpbins, ra_rand, dec_rand, z_rand,
                                   weights1=weights_rand, is_comoving_dist=comoving, verbose=verbose,
                                   weight_type=weight_type, isa=isa, proj_type=proj_type, nprojbins=nprojbins, projfn=projfn)
    end = time.time()
    print("Time RR pairs:", end - start)
    print("RR:", rr_proj)
    #print("QQ:", rr_projt)
    #print(rr_res_corrfunc)

    return dd_proj, dr_proj, rr_proj, rr_projt, \
           dd_res_corrfunc, dr_res_corrfunc, rr_res_corrfunc


def counts_smu_auto(ra, dec, z, rpbins,
            losmax, cosmo, nproc=1, weights=None,
            comoving=False, proj_type=None, nprojbins=None, projfn=None, qq=True):

    assert(len(ra)==len(dec) and len(ra)==len(z))

    nd1 = len(ra)

    if nprojbins is None:
        nprojbins = len(rpbins)

    if weights is None:
        weights = np.ones(nd1)

    cosmology = 1
    nthreads = nproc
    verbose = False
    weight_type = 'pair_product'
    isa = 'fallback'

    mumax = losmax
    nmubins = 1

    print('Computing auto pairs')
    start = time.time()
    dd_res_corrfunc, dd_proj, dd_projt = DDsmu_mocks(1, cosmology, nthreads, mumax, nmubins, rpbins, ra, dec, z,
                                   weights1=weights, is_comoving_dist=comoving, verbose=verbose,
                                   weight_type=weight_type, isa=isa, proj_type=proj_type, nprojbins=nprojbins, projfn=projfn)
    end = time.time()
    print("Time auto pairs:", end - start)
    print("pairs:", dd_proj)
    if qq:
        return dd_proj, dd_res_corrfunc, dd_projt
    else:
        return dd_proj, dd_res_corrfunc 


def counts_smu_cross(ra_data, dec_data, z_data, ra_rand, dec_rand, z_rand, rpbins,
            losmax, cosmo, nproc=1, weights_data=None, weights_rand=None,
            comoving=False, proj_type=None, nprojbins=None, projfn=None):

    assert(len(ra_data)==len(dec_data) and len(ra_data)==len(z_data))
    assert(len(ra_rand)==len(dec_rand) and len(ra_rand)==len(z_rand))

    nd1 = len(ra_data)
    nr1 = len(ra_rand)
    nd2 = nd1
    nr2 = nr1

    if nprojbins is None:
        nprojbins = len(rpbins)

    if weights_data is None:
        weights_data = np.ones(nd1)
    if weights_rand is None:
        weights_rand = np.ones(nr1)

    cosmology = 1
    nthreads = nproc
    verbose = False
    weight_type = 'pair_product'
    isa = 'fallback'

    mumax = losmax
    nmubins = 1

    print('Computing cross pairs')
    start = time.time()
    dr_res_corrfunc, dr_proj, dr_projt = DDsmu_mocks(0, cosmology, nthreads, mumax, nmubins, rpbins, ra_data, dec_data, z_data,
                                        RA2=ra_rand, DEC2=dec_rand, CZ2=z_rand, weights1=weights_data,
                                   weights2=weights_rand, is_comoving_dist=comoving, verbose=verbose,
                                   weight_type=weight_type, isa=isa, proj_type=proj_type, nprojbins=nprojbins, projfn=projfn)
    end = time.time()
    print("Time cross pairs:", end - start)
    print("pairs:", dr_proj)
    return dr_proj, dr_res_corrfunc
