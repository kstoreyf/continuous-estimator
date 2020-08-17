import numpy as np
import math
import glob

import nbodykit
from nbodykit.lab import *



# cosmo and cosmo_fid are nbodykit cosmology objects
def compute_alpha(z, cosmo, cosmo_fid):

    cosmoa = cosmo.to_astropy()
    cosmoa_fid = cosmo_fid.to_astropy()

    da = cosmoa.angular_diameter_distance(z)
    da_fid = cosmoa_fid.angular_diameter_distance(z)

    H = cosmoa.H(z)
    H_fid = cosmoa_fid.H(z)

    rs = radius_sound_horizon(cosmo.Omega0_cdm, cosmo.Omega0_b, cosmo.h)
    rs_fid = radius_sound_horizon(cosmo_fid.Omega0_cdm, cosmo_fid.Omega0_b, cosmo_fid.h)

    if z>0:
        alpha = (da/da_fid)**(2./3.) * (H_fid/H)**(1./3.) * rs_fid/rs
    else:
        #ignore angular diam dist?? 
        alpha = (H_fid/H)**(1./3.) * rs_fid/rs

    print("Cosmo: r_s:", rs, "D_A:", da, "H:", H)
    print("Fiducial: r_s:", rs_fid, "D_A:", da_fid, "H:", H_fid)
    print("alpha:", alpha)

    return alpha


def get_cosmo(cosmo_name):
    cosmo_names = ['planck', 'b17']
    if cosmo_name=='planck':
        cosmo = nbodykit.cosmology.Planck15
    # Beutler et al 2017 (& Ross et al 2017), as used in Barry paper (Hinton 2019)
    elif cosmo_name=='b17':
        Omega0_cdm = 0.31
        Omega0_b = 0.04814
        h0 = 0.676
        ns = 0.96
        cosmo = nbodykit.cosmology.Cosmology(h=h0, Omega0_b=Omega0_b, Omega0_cdm=Omega0_cdm, n_s=ns)
    else:
        print("Name not recognized! Must be one of the following:", cosmo_names)
        return      
    return cosmo


# Compute the Eisenstein and Hu 1998 value for the sound horizon
# from Barry (Hinton 2019): https://github.com/Samreay/Barry/blob/c34645ca56bed749d1480ba0e40d495fe5a96574/barry/cosmology/power_spectrum_smoothing.py
def radius_sound_horizon(om, ob, h0):

    # Fitting parameters
    b1 = 0.313
    b2 = -0.419
    b3 = 0.607
    b4 = 0.674
    b5 = 0.238
    b6 = 0.223
    a1 = 1291.0
    a2 = 0.251
    a3 = 0.659
    a4 = 0.828
    theta = 2.725 / 2.7  # Normalised CMB temperature

    obh2 = ob * h0 * h0
    omh2 = om * h0 * h0

    z_eq = 2.5e4 * omh2 / (theta ** 4)
    k_eq = 7.46e-2 * omh2 / (theta ** 2)

    zd1 = b1 * omh2 ** b2 * (1.0 + b3 * omh2 ** b4)
    zd2 = b5 * omh2 ** b6
    z_d = a1 * (omh2 ** a2 / (1.0 + a3 * omh2 ** a4)) * (1.0 + zd1 * obh2 ** zd2)

    R_eq = 3.15e4 * obh2 / (z_eq * theta ** 4)
    R_d = 3.15e4 * obh2 / (z_d * theta ** 4)

    s = 2.0 / (3.0 * k_eq) * math.sqrt(6.0 / R_eq) * math.log((math.sqrt(1.0 + R_d) + math.sqrt(R_d + R_eq)) / (1.0 + math.sqrt(R_eq)))

    return s


def calc_alpha(cat_tag, cf_tag, realizations=range(100)):
    cat_dir = '../catalogs'
    result_dir = '../results/results_lognormal{}'.format(cat_tag)
    
    xis = []
    n_converged = 0
    alphas = []
    for Nr in realizations:

        if 'baoiter' in cf_tag:
            fn_pattern = f"cf{cf_tag}_converged_*{cat_tag}_rlz{Nr}.npy"
            for cf_fn in glob.glob(f'{result_dir}/{fn_pattern}'):
                #print(cf_fn)
                r_avg, xi, amps, _, extra_dict = np.load(cf_fn, allow_pickle=True)
                #print("C:", amps[4])
                alphas.append(extra_dict ['alpha_result'])
                n_converged +=1
                break #should only be 1 match; but probs better way to do this
        else:
            cf_fn = '{}/cf{}{}_rlz{}.npy'.format(result_dir, cf_tags[i], cat_tag, Nr)
            r_avg, xi, amps = np.load(cf_fn, allow_pickle=True)

    print(cf_tag)
    print(f"Found {n_converged} converged BAO cfs ({len(realizations)-n_converged} not converged)")    
    print("alpha_mean:", np.mean(alphas))
    print("alpha_std:", np.std(alphas))
    return alphas

