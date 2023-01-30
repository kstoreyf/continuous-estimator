import numpy as np
import math
import glob

from scipy.optimize import curve_fit
#import nbodykit
#from nbodykit.lab import *



# cosmo and cosmo_fid are nbodykit cosmology objects
def compute_alpha(z, cosmo_truth, cosmo_fid):

    cosmoa_truth = cosmo_truth.to_astropy()
    cosmoa_fid = cosmo_fid.to_astropy()

    da = cosmoa_truth.angular_diameter_distance(z) #returns D_A in Mpc
    da_fid = cosmoa_fid.angular_diameter_distance(z)

    H = cosmoa_truth.H(z)
    H_fid = cosmoa_fid.H(z)

    
    #Omega0_m_truth = cosmo_truth.Omega0_cdm + cosmo_truth.Omega0_b
    #Omega0_m_fid = cosmo_fid.Omega0_cdm + cosmo_fid.Omega0_b
    # this is the radius of the sound horizon at the baryon drag epoch, so don't pass z
    #print(cosmo_truth.Omega0_m, cosmo_truth.Omega0_cdm, cosmo_truth.Omega0_b, cosmo_truth.h)
    #print(cosmo_fid.Omega0_m, cosmo_fid.Omega0_cdm, cosmo_fid.Omega0_b, cosmo_fid.h)
    rs = compute_radius_sound_horizon(cosmo_truth.Omega0_m, cosmo_truth.Omega0_b, cosmo_truth.h)
    rs_fid = compute_radius_sound_horizon(cosmo_fid.Omega0_m, cosmo_fid.Omega0_b, cosmo_fid.h)

    rs_col = compute_rs(cosmo_truth.Omega0_m, cosmo_truth.Omega0_b, cosmo_truth.h)
    rs_fid_col = compute_rs(cosmo_fid.Omega0_m, cosmo_fid.Omega0_b, cosmo_fid.h)

    if z>0.0:
        alpha = (da/da_fid)**(2./3.) * (H_fid/H)**(1./3.) * (rs_fid/rs)
        alpha_col = (da/da_fid)**(2./3.) * (H_fid/H)**(1./3.) * (rs_fid_col/rs_col)
    else:
        #ignore angular diam dist?? 
        alpha = (H_fid/H)**(1./3.) * (rs_fid/rs)

    print("Truth: r_s:", rs, "D_A:", da, "H:", H)
    print("Fiducial: r_s:", rs_fid, "D_A:", da_fid, "H:", H_fid)
    #print("rs_truth_col", rs_col, "rs_fid_col", rs_fid_col)
    print("alpha:", alpha)
    #print("alpha_col:", alpha_col)

    return alpha


def compute_da(z, cosmo):
    cosmoa = cosmo.to_astropy()
    da = cosmoa.angular_diameter_distance(z)
    return da

def compute_H(z, cosmo):
    cosmoa = cosmo.to_astropy()
    H = cosmoa.H(z)
    return H


def get_cosmo(cosmo_name):
    cosmo_names = ['planck', 'b17', 'wmap9', 'patchy']
    if cosmo_name=='planck':
        cosmo = nbodykit.cosmology.Planck15
    elif cosmo_name=='wmap9':
        cosmo = nbodykit.cosmology.WMAP9
    # Kitaura 2016; PATCHY mocks as used in BOSS, from Big Multidark sim 
    elif cosmo_name=='patchy':
        h0 = 0.6777
        m_ncdm = [] #pass empty list for no massive neutrinos

        Omega0_m = 0.307115
        Omega0_b = 0.048206
        Omega0_cdm = round(Omega0_m-Omega0_b, 6)

        ns = 0.9611
        sigma_8 = 0.8288
        cosmo = nbodykit.cosmology.Cosmology(h=h0, Omega0_b=Omega0_b, Omega0_cdm=Omega0_cdm, 
                                                n_s=ns, m_ncdm=m_ncdm)
        cosmo = cosmo.match(sigma8=sigma_8) 
    # Beutler et al 2017 (& Ross et al 2017), as used in Barry paper (Hinton 2019)
    elif cosmo_name=='b17':
        h0 = 0.676
        m_ncdm = [0.06]
        rho_crit = 93.14 * h0**2 #in eV
        Omega0_nu = np.sum(m_ncdm) / rho_crit

        Omega0_m = 0.31 
        Omega0_b = 0.022/(h0**2)
        Omega0_cdm = round(Omega0_m-Omega0_b-Omega0_nu, 6)
        
        ns = 0.96
        sigma_8 = 0.824
        cosmo = nbodykit.cosmology.Cosmology(h=h0, Omega0_b=Omega0_b, Omega0_cdm=Omega0_cdm, 
                                                n_s=ns, m_ncdm=m_ncdm)
        cosmo = cosmo.match(sigma8=sigma_8)
    else:
        print("Name not recognized! Must be one of the following:", cosmo_names)
        return      
    print(cosmo.Omega0_cdm, cosmo.Omega0_m, cosmo.Omega0_b, cosmo.h, cosmo.n_s, cosmo.m_ncdm, cosmo.sigma8, cosmo.N_ur)
    return cosmo


def nbodykit_to_lognormal_galaxies(cosmo):
    print("oc0h2 =", round(cosmo.Omega0_cdm * cosmo.h**2, 6), '  # \Omega_c h^2')
    print("mnu =", float(sum(cosmo.m_ncdm)), r'  # \Sigma m_{\nu} total neutrino mass')
    print("ns =", cosmo.n_s)
    print("lnAs =", round(np.log(cosmo.A_s/1e-10), 6))
    print("ob0h2 =", round(cosmo.Omega0_b * cosmo.h**2, 6), '  #\Omega_baryon h^2')
    print("h0 =", cosmo.h, '  # H0/100')
    print("w = -1")
    print("run = 0.0   # running index of pk")


# The sound horizon at recombination in Mpc (Mpc to be consistent with D_A and H)
# From Colossus: https://bitbucket.org/bdiemer/colossus/src/master/colossus/cosmology/cosmology.py
def compute_rs(om, ob, h0):
    obh2 = ob * h0 * h0
    omh2 = om * h0 * h0
    r_s = 44.5 * np.log(9.83 / omh2) / np.sqrt(1.0 + 10.0 * obh2**0.75)
    return r_s
    

# Compute the Eisenstein and Hu 1998 value for the sound horizon
# from Barry (Hinton 2019): https://github.com/Samreay/Barry/blob/c34645ca56bed749d1480ba0e40d495fe5a96574/barry/cosmology/power_spectrum_smoothing.py
def compute_radius_sound_horizon(om, ob, h0):

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


def get_alphas(cat_tag, cf_tag, realizations=range(100)):
    result_dir = '../results/results_lognormal{}'.format(cat_tag)
    
    n_converged = 0
    alphas = np.full(np.array(realizations).shape, np.NaN)

    if 'baoiter' not in cf_tag:
        raise ValueError("Need baoiter in cf_tag! Passed cf_tag={cf_tag}")
        return

    i = 0
    for Nr in realizations:

        fn_pattern = f"cf{cf_tag}_converged_*{cat_tag}_rlz{Nr}.npy"
        for cf_fn in glob.glob(f'{result_dir}/{fn_pattern}'):
            #print(cf_fn)
            r_avg, xi, amps, _, extra_dict = np.load(cf_fn, allow_pickle=True)
            #print("C:", amps[4])
            alphas[i] = extra_dict['alpha_result']
            if extra_dict ['alpha_result'] < 0.5 or extra_dict ['alpha_result'] > 1.5:
                print(Nr, extra_dict ['alpha_result'])
            n_converged +=1
            break #should only be 1 match; but probs better way to do this
        i += 1

    print(f"Found {n_converged} converged BAO cfs ({len(realizations)-n_converged} not converged)")    
    print("alpha_mean:", np.nanmean(alphas))
    print("alpha_median:", np.nanmedian(alphas))
    print("alpha_std:", np.nanstd(alphas))
    return alphas


def get_coverged_bao_fn(cat_tag, cf_tag, Nr):
    result_dir = '../results/results_lognormal{}'.format(cat_tag)
    fn_pattern = f"cf{cf_tag}_converged_*{cat_tag}_rlz{Nr}.npy"
    cf_fns = glob.glob(f'{result_dir}/{fn_pattern}')
    if len(cf_fns)==0:
        raise ValueError(f"File matching {fn_pattern} not found!")
    if len(cf_fns)>1:
        raise ValueError(f"Multiple files found matching {fn_pattern}, should only be one!")
    # unpack with: r_avg, xi, amps, proj, extra_dict = np.load(cf_fn, allow_pickle=True)
    return cf_fns[0]


def get_gradient_bao_params(cat_tag, cf_tag, cf_tag_bao=None, Nr=None, rmin=36.0, rmax=200.0, 
                            redshift=0.57, bias=2.0):
    
    assert (cat_tag is not None) and (cf_tag_bao is not None) and (Nr is not None), "Must pass cat_tag, cf_tag, and Nr!"

    bao_fn = get_coverged_bao_fn(cat_tag, cf_tag_bao, Nr)
    _, _, amps_bao, _, _ = np.load(bao_fn, allow_pickle=True) #params: r_avg, xi, amps, proj, extra_dict

    projfn_bao = f"../tables/bases{cat_tag}{cf_tag_bao}_r{rmin}-{rmax}_z{redshift}_bias{bias}_rlz{Nr}.dat"
    bases_bao = np.loadtxt(projfn_bao)

    r = bases_bao[:,0]
    bases_combined = bases_bao[:,1:] @ amps_bao
    
    projfn_grad = f"../tables/bases{cat_tag}{cf_tag}{cf_tag_bao}_rlz{Nr}.dat"
    np.savetxt(projfn_grad, np.array([r, bases_combined]).T)
    nprojbins = 4

    return projfn_grad, nprojbins, projfn_bao, amps_bao


def make_bao_fit(cf_func):
    
    def bao_fit(s, a1, a2, a3, bsq, alpha):
        b1, b2, b3, b4 = bao_bases(s, cf_func, alpha=alpha)
        return a1*b1 + a2*b2 + a3*b3 + bsq*b4
    
    return bao_fit


def bao_bases(s, cf_func, alpha=1):

    b1 = 1.0/s**2
    b2 = 0.1/s
    b3 = 0.001*np.ones(len(s))

    cf = cf_func(s*alpha)
    b4 = cf

    return b1, b2, b3, b4



def bao_fit_standard(r_arr, xi_arr, cosmo_base, redshift=0.0, bias=1.0, realizations=range(100)):
    Plin = cosmology.LinearPower(cosmo_base, redshift, transfer='EisensteinHu')
    CF = cosmology.correlation.CorrelationFunction(Plin)
    def cf_model(s):
        return bias * CF(s)

    alphas = np.zeros(len(realizations))
    for Nr in realizations:
        r_points = r_arr[Nr]
        xi_points = xi_arr[Nr]
        try:
            popt, _ = curve_fit(make_bao_fit(cf_model), r_points, xi_points)
            a1, a2, a3, bsq, alpha = popt
            alphas[Nr] = alpha
        except RuntimeError:
            print(f"Optimal parameters for realization {Nr} not found! continuing")
            alphas[Nr] = np.nan

    return alphas
