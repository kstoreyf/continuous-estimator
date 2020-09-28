import numpy as np
import glob


def get_label(pt):
    label_dict = {'generalr': 'cosmo deriv', 'tophat': 'tophat', 'piecewise':'triangle', 'linear_spline':'linear spline', 'quadratic_spline': 'quadratic spline', 'quadratic_spline_nbins8':'quadratic spline (8 bins)', 'gaussian_kernel':'gaussian kernel'}
    for k in label_dict.keys():
        if pt==k:
            return label_dict[k]
        pt0 = pt.split('_')[0]
        if '_n' in pt:
            nbins = pt.split('_')[1][1:]
            return '{}, {} bins'.format(pt0, nbins)
        if pt0 in k:
            return label_dict[k]
    return pt

orange='#FFA317'
green='#771298'
red='#D41159'
def get_color(pt):
    color_dict = {'tophat':'#1A85FF', 'standard': 'orange', 'piecewise':'crimson', 'linear_spline':'red', 'cosmo deriv':'purple', 'triangle':'crimson', 'quadratic_spline':'limegreen', 'quadratic_spline_nbins8':'limegreen', 'gaussian_kernel':'orangered', 'baoiter':green, 'cubic_spline':red}
    for k in color_dict.keys():
        pt0 = pt.split('_')[0]
        if pt0 in k:
            return color_dict[k]
    else:
        return 'blue'


def partial_derivative(f1, f2, dv):
    df = f2-f1
    deriv = df/dv
    return deriv


def covariance(arrs, zeromean=False):
    arrs = np.array(arrs)
    N = arrs.shape[0]

    if zeromean:
        w = arrs
    else:
        w = arrs - arrs.mean(0)

    outers = np.array([np.outer(w[n], w[n]) for n in range(N)])
    covsum = np.sum(outers, axis=0)
    cov = 1.0/float(N-1.0) * covsum
    return cov


# aka Correlation Matrix
def reduced_covariance(cov):
    cov = np.array(cov)
    Nb = cov.shape[0]
    reduced = np.zeros_like(cov)
    for i in range(Nb):
        ci = cov[i][i]
        for j in range(Nb):
            cj = cov[j][j]
            reduced[i][j] = cov[i][j]/np.sqrt(ci*cj)
    return reduced


# The prefactor unbiases the inverse; see e.g. Pearson 2016
def inverse_covariance(cov, N):
    inv = np.linalg.inv(cov)
    Nb = cov.shape[0]
    prefac = float(N - Nb - 2)/float(N - 1)
    return prefac * inv


def load_data(cat_tag, cf_tag, Nrealizations=100, return_amps=False):
    
    cat_dir = '../catalogs'
    result_dir = '../results/results_lognormal{}'.format(cat_tag)

    rs = []
    xis = []
    amps = []
    n_converged = 0

    for Nr in range(Nrealizations):

        if 'baoiter' in cf_tag:
            fn_pattern = f"cf{cf_tag}_converged_*{cat_tag}_rlz{Nr}.npy"
            matches = glob.glob(f'{result_dir}/{fn_pattern}')
            for cf_fn in matches:
                r_avg, xi, amp, _, _ = np.load(cf_fn, allow_pickle=True)
                n_converged +=1
                break #should only be 1 match; but probs better way to do this
            if len(matches)==0:
                continue
        else:
            cf_fn = '{}/cf{}{}_rlz{}.npy'.format(result_dir, cf_tag, cat_tag, Nr)
            r_avg, xi, amp, proj, extra = np.load(cf_fn, allow_pickle=True)
        rs.append(r_avg)
        xis.append(xi)
        amps.append(amp)
        
    if 'baoiter' in cf_tag:
        print(f'Number converged: {n_converged}/{Nrealizations}')
        
    if return_amps:
        return rs, xis, amps
    else:
        return rs, xis


def load_true(cat_tag, bias=1.0):
    cat_dir = '../catalogs'
    true_fn = '{}/inputs/cat{}_Rh_xi.txt'.format(cat_dir, cat_tag)
    r_true, xi_true = np.loadtxt(true_fn, unpack=True)
    xi_true *= bias**2
    return r_true, xi_true


def load_bao(cat_tag, cf_tag, Nr):
    
    cat_dir = '../catalogs'
    result_dir = '../results/results_lognormal{}'.format(cat_tag)

    rs = []
    xis = []
    amps = []
    extras = []
    niters = []
    assert 'baoiter' in cf_tag, "baoiter not in cf_tag!"
    fn_pattern = f"cf{cf_tag}_*{cat_tag}_rlz{Nr}.npy"
    matches = glob.glob(f'{result_dir}/{fn_pattern}')
    
    for cf_fn in matches:
        r_avg, xi, amp, proj, extra_dict = np.load(cf_fn, allow_pickle=True)
    
        rs.append(r_avg)
        xis.append(xi)
        amps.append(amp)
        extras.append(extra_dict)

        fn_split = cf_fn.split('_')
        for nn in fn_split:
            if nn.startswith('niter'):
                niter = int(nn[len('niter'):])
                niters.append(niter)
    return rs, xis, amps, extras, niters


# deriv: 2ax + b = 0
# x = -b/(2a)
def quadratic(x, a, b, c):
    return a*x**2 + b*x + c


def find_peaks_center(r_arr, xi_arr, rpeak_guess, bws, r_widths=[11.0], region=(85,115), show_bad=False):
    xi_arr = np.array(xi_arr)
    if len(xi_arr.shape)<3:
        r_arr = [r_arr]
        xi_arr = [xi_arr]
        
    ntags = np.array(xi_arr).shape[0]
    N = np.array(xi_arr).shape[1]
    n_nans_tot = 0
    
    if show_bad:
        plt.figure()
        #plt.ylim(-0.01, 0.05)
        plt.xlim(36, 156)
    
    r_peak_arr = np.zeros((ntags, N))
    for i in range(ntags):
        bw = bws[i]
        r_width = r_widths[i]

        rs = r_arr[i]
        xis = xi_arr[i]
        
        n_nans = 0
        n_botedge = 0
        n_topedge = 0
        #r_maxes = []
        #r_peaks = []
        for j in range(N):
            r = rs[j]
            xi = xis[j]

            xi_func = interp1d(r, xi, kind='cubic')
            
            r_edges = np.arange(min(r), max(r)+bw, bw)
            r_avg = 0.5*(r_edges[:-1] + r_edges[1:])
            
            r_points = r_avg[np.where(np.abs(r_avg-rpeak_guess)<r_width)]
            assert len(r_points)>2, "Bad points! <3"
            
            xi_points = [xi_func(rp) for rp in r_points]
            popt, _ = curve_fit(quadratic, r_points, xi_points)
            a, b, c = popt

            if a>0:
                r_peak_arr[i][j] = np.random.choice(region)
                if show_bad:
                    plt.plot(r, xi+n_nans_tot*0.005)
                n_nans += 1
                n_nans_tot += 1
                continue

            r_peak = -b/(2*a)
            if r_peak<region[0]:
                r_peak=region[0]
                n_botedge +=1
            elif r_peak>region[1]:
                r_peak=region[1]
                n_topedge += 1
            r_peak_arr[i][j] = r_peak
#             if r_peak<region[0] or r_peak>region[1]:
#                 r_peak = np.NaN
#                 n_nans += 1
            r_peak_arr[i][j] = r_peak 
        #r_peak_arr.append(np.array(r_peaks))
        
        print('Number of NaNs:', n_nans, ', Bottom edges:', n_botedge, ', Top edges:', n_topedge)
     
    if show_bad:
        plt.ylim(-0.01, n_nans/45.)
    return r_peak_arr, r_points, xi_points, popt


