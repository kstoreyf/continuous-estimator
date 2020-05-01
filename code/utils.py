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
