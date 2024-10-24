import numpy as np
import astropy.units as u

class Spectrum(np.ndarray):    
    def __new__(cls, arr, wavelengths,err=None):
        spec = np.asarray(arr).view(cls)
        spec.wavelengths = wavelengths
        spec.wave=wavelengths #alias
        spec.flux=arr
        if err is not None:
            spec.err=err
        return spec
    
    def __array_finalize__(self, spec):
        if spec is None:
            return
        self.wavelengths = getattr(spec, 'wavelengths', None)
    
    def at(self, new_wavelengths):
        new_spec = np.interp(new_wavelengths, self.wavelengths, self)
        return Spectrum(new_spec, new_wavelengths)

def convolve_to_resolution(spec, out_res, in_res=None, verbose=False):
    """
    Convolve the input spectrum to a lower resolution.
    ----------
    Parameters
    ----------
    in_wlen : Wavelength array 
    in_flux : spectrum at high resolution
    in_res : input resolution (high) R~w/dw
    out_res : output resolution (low)
    verbose : if True, print out the sigma of Gaussian filter used
    ----------
    Returns
    ----------
    Convolved spectrum
    """
    from scipy.ndimage import gaussian_filter
    in_wlen = spec.wavelengths
    in_flux = spec
    if isinstance(in_wlen, u.Quantity):
        in_wlen = in_wlen.to(u.nm).value
    if in_res is None:
        in_res = np.mean((in_wlen[:-1]/np.diff(in_wlen)))
    # delta lambda of resolution element is FWHM of the LSF's standard deviation:
    sigma_LSF = np.sqrt(1./out_res**2-1./in_res**2)/(2.*np.sqrt(2.*np.log(2.)))

    spacing = np.mean(2.*np.diff(in_wlen)/ \
      (in_wlen[1:]+in_wlen[:-1]))

    # Calculate the sigma to be used in the gauss filter in pixels
    sigma_LSF_gauss_filter = sigma_LSF/spacing

    result = np.tile(np.nan, in_flux.shape)
    nans = np.isnan(spec)

    result[~nans] = gaussian_filter(in_flux[~nans], \
                               sigma = sigma_LSF_gauss_filter, \
                               mode = 'reflect')
    if verbose:
        print("Guassian filter sigma = {} pix".format(sigma_LSF_gauss_filter))
    return Spectrum(result, spec.wavelengths)
