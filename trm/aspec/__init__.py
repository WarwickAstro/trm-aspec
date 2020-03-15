#!/usr/bin/env python

"""package to compute amplitude spectra.

astropy.stats.lombscargle computes various forms of Lomb-Scargle spectra but
not amplitude spectra which I tend to find more informative. trm.aspec uses
the astropy.stats.lombscargle code to calculate the various trig sums in an
efficient manner but calculates amplitude spectra instead.

There is a single function. Here is example usage::

  from trm.aspec import amp_spec

  # Generate some non-uniformly spaced X values
  t = np.random.uniform(0,1000.,1000)

  # Equivalent y values (single sinusoid, amplitude
  # 0.15, frequency 0.1 cycles/unit time.
  f,a,c,phi,sigma = 0.1, 0.15, 0.1, 2., 0.02
  y = c + a*np.sin(2.*np.pi*f*t+phi)

  # Add a little noise
  y = np.random.normal(y,sigma)
  ye = sigma*np.ones_like(y)

  # Define frequencies
  over = 10 # oversampling factor
  f0, df, Nf = 0.075, 1/(t.max()-t.min())/2/over, 1000

  # Compute the amplitude spectrum
  fqs, amps = amp_spec(t, y, ye, f0, df, Nf)

"""

import numpy as np

from astropy.timeseries.periodograms.lombscargle.implementations.utils import trig_sum
from astropy.timeseries import LombScargle

def amp_spec(t, y, ye, f=None, samples_per_peak=5, nyquist_factor=5, 
             minimum_frequency=None, maximum_frequency=None, use_fft=True):

    """Fast computation of amplitude spectra adapted from code from
    Vanderplas's fast Lomb-Scargle implementation in
    astropy.timeseries.Lombscargle

    This implements the Press & Rybicki method [1]_ for fast
    O[N log(N)] Lomb-Scargle trig sums (if use_fft). Look at
    LombScargle.autofrequency for setting of frequency grid.

    Parameters
    ----------
    t, y, ye : array_like
        times, values, and errors of the data points. These should be
        broadcastable to the same 1D array length.
    f : None or array
        Either an array of frequencies (regularly spaced) or None to generate
        automatically using next parameters
    samples_per_peak : float (optional, default=5)
        The approximate number of desired samples across the typical peak
    nyquist_factor : float (optional, default=5)
        The multiple of the average nyquist frequency used to choose the
        maximum frequency if maximum_frequency is not provided.
    minimum_frequency : float (optional)
        If specified, then use this minimum frequency rather than one
        chosen based on the size of the baseline.
    maximum_frequency : float (optional)
        If specified, then use this maximum frequency rather than one
        chosen based on the average nyquist frequency.
    use_fft : bool (default=True)
        If True, then use the Press & Rybicki O[NlogN] algorithm to compute
        the trig sums result. Otherwise, use a slower O[N^2] algorithm.

    Returns
    -------
    freq, amp : arrays
        Arrays of frequency and amplitude associated with each frequency.
        The frequencies have units of cycles per unit time, according to
        the units of "t". The amplitudes have the same units as y.

    """

    if ye is None:
        ye = 1

    # Validate and setup input data
    t, y, ye = np.broadcast_arrays(t, y, ye)
    if t.ndim != 1:
        raise ValueError("t, y, ye should be one dimensional")

    if f is not None:
        # checks on frequencies
        f0 = f.min()
        Nf = len(f)
        if f0 <= 0:
            raise ValueError("Frequencies must be positive")

        step = (f.max()-f.min())/(Nf-1)
        if step <= 0:
            raise ValueError("Frequency steps must be positive")

        df = f[1:]-f[:-1]
        if (df != step).any():
            raise ValueError("Frequency grid must be regular")
    else:
        # compute frequency grid
        ls = LombScargle(t,y,ye)
        f = ls.autofrequency(
            samples_per_peak, nyquist_factor,
            minimum_frequency, maximum_frequency
        )
        f0 = f.min()
        step = (f.max()-f.min())/(len(f)-1)
        Nf = len(f)

    # Normalised weights
    w = ye ** -2.0
    w /= w.sum()

    # Center the data. Even though we're fitting the offset,
    # this step makes the expressions below more succinct
    y -= np.dot(w, y)

    # set up arguments to trig_sum
    kwargs = {'f0' : f0, 'df' : step, 'use_fft' : use_fft, 'N' : Nf}

    # ----------------------------------------------------------------------
    # 1. compute functions of the time-shift tau at each frequency
    Sh, Ch = trig_sum(t, w * y, **kwargs)
    S2, C2 = trig_sum(t, w, freq_factor=2, **kwargs)

    S, C = trig_sum(t, w, **kwargs)
    tan_2omega_tau = (S2 - 2 * S * C) / (C2 - (C * C - S * S))

    # This is what we're computing below; the straightforward way is slower
    # and less stable, so we use trig identities instead
    #
    # omega_tau = 0.5 * np.arctan(tan_2omega_tau)
    # S2w, C2w = np.sin(2 * omega_tau), np.cos(2 * omega_tau)
    # Sw, Cw = np.sin(omega_tau), np.cos(omega_tau)

    S2w = tan_2omega_tau / np.sqrt(1 + tan_2omega_tau * tan_2omega_tau)
    C2w = 1 / np.sqrt(1 + tan_2omega_tau * tan_2omega_tau)
    Cw = np.sqrt(0.5) * np.sqrt(1 + C2w)
    Sw = np.sqrt(0.5) * np.sign(S2w) * np.sqrt(1 - C2w)

    # ----------------------------------------------------------------------
    # 2. Compute the periodogram, following Zechmeister & Kurster
    #    and using tricks from Press & Rybicki.
    YY = np.dot(w, y ** 2)
    YC = Ch * Cw + Sh * Sw
    YS = Sh * Cw - Ch * Sw
    CC = 0.5 * (1 + C2 * C2w + S2 * S2w)
    SS = 0.5 * (1 - C2 * C2w - S2 * S2w)

    CC -= (C * Cw + S * Sw) ** 2
    SS -= (S * Cw - C * Sw) ** 2

    amp = np.sqrt( (YC/CC)**2 + (YS/SS)**2 )

    return (f, amp)

if __name__ == '__main__':

    # Tests the amplitude spectrum code
    import matplotlib.pyplot as plt

    # Generate some non-uniformly spaced X values
    t = np.random.uniform(0,1000.,1000)
    t.sort()

    # Equivalent y values
    f,a,c,phi,sigma = 0.1, 0.15, 0.1, 2., 0.02
    y = c + a*np.sin(2.*np.pi*f*t+phi)

    # Add a little noise
    y = np.random.normal(y,sigma)
    ye = sigma*np.ones_like(y)

    # Compute the power
    fqs, amps = amp_spec(t, y, ye)

    # Plot
    plt.plot(fqs, amps)
    plt.show()
