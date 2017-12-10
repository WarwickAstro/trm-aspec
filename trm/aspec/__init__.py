#!/usr/bin/env python

"""
package to compute amplitude spectra.

astropy.stats.lombscargle computes various forms of Lomb-Scargle
spectra but not amplitude spectra which I tend to find more informative.
It uses the astropy.stats.lombscargle code to calculate the various sums
in an efficient manner.

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

from astropy.stats.lombscargle.implementations.utils import trig_sum

def amp_spec(t, y, ye, f0, df, Nf, use_fft=True):

    """Fast computation of amplitude spectra adapted from code from
    Vanderplas' fast Lomb-Scargle implementation.

    This implements the Press & Rybicki method [1]_ for fast O[N log(N)]
    Lomb-Scargle trig sums (if use_fft).

    Parameters
    ----------
    t, y, ye : array_like
        times, values, and errors of the data points. These should be
        broadcastable to the same shape.

    f0, df, Nf : (float, float, int)
        parameters describing the (regular) frequency grid,
        f = f0 + df * arange(Nf). The frequencies are measured in
        cycles per unit t.

    use_fft : bool (default=True)
        If True, then use the Press & Rybicki O[NlogN] algorithm to compute
        the trig sums result. Otherwise, use a slower O[N^2] algorithm.

    Returns
    -------
    freq, amp : ndarray
        Arrays of frequency and amplitude associated with each frequency.
        The frequencies have units of cycles per unit time, according on
        the units of "t". The amplitudes have the same units as y.

    Typically "df" the frequency spacing should be a similar order of
    magnitude to 1
    """

    if ye is None:
        ye = 1

    # Validate and setup input data
    t, y, ye = np.broadcast_arrays(t, y, ye)
    if t.ndim != 1:
        raise ValueError("t, y, ye should be one dimensional")

    # Validate and setup frequency grid
    if f0 < 0:
        raise ValueError("Frequencies must be positive")
    if df <= 0:
        raise ValueError("Frequency steps must be positive")
    if Nf <= 0:
        raise ValueError("Number of frequencies must be positive")

    # Normalised weights
    w = ye ** -2.0
    w /= w.sum()

    # Center the data. Even though we're fitting the offset,
    # this step makes the expressions below more succinct
    y -= np.dot(w, y)

    # set up arguments to trig_sum
    kwargs = {'f0' : f0, 'df' : df, 'use_fft' : use_fft, 'N' : Nf}

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

    return (f0+df*np.arange(Nf), amp)

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

    # Define frequencies
    over = 10 # oversampling factor
    f0, df, Nf = 0.075, 1/(t.max()-t.min())/2/over, 1000

    # Compute the power
    fqs, amps = amp_spec(t, y, ye, f0, df, Nf)

    # Plot
    plt.plot(fqs, amps)
    plt.show()
