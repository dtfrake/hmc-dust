import numpy as np
import jax.numpy as jnp
from scipy.special import gamma, jv
 
 
def compute_hankel_matrix(k, r, *, d):
    """
    Compute matrix to convert P(k) to C(r) via Hankel transform in d dimensions.
    Assumes P(k) is piecewise constant between k bins, output will have shape (len(r), len(k)-1).
    Requires scipy for Bessel functions, so this cannot be differentiated through.
    """
    k = np.asarray(k, dtype=float)
    r = np.asarray(r, dtype=float)[:, None]
 
    # Both branches of np.where get evaluated, so the r = 0 rows produce
    # harmless inf/nan values in `limits` that are discarded. Silence the warnings.
    with np.errstate(divide="ignore", invalid="ignore"):
        limits = (k / (2 * np.pi * r)) ** (d / 2) * jv(d / 2, k * r)
 
    zero = (k / 2) ** d / (np.pi ** (d / 2) * gamma(d / 2 + 1))
    weights = np.where(
        r > 0,
        limits[:, 1:] - limits[:, :-1],
        zero[None, 1:] - zero[None, :-1],
    )
    return weights


 
def covariance_from_spectrum(cov_bins, logk, *, d):
    """
    Create a function which converts a discretized power spectrum to a discretized covariance.
    Certain values must be precomputed outside of JAX, hence the factory function.
    Specifically, this computes the integrals analytically using Bessel functions assuming piecewise constant P(k).
 
    Args:
        cov_bins: Radial bins at which to evaluate the covariance. First entry should probably be zero.
        logk: Logarithmic k values defining the upper edges of the power spectrum bins. The first bin is assumed to start at k=0.
        d: Dimensionality of the space.
 
    Returns:
        cov_func: Callable taking logarithmic power spectrum in the defined k bins, plus overall variance kwarg, returning covariance values at cov_bins.
    """
    # Precompute Hankel matrix with NumPy/SciPy, then hand it to JAX
    k = np.concatenate([[0.0], np.exp(np.asarray(logk, dtype=float))])
    hankel_matrix = jnp.asarray(compute_hankel_matrix(k, cov_bins, d=d))
 
    def cov_func(logp, *, variance):
        cov_vals = hankel_matrix @ jnp.exp(logp)
        cov_vals = variance * (cov_vals / cov_vals[0])
        return cov_vals
 
    return cov_func