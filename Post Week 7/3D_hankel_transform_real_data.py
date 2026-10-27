#%%
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as jr
import jax.scipy.special as jss
import healpy as hp
import astropy as ap
from astropy.coordinates import SkyCoord
from dustmaps.edenhofer2023 import Edenhofer2023Query
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.special import j0

NUM_DIMENSIONS = 3

jax.config.update("jax_enable_x64", True)
logvar = jnp.log(1)
logscale = jnp.log(2000)
lognu = jnp.log(1.5)
dx = 0.5
dw = 0.0001
num_omegas = 200
num_rs = 3000
omegas = jnp.arange(num_omegas)*dw
rs = jnp.arange(num_rs)*dx
bessel_xs = jnp.arange(num_omegas*num_rs)*dx*dw
trapezoid_weights = jnp.ones(shape = omegas.shape).at[jnp.array([0, -1])].set(0.5)

def radial_kernel(z, d):
    """g(z) = z^(1-d/2) J_{d/2-1}(z); finite at z = 0."""
    if d == 1: return jnp.sqrt(2/np.pi) * jnp.cos(z)
    if d == 2: return jnp.asarray(j0(np.asarray(z)))
    if d == 3: return jnp.sqrt(2/np.pi) * jnp.sinc(z/np.pi)   # sin(z)/z
    raise ValueError("d must be 1, 2, or 3")

bessel_kernel_values = radial_kernel(bessel_xs, NUM_DIMENSIONS)

def matern_spectral_density(omega, params, d=NUM_DIMENSIONS):
    logvar, logscale, lognu = params
    nu = jnp.exp(lognu)
    log_norm = (logvar
                + d * jnp.log(2 * jnp.sqrt(jnp.pi))
                + jss.gammaln(nu + d / 2) - jss.gammaln(nu)
                + d * (logscale - 0.5 * jnp.log(2 * nu)))
    x = jnp.square(omega) * jnp.exp(2 * logscale - jnp.log(2.0) - lognu)
    return jnp.exp(log_norm - (nu + d / 2) * jnp.log1p(x))


def matern_32(r, variance=jnp.exp(logvar), scale=jnp.exp(logscale), jitter=1e-6):
    result = variance * (1 + jnp.sqrt(3)*r/scale)*jnp.exp(-1*jnp.sqrt(3)*r/scale);
    return jnp.where(r > 0, result, result * (1.0 + jitter))

def matern_covariance_by_hankel_nD(r_index, params, d = NUM_DIMENSIONS):
    bessel_input_indices = r_index*jnp.arange(num_omegas)
    integrands = omegas**(d - 1)*matern_spectral_density(omegas, params)*bessel_kernel_values[bessel_input_indices]*dw
    return jnp.sum(integrands*trapezoid_weights)/(jnp.sqrt(2*jnp.pi)**(d))


fig, axes = plt.subplots(1, 3, figsize = (18, 5))
axes[0].plot(rs, matern_32(rs), color = "blue")
hardcoded_cov_array = jax.vmap(matern_covariance_by_hankel_nD, in_axes=(0, None))(
    jnp.arange(num_rs), (logvar, logscale, lognu))
axes[0].plot(rs, hardcoded_cov_array, color="orange")
axes[1].plot(bessel_xs, bessel_kernel_values)

# %%
