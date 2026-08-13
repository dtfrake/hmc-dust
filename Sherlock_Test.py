import blackjax
import sys, os
sys.path.insert(0, os.path.abspath(".."))   # notebook is in Toy Data/; Better_HMC.py is one level up
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import jax
import jax.random as jr
import jax.scipy.special as jss
from functools import partial
# Import the custom package you generated
from Better_HMC import HMCSampler 
import matplotlib.pyplot as plt
jax.config.update("jax_enable_x64", True)


fixed_logscale = 0
num_dimensions = 250
num_data = 100
noise_magnitude_on_each_point = 0.1
total_length = 10
## Very important: Dividing by num_dims - 1 is inconsistent with the other notebooks that fit two params. If I divide by num_dims instead, then I am essentially
## measuring the field at bin centers instead of bin edges which is more consistent with the 2D case, but then some field on each edge is not measured
dx = total_length/(num_dimensions - 1)
fixed_points_linspace = jnp.arange(num_dimensions)*dx 

num_fitting_params = 2

fixed_logscale = jnp.log(20)
hardcoded_logvar = 0
hardcoded_lognu = jnp.log(1.5)

hardcoded_params = (hardcoded_logvar, hardcoded_lognu)
num_overall_steps = 1000
burn_in = 0

num_integration_steps = 500
step_size = 0.002

use_NUTS = False
use_blackjax_hmc = False
window_adapt_square_mass_matrix = False

def matern_32(r, variance=jnp.exp(hardcoded_logvar), scale=jnp.exp(fixed_logscale), jitter=1e-6):
    result = variance * (1 + jnp.sqrt(3)*r/scale)*jnp.exp(-1*jnp.sqrt(3)*r/scale)
    return jnp.where(r > 0, result, result * (1.0 + jitter))

def matern_12(r, variance=jnp.exp(hardcoded_logvar), scale=jnp.exp(fixed_logscale), jitter=1e-6):
    result = variance * jnp.exp(-1*r/scale)
    return jnp.where(r > 0, result, result * (1.0 + jitter));

# 8/6 I removed the jitter
def general_matern_spectral_density(omega, inference_params):
        logvar, lognu = inference_params
        # Scale is fixed
        logscale = fixed_logscale
        nu = jnp.exp(lognu)
        log_ratio = jss.gammaln(nu + 0.5) - jss.gammaln(nu)
        r = jnp.square(omega) * jnp.exp(2 * logscale - jnp.log(2.0) - lognu)
        logdensity = logvar + log_ratio - 0.5*jnp.log(2*nu) + logscale - (nu + 0.5)*jnp.log1p(r)
        result = 2*jnp.sqrt(jnp.pi)*jnp.exp(logdensity)
        return result


def get_matern_covariance(inference_params, dx_internal, dx_external, N_internal, N_external):
    omegas = 2 * jnp.pi * jnp.fft.fftfreq(2*N_internal, d=dx_internal)    
    densities = general_matern_spectral_density(omegas, inference_params)
    full_period_complex_covariance = jnp.fft.ifft(densities)/dx_internal
    #assert jnp.abs(full_period_complex_covariance.imag).max() < 0.0001
    full_period_covariance = jnp.real(full_period_complex_covariance)
    positive_x_linspace = jnp.arange(N_internal + 1)*dx_internal
    dx_return_linspace = jnp.arange(N_external)*dx_external
    assert dx_external*(N_external - 1) <= dx_internal*N_internal
    return jnp.interp(dx_return_linspace, positive_x_linspace, full_period_covariance[:N_internal + 1])

def dist_function(i, j):
    return jnp.abs(i - j)*dx
shape = (num_dimensions, num_dimensions)
dist_matrix = jnp.fromfunction(dist_function, shape)
def matrix_field(xi, inference_params):
    # This is equivalent to 10x padding
    matern_covariance = get_matern_covariance(inference_params, dx, dx, 10*num_dimensions, num_dimensions)
    idx = jnp.arange(matern_covariance.shape[0])
    K = matern_covariance[jnp.abs(idx[:, None] - idx[None, :])]
    K = K + 1e-6 * matern_covariance[0] * jnp.eye(K.shape[0])
    L = jnp.linalg.cholesky(K)
    return L @ xi



def cdf(x):
    return x + jnp.sin(4 * jnp.pi * x) / (4 * jnp.pi)

def pdf(x):
    return 1 + jnp.cos(4 * jnp.pi * x)

@partial(jax.jit, static_argnames=("n", "iters"))
def sample(key, n, iters=8):
    u = jr.uniform(key, (n,))
    x = u  # good initial guess since CDF(x) ≈ x on average
    for _ in range(iters):
        x = x - (cdf(x) - u) / pdf(x)   # Newton's method
        x = jnp.clip(x, 0.0, 1.0)
    return x

diagonal_values = jnp.ones(num_data)*noise_magnitude_on_each_point**2

cov_data_matrix = jnp.diag(diagonal_values)
inv_cov_data_matrix = jnp.linalg.inv(cov_data_matrix)

k1, k2, k3, k4 = jr.split(jr.key(25), 4)



x_obs = sample(k1, num_data)*total_length
xi = jr.normal(k3, shape = num_dimensions)
true_field = matrix_field(xi, (hardcoded_logvar, hardcoded_lognu))
y_obs_true = jnp.interp(x_obs, fixed_points_linspace, true_field)
#y_obs_true = jnp.sin(30*x_obs/10) * jnp.exp(-5*x_obs/10)
y_obs_err = noise_magnitude_on_each_point * jnp.ones_like(y_obs_true)
noise = y_obs_err * jr.normal(k2, shape=y_obs_true.shape)
y_obs = y_obs_true + noise

plt.errorbar(x_obs, y_obs, yerr = noise_magnitude_on_each_point, fmt='o', label='Observations')

plt.plot(fixed_points_linspace, true_field)
def response_function(xi, inference_params):
    field_points = matrix_field(xi, inference_params)
    return jnp.interp(x_obs, fixed_points_linspace, field_points)


def logvarprior(logvar):
    return -0.5*logvar*logvar
def logscaleprior(logscale):
    return -0.5*logscale*logscale
def lognuprior(lognu):
    return -0.5*lognu*lognu

def negative_logdensity(x):
    logvar = x[0]
    lognu = x[1]
    xi = x[2:]
    inference_params = (logvar, lognu)
    negative_log_p_s = -logvarprior(logvar) - lognuprior(lognu) + 0.5*xi.T @ xi
    res = y_obs - response_function(xi, inference_params)
    negative_log_p_d_given_s = 0.5*res.T @ inv_cov_data_matrix @ res
    return negative_log_p_s + negative_log_p_d_given_s


num_good_samples = num_overall_steps - burn_in
time_arr = jnp.arange(num_good_samples)


# NUTS samples a POSITIVE log-density, so negate your negative one.
logdensity_fn = lambda x: -negative_logdensity(x)

rng = jr.PRNGKey(5)
rng, k1, k2 = jr.split(rng, 3)
initial_coordinates = jnp.zeros(num_dimensions + num_fitting_params)
initial_coordinates = initial_coordinates.at[0].set(hardcoded_logvar)
initial_coordinates = initial_coordinates.at[1].set(hardcoded_lognu)

if(use_NUTS):
        def inference_loop(rng_key, kernel, initial_state, num_samples):
            @jax.jit
            def one_step(state, key):
                state, info = kernel(key, state)
                return state, (state, info)
            keys = jr.split(rng_key, num_samples)
            _, (states, infos) = jax.lax.scan(one_step, initial_state, keys)
            return states, infos


        warmup = blackjax.window_adaptation(blackjax.nuts, logdensity_fn, is_mass_matrix_diagonal= not window_adapt_square_mass_matrix)
        (state, parameters), _ = warmup.run(k1, initial_coordinates, num_steps=burn_in)
        nuts = blackjax.nuts(logdensity_fn, **parameters)
        initial_state = state

        states, infos = inference_loop(k2, nuts.step, initial_state, num_good_samples)

        overall_data_arr = states.position
        accept_prob_arr  = infos.acceptance_rate
elif use_blackjax_hmc:
    def inference_loop(rng_key, kernel, initial_state, num_samples):
        @jax.jit
        def one_step(state, key):
            state, info = kernel(key, state)
            return state, (state, info)
        keys = jr.split(rng_key, num_samples)
        _, (states, infos) = jax.lax.scan(one_step, initial_state, keys)
        return states, infos

    warmup = blackjax.window_adaptation(
        blackjax.hmc, logdensity_fn,
        num_integration_steps=num_integration_steps,
        is_mass_matrix_diagonal=not window_adapt_square_mass_matrix,
    )
    (state, parameters), _ = warmup.run(k1, initial_coordinates, num_steps=burn_in)

    hmc = blackjax.hmc(logdensity_fn, **parameters)
    initial_state = state

    states, infos = inference_loop(k2, hmc.step, initial_state, num_good_samples)

    overall_data_arr = states.position
    accept_prob_arr  = infos.acceptance_rate
    np.save('blackjax_cov_matrix.npy', np.asarray(parameters["inverse_mass_matrix"]))
else :
    sampler = HMCSampler(
        negative_logdensity= negative_logdensity,
        num_integration_steps =  num_integration_steps,
        step_size= step_size,
        inv_mass_matrix = jnp.eye(num_dimensions + num_fitting_params),
        alpha=1.0
    )

    overall_data_arr, overall_momentum_arr, accept_prob_arr = sampler.sample(
        start_position=initial_coordinates, 
        num_samples=num_overall_steps,
        burn_in = burn_in,
        rng_key=rng
    )



import os
OUTDIR = os.environ.get("OUTDIR", ".")
os.makedirs(OUTDIR, exist_ok=True)
def out(name):
    return os.path.join(OUTDIR, name)

np.savez_compressed(
    out("samples.npz"),
    overall_data_arr=np.asarray(overall_data_arr),
    accept_prob_arr=np.asarray(accept_prob_arr),
    x_obs=np.asarray(x_obs),
    y_obs=np.asarray(y_obs),
    true_field=np.asarray(true_field),
    num_dimensions=num_dimensions,
    num_data=num_data,
    num_overall_steps=num_overall_steps,
    burn_in=burn_in,
    step_size=step_size,
    num_integration_steps=num_integration_steps,
    hardcoded_logvar=hardcoded_logvar,
    hardcoded_lognu=hardcoded_lognu,
    fixed_logscale=fixed_logscale,
)
print(f"saved raw samples to {out('samples.npz')}", flush=True)


logvars = overall_data_arr[:, 0]
lognus = overall_data_arr[:, 1]
overall_position_arr = overall_data_arr[:, 2:]  
logscales = jnp.zeros_like(logvars)
inference_params = (logvars, lognus)

def field_from_sample(sample):
    logvar, lognu, xi = sample[0], sample[1], sample[2:]
    return matrix_field(xi, (logvar, lognu))

overall_position_arr = jax.lax.map(jax.jit(field_from_sample), overall_data_arr)

mean_position = overall_position_arr.sum(axis=0)/num_good_samples

vars = jnp.exp(logvars)
scales = jnp.exp(logscales)
nus = jnp.exp(lognus)
cov= jnp.cov(overall_data_arr, rowvar = False)

np.savez_compressed(
    out("fields.npz"),
    overall_position_arr=np.asarray(overall_position_arr),
    mean_position=np.asarray(mean_position),
    cov=np.asarray(cov),
)
print(f"saved derived fields to {out('fields.npz')}", flush=True)