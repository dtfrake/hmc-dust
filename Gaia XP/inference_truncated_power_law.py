#%%
import blackjax
import sys, os
sys.path.insert(0, os.path.abspath(".."))   # notebook is in Toy Data/; Better_HMC.py is one level up
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import jax
import jax.random as jr
import arviz as az
import jax.scipy.special as jss
from functools import partial
# Import the custom package you generated
from hmc_dust.better_hmc import HMCSampler 
import matplotlib.pyplot as plt
jax.config.update("jax_enable_x64", True)
#

num_dimensions = 250
num_data = 100
noise_magnitude_on_each_point = 0.03
num_fourier_entries = 350
total_length = 0.5
dx = total_length/(num_dimensions - 1)
fixed_points_linspace = jnp.arange(num_dimensions)*dx 
full_fourier_period_linspace = jnp.arange(num_fourier_entries)*dx 
boundary_padding = total_length*((num_fourier_entries - num_dimensions)/(num_dimensions - 1))
omegas = 2 * jnp.pi * jnp.fft.fftfreq(num_fourier_entries, d = dx)
num_fitting_params = 2
fourier_length = total_length + boundary_padding
normalization_factor = num_fourier_entries/jnp.sqrt(fourier_length)

num_fitting_params = 4

# change this to overall variance
# try integrating numerically 
hardcoded_log_overall_variance = 0
hardcoded_log_exponent = 1
hardcoded_log_zero_mean = -6
hardcoded_log_zero_logvar = -2


hardcoded_params = (hardcoded_log_overall_variance, hardcoded_log_exponent, hardcoded_log_zero_mean, hardcoded_log_zero_logvar)
num_overall_steps = 6000
burn_in = 1000

num_integration_steps = 600
step_size = 0.002

use_NUTS = False
use_blackjax_hmc = False
window_adapt_square_mass_matrix = False
#%%
def hartley(x):
    X = jnp.fft.fft(x)
    return jnp.real(X) - jnp.imag(X)

def power_law_field(xi, inference_params):
    log_overall_variance, log_exponent, log_zero_mean, log_zero_logvar = inference_params
    exponent = -1*jnp.exp(log_exponent)
    log_zero_std = jnp.exp(0.5*log_zero_logvar)        # not sqrt(exp(...))

    log_first_power = log_zero_mean + xi[0]*log_zero_std

    omegas = 2*jnp.pi*jnp.fft.fftfreq(num_fourier_entries, d=dx)[1:]
    log_powers = exponent * jnp.log(jnp.abs(omegas))
    log_sum_powers = jax.scipy.special.logsumexp(log_powers)

    # log(overall_variance * fourier_length / (first_power + sum(powers)))
    log_scale = (log_overall_variance + jnp.log(fourier_length)
                 - jnp.logaddexp(log_first_power, log_sum_powers))

    first_amp = jnp.exp(0.5*(log_first_power + log_scale))
    amps      = jnp.exp(0.5*(log_powers      + log_scale))

    scaled_noise = jnp.concatenate([xi[1:2]*first_amp, xi[2:]*amps])
    field = hartley(scaled_noise)/jnp.sqrt(fourier_length)
    return field[:num_dimensions]
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
xi = jr.normal(k3, shape = num_fourier_entries + 1)
true_field = power_law_field(xi, hardcoded_params)
y_obs_true = jnp.interp(x_obs, fixed_points_linspace, true_field)
#y_obs_true = jnp.sin(30*x_obs/10) * jnp.exp(-5*x_obs/10)
y_obs_err = noise_magnitude_on_each_point * jnp.ones_like(y_obs_true)
noise = y_obs_err * jr.normal(k2, shape=y_obs_true.shape)
y_obs = y_obs_true + noise


plt.errorbar(x_obs, y_obs, yerr = noise_magnitude_on_each_point, fmt='o', label='Observations')
plt.plot(fixed_points_linspace, true_field)
plt.legend()
plt.savefig('combined.png', dpi=300, bbox_inches='tight')
plt.close()
def response_function(xi, inference_params):
    field_points = power_law_field(xi, inference_params)
    return jnp.interp(x_obs, fixed_points_linspace, field_points)
#%%
def log_params_prior(inference_params):
    log_overall_variance, log_exponent, log_zero_mean, log_zero_var = inference_params 
    return -0.5*log_overall_variance**2 - 0.5*log_exponent**2 - 0.5*log_zero_mean**2 - 0.5*log_zero_var**2

def negative_logdensity(x):
    inference_params, xi = tuple(x[:num_fitting_params]), x[num_fitting_params:]
    negative_log_p_s = -log_params_prior(inference_params) + 0.5*xi.T @ xi
    res = y_obs - response_function(xi, inference_params)
    negative_log_p_d_given_s = 0.5*res.T @ inv_cov_data_matrix @ res
    return negative_log_p_s + negative_log_p_d_given_s

#%%
num_good_samples = num_overall_steps - burn_in
time_arr = jnp.arange(num_good_samples)


# NUTS samples a POSITIVE log-density, so negate your negative one.
logdensity_fn = lambda x: -negative_logdensity(x)

rng = jr.PRNGKey(5)
rng, k1, k2 = jr.split(rng, 3)
initial_coordinates = jnp.zeros(num_fourier_entries + 1 + num_fitting_params)

initial_coordinates = initial_coordinates.at[0].set(hardcoded_log_overall_variance)
initial_coordinates = initial_coordinates.at[1].set(hardcoded_log_exponent)
initial_coordinates = initial_coordinates.at[2].set(hardcoded_log_zero_mean)
initial_coordinates = initial_coordinates.at[3].set(hardcoded_log_zero_logvar)

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
        inv_mass_matrix = jnp.eye(num_fourier_entries + 1 + num_fitting_params),
        alpha=1.0
    )

    overall_data_arr, overall_momentum_arr, accept_prob_arr = sampler.sample(
        start_position=initial_coordinates, 
        num_samples=num_overall_steps,
        burn_in = burn_in,
        rng_key=rng
    )

cov= jnp.cov(overall_data_arr, rowvar = False)
np.save("truncated_power_law_overall_data_arr_output.npy", overall_data_arr)
np.save("truncated_power_law_accept_prob_arr_output.npy", accept_prob_arr)
#%%
