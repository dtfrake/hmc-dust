#%%
import os
import jax
from astropy.table import Table
import numpy as np
import jax.numpy as jnp
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
from dustmaps.config import config
import healpy as hp
import astropy as ap
from astropy.coordinates import SkyCoord
import jax.random as jr
from astropy.io import fits
from hmc_dust.better_hmc import HMCSampler 
import jax.scipy.special as jss
import arviz as az
import blackjax
from jax.scipy.stats import t
from jax.scipy.special import logsumexp
from jax.scipy.special import gammaln
from jax.scipy.stats import norm


from hmc_dust import DATA, GAIA

RUN = os.environ.get("SLURM_JOB_ID", "local")

def out(name):
    root, ext = os.path.splitext(name)
    return f"{root}_{RUN}{ext}"


#def out(name):
    #root, ext = os.path.splitext(name)
    #return f"{root}_mean{INV_GAMMA_MEAN:g}_var{INV_GAMMA_VAR:g}_{RUN}{ext}"

fits_table = Table.read(GAIA / "gc_sightline.fits")

gc_builtin = SkyCoord(0*u.deg, 0*u.deg, frame='galactic').icrs

star_coords = SkyCoord(np.asarray(fits_table["ra"])*u.deg, np.asarray(fits_table["dec"])*u.deg)
fits_table["sep_gc_deg"] = star_coords.separation(gc_builtin).deg


plx_np = fits_table["parallax"]              # parallax measurements (mas)
plx = jnp.asarray(np.asarray(plx_np).astype(np.float32))
plx_err_np = fits_table["parallax_err"]         # their uncertainties (mas)
plx_err = jnp.asarray(np.asarray(plx_err_np).astype(np.float32))
ext_np  = fits_table["ext"]                  # extinction
ext = jnp.asarray(np.asarray(ext_np).astype(np.float32))
ext_err_np = fits_table["ext_err"]
ext_err = jnp.asarray(np.asarray(ext_err_np).astype(np.float32))
sep_np = fits_table["sep_gc_deg"]
separation = jnp.asarray(np.asarray(sep_np).astype(np.float32))
print(fits_table.info())

with fits.open(DATA / "mean_and_std_healpix.fits") as hdul:
    hdul.info()
    print(hdul['MEAN'].header['ORDERING'])
    print(hdul['STD.'].header['ORDERING'])
    galactic_center_healpix = hp.pixelfunc.ang2pix(256, theta = 0.0, phi = 0.0, nest = True, lonlat = True)
    mean_along_center = jnp.asarray(hdul["MEAN"].data[:, galactic_center_healpix].astype(np.float32))
    std_along_center  = jnp.asarray(hdul["STD."].data[:, galactic_center_healpix].astype(np.float32))
    radial_centers    = jnp.asarray(hdul["RADIAL PIXEL CENTERS"].data.field(0),    dtype=jnp.float32)
    radial_boundaries = jnp.asarray(hdul["RADIAL PIXEL BOUNDARIES"].data.field(0), dtype=jnp.float32)
    distance_gaps = np.diff(radial_boundaries)

# 1000/x is its own inverse so this fcn is also plx_to_dist
def dist_to_plx(dist):
    return 1000/dist

def get_x_err_basic(plx_err, plx):
    return 1000*(plx**(-2))*plx_err

def logposterior_on_r(r, w, w_err):
    p_w_given_r = -0.5*(w - 1000/r)**2/(w_err**2) 
    p_r = distance_logprior(r)
    return p_r + p_w_given_r

def get_x_err_advanced(plx_err, plx):
    x = jnp.linspace(0.1, 100000, 20000)
    log_posterior = logposterior_on_r(x, plx, plx_err)
    y = jnp.exp(log_posterior - jnp.max(log_posterior))
    Z    = jnp.trapezoid(y, x)                    
    mean = jnp.trapezoid(x * y, x) / Z
    var  = jnp.trapezoid((x - mean) ** 2 * y, x) / Z
    return jnp.sqrt(var)

def distance_logprior(r):
    return jnp.where(r > 0, -r/10000 + 2*jnp.log(r), -jnp.inf)




assert mean_along_center.shape == distance_gaps.shape
num_stars = plx.size
distances_psc = dist_to_plx(plx)

linspace = jnp.linspace(0.1, 20000, 500)
x_err_basic = jax.vmap(get_x_err_basic, in_axes=(None, 0))(0.1, 1000/linspace)
x_err_advanced = jax.vmap(get_x_err_advanced, in_axes=(None, 0))(0.1, 1000/linspace)
plt.plot(linspace, x_err_basic, label = "Propogation of Uncertainty Error")
plt.plot(linspace, x_err_advanced, label = "Square Root of integrated posterior variance (x^2 included in exponential prior)")
plt.xscale("log")
plt.yscale("log")
plt.xlim(100, 100000)
plt.xlabel("Distance")
plt.ylabel("Distance error (given plx_err = 0.1)")
plt.legend()
#%%
num_dimensions = 600
num_fourier_dimensions = 1200
left_distance_padding = 40
right_distance_padding = 600


min_distance = radial_boundaries[0] - left_distance_padding
max_distance = radial_boundaries[radial_boundaries.size - 1] + right_distance_padding
dx = (max_distance - min_distance)/(num_dimensions - 1)
# This is the analouge of fixed_points_linspace
integrating_dims_padding_left = 20
integrating_dims_padding_right = 300
total_integrating_dims = integrating_dims_padding_left + integrating_dims_padding_right + num_dimensions
integrating_dust_x_linspace = jnp.arange(num_dimensions + integrating_dims_padding_left + integrating_dims_padding_right)*dx + min_distance - integrating_dims_padding_left*dx
dust_x_linspace = jnp.arange(num_dimensions)*dx + min_distance
fourier_length = dx*(num_fourier_dimensions - 1)
num_fitting_params = 5
omegas = 2*jnp.pi*jnp.fft.fftfreq(num_fourier_dimensions, d = dx)
error_bars_on_data = 0.025

cumulative_integrated_dust = jnp.concatenate(
    [jnp.array([0.0]), jnp.cumsum(mean_along_center * distance_gaps)])
#%%
x_obs = dist_to_plx(plx)
y_obs = ext


mask = (separation < 0.3) & (plx - plx_err > 1000/1800) & (plx + plx_err < 1000/40)
x_obs = x_obs[mask]
y_obs = y_obs[mask]
plx_obs = plx[mask]
data_ext_err = ext_err[mask]
data_plx_err = plx_err[mask]
x_error = jax.vmap(get_x_err_basic)(plx_err[mask], plx_obs)
num_data = x_obs.size
inv_cov_data_matrix = jnp.linalg.inv(jnp.diag(ext_err[mask]**2))
inv_cov_plx_matrix = jnp.linalg.inv(jnp.diag(plx_err[mask]**2))



fig, axes = plt.subplots(1, 3, figsize = (18, 5))
axes[2].errorbar(x_obs, y_obs, yerr = ext_err[mask], xerr = x_error, fmt='.', markersize=4, ecolor='#9ab8d8', elinewidth=0.8, capsize=0, alpha=0.8, label='observed')
axes[2].plot(dust_x_linspace, jnp.interp(dust_x_linspace, radial_boundaries, cumulative_integrated_dust), color='crimson', lw=1.5, label='Edenhofer Map', zorder = 10)
axes[2].set_xlabel('distance [pc]')
axes[2].set_ylabel('cumulative integrated dust')
axes[2].legend(frameon=False)
dist_err = jax.vmap(get_x_err_basic)(plx_err, plx)
distances_psc = 1000/plx
axes[0].errorbar(distances_psc[mask], ext[mask],
             yerr=ext_err[mask],      # vertical error bars
             xerr=dist_err[mask],     # horizontal error bars (optional)
             fmt='o',                 # marker style, no connecting line
             ecolor='gray',           # error bar color
             elinewidth=1,
             capsize=3,               # little caps on the ends
             alpha=0.7)
axes[0].plot(radial_boundaries, cumulative_integrated_dust, color="orange", alpha=1, zorder=5, label="Edenhofer Integrated Dust")
axes[0].legend()
axes[0].set_xlabel("Distance towards GC (pc)")
axes[0].set_ylabel("Extinction (arb units)")

axes[1].hist(
    distances_psc[mask], bins=30, color="skyblue", edgecolor="black", alpha=0.7
)
axes[1].set_title("Distrubtion of Distances for Gaia XP stars within 0.3 degree separation from the GC")
#%%
def mask_cov_square(cov, indices_to_keep):
    n = cov.shape[0]
    idx  = np.arange(n)
    M = np.eye(n)                              # identity baseline
    M[np.ix_(indices_to_keep, indices_to_keep)] = cov[np.ix_(indices_to_keep, indices_to_keep)]
    return M

def mask_cov_diagonal(cov, indices_to_keep):
    cov_diag = jnp.diagonal(cov)
    n = cov_diag.size
    M_diag = np.ones(n)
    M_diag[indices_to_keep] = cov_diag[indices_to_keep]
    return jnp.diag(M_diag)

num_overall_steps = 0
burn_in = 0
num_integration_steps = 60000
initial_step_size = 0.00002
step_size = initial_step_size
inv_mass_matrix = jnp.eye(num_fourier_dimensions + num_fitting_params + num_data)

num_initial_dims = num_fourier_dimensions + num_fitting_params
indices_to_keep = np.concatenate([np.arange(0, 20), np.arange(num_initial_dims-20, num_initial_dims)])
#inv_mass_matrix = mask_cov_diagonal(jnp.diag(jnp.asarray(np.load('inv_mass_matrix_Gaia_XP_fitting_actual_data.npy'))), indices_to_keep)

ending_diag_values = x_error**2
ending_indices = jnp.arange(num_fourier_dimensions + num_fitting_params, num_fourier_dimensions + num_fitting_params + num_data)
inv_mass_matrix = inv_mass_matrix.at[ending_indices, ending_indices].set(ending_diag_values)
alpha_beta_indices = jnp.array([3, 4])
inv_mass_matrix = inv_mass_matrix.at[alpha_beta_indices, alpha_beta_indices].set([0.25, 0.25])


initial_logalpha = jnp.log(1.5)
initial_logbeta = jnp.log(2.0)
initial_logvar = 4.0
initial_lognu = 0
initial_offset = -7.6

logalpha_prior_mean = initial_logalpha
logbeta_prior_mean = initial_logbeta
logvar_prior_mean = initial_logvar
lognu_prior_mean = 0.0
offset_prior_mean = initial_offset

fixed_logscale = jnp.log(2*(max_distance - min_distance))

normalization_factor = num_fourier_dimensions/jnp.sqrt(fourier_length)

print(f"Max distance is: {max_distance}")
print(f"Min distance is {min_distance}")
doing_burn_in = False
use_NUTS = False
use_blackjax_hmc = False
window_adapt_square_mass_matrix = False
#%%
def hartley(x):
    fft = jnp.fft.fft(x)
    return jnp.real(fft) - jnp.imag(fft)

def cumulative_trapezoid(y, dx):
    seg = dx * (y[:-1] + y[1:]) / 2.0            # area of each trapezoid segment
    return jnp.concatenate([jnp.zeros(1, y.dtype), jnp.cumsum(seg)])

def inv_gamma_logpdf(x, alpha, beta):
    # β^α / Γ(α) * x^(-α-1) * exp(-β/x),  x > 0
    return jnp.where(
        x > 0,
        alpha * jnp.log(beta) - gammaln(alpha) - (alpha + 1.0) * jnp.log(x) - beta / x,
        -jnp.inf,
    )

def inv_gamma_logpdf(x, alpha, beta):
    # β^α / Γ(α) * x^(-α-1) * exp(-β/x),  x > 0
    return jnp.where(
        x > 0,
        alpha * jnp.log(beta) - gammaln(alpha) - (alpha + 1.0) * jnp.log(x) - beta / x,
        -jnp.inf,
    )


def exponentiated_integrated_density(logdensity):
    density = jnp.exp(logdensity)
    return cumulative_trapezoid(density, dx)

def response_function(xi, r, inference_params, offset):
    field = fft_field_hartley(xi, inference_params) + offset
    exp_int_field = exponentiated_integrated_density(field)
    return jnp.interp(r, dust_x_linspace, exp_int_field)

def general_matern_spectral_density(omega, inference_params, jitter = 1e-6):
        logvar, lognu = inference_params
        #logvar, lognu = 5.0, jnp.log(0.475)
        logscale = fixed_logscale
        nu = jnp.exp(lognu)
        log_ratio = jss.gammaln(nu + 0.5) - jss.gammaln(nu)
        r = jnp.square(omega) * jnp.exp(2 * logscale - jnp.log(2.0) - lognu)
        logdensity = logvar + log_ratio - 0.5*jnp.log(2*nu) + logscale - (nu + 0.5)*jnp.log1p(r)
        result = 2*jnp.sqrt(jnp.pi)*jnp.exp(logdensity)
        return jnp.where(omega > 0, result, result * (1.0 + jitter))

def average_chi_2_based_on_extinction(r, extinction, x_field, y_field, inverse_data_cov):
    def single(r_i, y_field_i):
        model_at_stars = jnp.interp(r_i, x_field, y_field_i)
        residual = extinction - model_at_stars
        return residual.T @ inverse_data_cov @ residual
    return jnp.average(jax.vmap(single, in_axes=(0, 0))(r, y_field))/extinction.size


def average_chi_2_based_on_r(r, plx_obs, inv_plx_cov):
    def single(r_i):
        residual = dist_to_plx(r_i) - plx_obs
        return residual.T @ inv_plx_cov @ residual
    return jnp.average(jax.vmap(single, in_axes = (0,))(r))/plx_obs.size

def matern_magnitude(omegas, inference_params):
     return jnp.sqrt(general_matern_spectral_density(omegas, inference_params))


def fft_field_hartley(noise, inference_params):
    size = noise.shape[0]
    raw_average_amplitude_data = matern_magnitude(omegas, inference_params)
    H = noise * raw_average_amplitude_data          # elementwise, whole array, in order
    y_data = (hartley(H) / size) * normalization_factor 
    return y_data[0:num_dimensions]



def logvarprior(logvar):
    return -0.5*(logvar - logvar_prior_mean)**2
def lognuprior(lognu):
    return -0.5*(lognu - lognu_prior_mean)**2
def logoffsetprior(offset):
    return -0.5*(offset - offset_prior_mean)**2
def logalphaprior(logalpha):
    return -0.5*(logalpha - logalpha_prior_mean)**2
def logbetaprior(logbeta):
    return -0.5*(logbeta - logbeta_prior_mean)**2
def negative_logdensity(x):
    logvar = x[0]
    lognu = x[1]
    offset = x[2]
    logalpha = x[3]
    logbeta = x[4]
    alpha = jnp.exp(logalpha)
    beta = jnp.exp(logbeta)
    xi = x[num_fitting_params:num_fitting_params + num_fourier_dimensions]
    r = x[num_fitting_params + num_fourier_dimensions:]
    # hardcoded r back to fixed values
    #r = x_obs_true
    inference_params = (logvar, lognu)
    # There are priors on everything: logvar, lognu, offset, xi, and r
    negative_log_p_s = -logvarprior(logvar) - lognuprior(lognu) - logoffsetprior(offset) + 0.5*xi.T @ xi - jnp.sum(distance_logprior(r))
    field_res = y_obs - response_function(xi, r, inference_params, offset)
    plx_res = plx_obs - dist_to_plx(r)
    student_t_sigma = jnp.sqrt(beta/alpha)
    student_t_nu = 2*alpha
    field_t_scores = field_res/ext_err[mask]
    log_data_likelihoods = t.logpdf(field_t_scores, df = student_t_nu, loc = 0.0, scale = student_t_sigma)



    # P(all data given field, params, distance) is now the probability of observing the extinctions (first term) times the probabilty of observing the correct distances (second term)
    negative_log_p_d_given_s = -1*jnp.sum(log_data_likelihoods) + 0.5*plx_res.T @ inv_cov_plx_matrix @ plx_res
    return negative_log_p_s + negative_log_p_d_given_s


if(doing_burn_in):
    initial_coordinates = jnp.zeros(num_fourier_dimensions + num_fitting_params) 
    initial_coordinates = initial_coordinates.at[0].set(initial_logvar)
    initial_coordinates = initial_coordinates.at[1].set(initial_lognu)
    initial_coordinates = initial_coordinates.at[2].set(initial_offset)
    initial_coordiantes = initial_coordinates.at[3].set(initial_logalpha)
    initial_coordiantes = initial_coordinates.at[4].set(initial_logbeta)
    initial_coordinates = jnp.concatenate([initial_coordinates, x_obs])
else:
    initial_coordinates = jnp.asarray(np.load("Gaia XP/inferring_alpha_beta_last_position.npy"))

# Size of initial coordinates: num_fourier_dimensions + num_fitting_params + num_data
num_good_samples = num_overall_steps - burn_in
num_good_samples = 250
time_arr = jnp.arange(num_good_samples)


#%%
rng = jr.PRNGKey(5)
rng, k1, k2 = jr.split(rng, 3)
logdensity_fn = lambda x: -negative_logdensity(x)
if(use_NUTS):
        def inference_loop(rng_key, kernel, initial_state, num_samples):
            @jax.jit
            def one_step(state, key):
                state, info = kernel(key, state)
                return state, (state, info)
            keys = jr.split(rng_key, num_samples)
            _, (states, infos) = jax.lax.scan(one_step, initial_state, keys)
            return states, infos
        warmup = blackjax.window_adaptation(blackjax.nuts, logdensity_fn, is_mass_matrix_diagonal= not window_adapt_square_mass_matrix, initial_inverse_mass_matrix=jnp.diagonal(inv_mass_matrix),   imm_shrinkage_to_previous=0.0)
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

    # HMC needs num_integration_steps; pass it through window_adaptation.
    warmup = blackjax.window_adaptation(
        blackjax.hmc, logdensity_fn,
        num_integration_steps=num_integration_steps,
        is_mass_matrix_diagonal=not window_adapt_square_mass_matrix,
        initial_inverse_mass_matrix=jnp.diagonal(inv_mass_matrix),  
        imm_shrinkage_to_previous=0.0
    )
    (state, parameters), _ = warmup.run(k1, initial_coordinates, num_steps=burn_in)

    # ...and again when building the tuned kernel.
    hmc = blackjax.hmc(logdensity_fn, **parameters)
    initial_state = state

    states, infos = inference_loop(k2, hmc.step, initial_state, num_good_samples)

    overall_data_arr = states.position
    accept_prob_arr  = infos.acceptance_rate
else :
    sampler = HMCSampler(
        negative_logdensity= negative_logdensity,
        num_integration_steps =  num_integration_steps,
        step_size= step_size,
        inv_mass_matrix = inv_mass_matrix,
        #inv_mass_matrix = jnp.diag(jnp.array(np.load('blackjax_cov_matrix.npy'))),
        alpha=1.0
    )
    print("HMC set up!")
    overall_data_arr, overall_momentum_arr, accept_prob_arr = sampler.sample(
        start_position=initial_coordinates, 
        num_samples=num_overall_steps,
        burn_in = burn_in,
        rng_key=rng
    )



np.save(out('overall_data_inferring_alpha_beta.npy'), overall_data_arr)

overall_data_arr = jnp.asarray(np.load("sherlock_results/overall_data_inferring_alpha_beta_40073425.npy"))
print(overall_data_arr)
accept_prob_arr = jnp.zeros(num_good_samples)

#%%
logvars = overall_data_arr[:, 0]
lognus = overall_data_arr[:, 1]
offsets = overall_data_arr[:, 2]
logalphas = overall_data_arr[:, 3]
logbetas = overall_data_arr[:, 4]
overall_extinction_arr = overall_data_arr[:, num_fitting_params:num_fitting_params + num_fourier_dimensions]  
overall_star_location_arr = overall_data_arr[:, num_fitting_params + num_fourier_dimensions:]

mean_star_location_arr = overall_star_location_arr.mean(axis=0)
logscales = jnp.zeros_like(logvars)
inference_params = (logvars, lognus)


overall_extinction_arr_raw_before_offsets = jax.vmap(fft_field_hartley, in_axes=(0, 0))(
    overall_extinction_arr, inference_params
)
overall_extinction_arr_raw = overall_extinction_arr_raw_before_offsets + offsets[:, None]
overall_extinction_arr_exp = jax.vmap(exponentiated_integrated_density, in_axes=(0))(
    overall_extinction_arr_raw
)
overall_extinction_arr_raw = overall_extinction_arr_raw[:, 0:num_dimensions]
overall_extinction_arr_exp = overall_extinction_arr_exp[:, 0:num_dimensions]


mean_extinction_raw = overall_extinction_arr_raw.sum(axis=0)/num_good_samples
mean_extinction_exp = jnp.exp(overall_extinction_arr_raw).sum(axis = 0)/num_good_samples
log_mean_extinction_exp = jnp.log(mean_extinction_exp)
mean_extinction_exp_int = overall_extinction_arr_exp.sum(axis=0)/num_good_samples
vars = jnp.exp(logvars)
scales = jnp.exp(logscales)
nus = jnp.exp(lognus)
cov= jnp.cov(overall_data_arr, rowvar = False)


cov= jnp.cov(overall_data_arr, rowvar = False)


fig, axes = plt.subplots(2, 3, figsize = (18, 10))
timegrid = jnp.arange(num_fourier_dimensions + num_fitting_params + num_data)
axes[0][0].plot(timegrid, jnp.diagonal(cov), color = "Green", alpha = 0.5)
axes[0][0].set_title("Diagonal Covariance")
axes[0][0].set_yscale("log")
axes[0][2].plot(jnp.arange(data_plx_err.size), x_error**2)
C = np.asarray(cov)
off = C - np.diag(np.diag(C))
vmax = np.abs(off).max()         # symmetric limits so white = 0
im = axes[0][1].imshow(C, cmap='RdBu_r', vmin= -vmax, vmax=vmax)
fig.colorbar(im, ax=axes[0][1], label='covariance')
axes[0][1].set_title('Posterior covariance')
axes[0][1].set_xlabel('parameter index')
axes[0][1].set_ylabel('parameter index')




if use_NUTS or use_blackjax_hmc:
    step_size = parameters['step_size']
    inv_mass_matrix = parameters['inverse_mass_matrix']   # not 'inv_mass_matrix'
    if(window_adapt_square_mass_matrix):
        C = np.asarray(inv_mass_matrix)
        off = C - np.diag(np.diag(C))
        vmax = np.abs(off).max()        
        im = axes[1][0].imshow(jnp.log(C), cmap='RdBu_r', vmin= -vmax, vmax=vmax)
        axes[1][0].colorbar(im, ax=axes[0], label='covariance')
        axes[1][0].set_title('Posterior covariance')
        axes[1][0].set_xlabel('parameter index')
        axes[1][0].set_ylabel('parameter index')
        axes[1][1].plot(jnp.arange(jnp.diagonal(inv_mass_matrix)).size, inv_mass_matrix)
    else :
        axes[1][0].plot(jnp.arange(inv_mass_matrix.size), inv_mass_matrix)


fig.tight_layout()
fig.savefig(out("covariance_plots_inferring_alpha_beta.png"), dpi=120, bbox_inches="tight")
#%%
fig, axes = plt.subplots(12, 3, figsize = (24, 60))
star_indices = jnp.arange(num_data)
star_mask = x_obs < 1400
star_indices = star_indices[star_mask]
samples_to_highlight = [10, 20, 30]


number_bins_x = 40
number_bins_y = 40
x_min, x_max = -0, 6
y_min, y_max = -4, 4
dx_plot = (x_max - x_min)/number_bins_x
dy_plot = (y_max - y_min)/number_bins_y


x_centers = (jnp.arange(number_bins_x) + 0.5)*dx_plot + x_min
y_centers = (jnp.arange(number_bins_y) + 0.5)*dy_plot + y_min
x_edges = jnp.arange(number_bins_x + 1)*dx_plot + x_min
y_edges = jnp.arange(number_bins_y + 1)*dy_plot + y_min


H_HMC_samples, _, _ = jnp.histogram2d(logvars, lognus, bins=[x_edges, y_edges])


im2 = axes[0][0].imshow(
    H_HMC_samples.T,
    origin='lower',
    extent=[x_min, x_max, y_min, y_max],
    aspect='auto',
    cmap='viridis',
)


fig.colorbar(im2, ax=axes[0][0], label='count')
axes[0][0].set_xlabel("log(Variance)")
axes[0][0].set_ylabel("log(Nu)")

for i in range (0, 10):
    axes[0][2].plot(dust_x_linspace, overall_extinction_arr_exp[20*i, :], color = "teal", alpha = 0.3)
    axes[0][1].plot(dust_x_linspace, overall_extinction_arr_raw[20*i, :], color = "teal", alpha = 0.3)


axes[0][1].plot(dust_x_linspace, mean_extinction_raw, color = "orange", alpha = 1, label = "HMC mean log differential extinction")
axes[0][1].plot(dust_x_linspace, log_mean_extinction_exp, color = "blue", alpha = 1, label = "log(HMC mean differential extinction)")
axes[0][1].plot(dust_x_linspace, jnp.interp(dust_x_linspace, radial_centers, jnp.log(mean_along_center)), color = "red", alpha = 1, label = "True Edenhofer log differential extinction")
axes[0][1].legend(loc = "upper right")


axes[0][2].plot(dust_x_linspace, mean_extinction_exp_int, color = "orange", alpha = 0.7, zorder = 45, label = "Mean HMC dust extinction")
axes[0][2].plot(dust_x_linspace, jnp.interp(dust_x_linspace, radial_boundaries, cumulative_integrated_dust), color = "red", alpha = 0.7, zorder = 50, label = "True Edenhofer extinction")
axes[0][2].errorbar(x_obs, y_obs, yerr = data_ext_err, xerr = x_error, fmt='.', markersize=4, ecolor='#9ab8d8', elinewidth=0.8, capsize=0, alpha=0.8, label="Star extinction/location errorbars")
axes[0][2].scatter(mean_star_location_arr, y_obs, color = "purple", s = 2, zorder = 10, label = "Mean HMC star locations")
axes[0][2].set_xlim(0, 2500)
axes[0][2].legend(loc = "upper right")


axes[1][0].set_ylabel("Log extinction density (inferred)")
axes[1][0].set_xlabel("Distance towards galactic center (parsecs)")
axes[1][0].plot(dust_x_linspace, mean_extinction_raw, color = "orange", alpha = 1, zorder = 10, label = "HMC mean log differential extinction")

indices_plot_10 = np.array([20*i for i in range(10)])
for k, i in enumerate(range(indices_plot_10.size)):
    axes[1][0].plot(dust_x_linspace, overall_extinction_arr_raw[indices_plot_10[i], :], color = "teal", alpha=0.2,)


axes[1][0].plot(dust_x_linspace, jnp.interp(dust_x_linspace, radial_centers, jnp.log(mean_along_center)), color = "red", alpha = 1, label = "True Edenhofer log differential extinction")
axes[1][0].legend(loc = "upper right")


axes[1][1].plot(time_arr, vars, color = "Orange")
axes[1][1].set_title("Variance Trace Plot")


axes[1][2].set_title("Acceptance Probability Histogram")
axes[1][2].hist(accept_prob_arr, bins=50, edgecolor='black', color='skyblue')


axes[2][0].plot(time_arr, nus, color = "Orange")
axes[2][0].set_title("Nu Trace Plot")


axes[2][1].plot(time_arr, overall_extinction_arr[:, 0], color = "Orange")
axes[2][1].set_title("Trace Plot of 1st Fourier Mode")


axes[2][2].plot(time_arr, overall_extinction_arr[:, 1], color = "Orange")
axes[2][2].set_title("Trace Plot of 2nd Fourier Mode")

axes[3][0].plot(time_arr, overall_extinction_arr[:, 9], color = "Orange")
axes[3][0].set_title("Trace Plot of 10th Fourier Mode")

axes[3][1].plot(time_arr, overall_extinction_arr[:, 99], color = "Orange")
axes[3][1].set_title("Trace Plot of 100th Fourier Mode")


axes[3][2].plot(time_arr, overall_extinction_arr[:, num_fourier_dimensions - 1], color = "Orange")
axes[3][2].set_title("Trace Plot of Last Fourier Mode")


h = axes[4][0].hist2d(logalphas, logbetas, bins=50, cmap='viridis')
axes[4][0].set_xlabel(r'$\log \alpha$')
axes[4][0].set_ylabel(r'$\log \beta$')
fig.colorbar(h[3], ax=axes[4][0], label='counts')
plt.show()
axes[4][0].set_title("2D posterior samples of log alpha and log beta")
axes[4][1].plot(time_arr, logalphas, color = "orange")
axes[4][1].set_title("Log alpha trace plot")
axes[4][2].plot(time_arr, logbetas, color = "orange")
axes[4][2].set_title("Log beta trace plot")

axes[5][0].plot(time_arr, overall_star_location_arr[:, star_indices[0]], color = "Green")
axes[5][0].set_title(f"Trace Plot of position of star {star_indices[0]}")


axes[5][1].plot(time_arr, overall_star_location_arr[:, star_indices[1]], color = "Green")
axes[5][1].set_title(f"Trace Plot of position of star {star_indices[1]}")


axes[5][2].plot(time_arr, offsets, color = "Purple")
axes[5][2].set_title("Offset Trace Plot")


fig.tight_layout()
fig.savefig(out("final_plots_inferring_alpha_beta.png"), dpi=120, bbox_inches="tight")


posterior_samples = overall_data_arr[np.newaxis, :, :]  # add chain axis -> (1, num_samples, 50)
idata = az.from_dict({"posterior": {"theta": posterior_samples}})
ess_param_index = 30  
az.plot_autocorr(idata, var_names="theta", coords={"theta_dim_0": [ess_param_index]})


ess_all = az.ess(idata, var_names="theta")  # a Dataset, ESS for all 50 parameters at once
indices_of_interest = {"Variance": 0, "Nu": 1, "First mode: ": num_fitting_params, "Second mode: ": num_fitting_params + 1, "10th mode ": num_fitting_params + 9, "100th mode": num_fitting_params + 99, 
                       "Last mode": num_fourier_dimensions + num_fitting_params - 1,  f"Star {star_indices[0]}": star_indices[0] + num_fitting_params + num_fourier_dimensions, f"Star {star_indices[1]}": star_indices[1] + num_fitting_params + num_fourier_dimensions}
for label, idx in indices_of_interest.items():
    ess_val = ess_all["theta"].sel(theta_dim_0=idx).values
    print(f"ESS for {label} parameter (index {idx}): {ess_val}")


print(f"Use NUTS? {use_NUTS} Use blackjax HMC? {use_blackjax_hmc} Initial Step Size: {initial_step_size} Step Size: {step_size} Num integration steps (only valid if no NUTS) {num_integration_steps} Burn in: {burn_in}")
print(f"Average acceptance probability (acceptance rate) is: {jnp.average(accept_prob_arr)}")
print(f"Num dimensions: {num_dimensions} Num data: {num_data} Total length: {max_distance - min_distance} Noise on each point: {error_bars_on_data} Num HMC trials: {num_overall_steps}")
prior_variances = jnp.diag(cov)[num_fitting_params:num_fitting_params + num_fourier_dimensions]
print(f"Error bars on data: {error_bars_on_data}")
print(f"Logvar prior mean: {logvar_prior_mean} Lognu prior mean: {lognu_prior_mean} Offset prior mean {offset_prior_mean} Initial logvar/lognu/offset values: {initial_logvar}, {initial_lognu}, {initial_offset}")
print(f"Average chi_2 statistic based on the prior: {jnp.average(prior_variances)}")  
print(f"Average chi_2 statistic based on the extinction per star: {average_chi_2_based_on_extinction(overall_star_location_arr, y_obs, dust_x_linspace, overall_extinction_arr_exp, inv_cov_data_matrix)}")
print(f"Average chi_2 statistic based on the distance/parallax per star: {average_chi_2_based_on_r(overall_star_location_arr, plx_obs, inv_cov_plx_matrix)}")

plt.close()

# %%
