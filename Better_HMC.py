import jax
import jax.numpy as jnp
from functools import partial
from typing import Callable, Tuple, Optional
 
 
@partial(
    jax.jit,
    static_argnames=(
        'negative_logdensity',
        'num_samples',
        'num_integration_steps',
        'track_leapfrog_positions',
    ),
)
def sample_hmc(
    negative_logdensity: Callable[[jnp.ndarray], float],
    start_position: jnp.ndarray,
    num_samples: int,
    step_size: float,
    num_integration_steps: int,
    inv_mass_matrix: jnp.ndarray,
    alpha: float,
    rng: jax.random.PRNGKey,
    track_leapfrog_positions: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray]]:
    """Production-grade generalized HMC using JAX Scan Operations.
 
    When track_leapfrog_positions=True, also returns overall_exact_position_arr
    of shape (num_samples, num_integration_steps, num_dimensions): the particle's
    position after each individual leapfrog step, not just after each accept/reject
    loop. Rejected proposals are reported as flat at the pre-step position, since
    that's where the chain actually stayed -- see the `jnp.where(accept, ...)` below.
    """
    num_dimensions = start_position.shape[0]
    mass_matrix = jnp.linalg.inv(inv_mass_matrix)
    mass_matrix_chol = jnp.linalg.cholesky(mass_matrix)
    negative_logdensity_grad = jax.grad(negative_logdensity)
 
    def one_leapfrog_step(position, momentum, dt, momentum_scale_factor):
        momentum = momentum - 0.5 * dt * negative_logdensity_grad(position)
        momentum *= momentum_scale_factor
        position = position + dt * (inv_mass_matrix @ momentum)
        momentum = momentum - 0.5 * dt * negative_logdensity_grad(position)
        momentum *= momentum_scale_factor
        return position, momentum
 
    def leapfrog_scan_body(carry, scale):
        position, momentum = carry
        position, momentum = one_leapfrog_step(position, momentum, step_size, scale)
        # Only stacking `position` here (not momentum) since that's what was asked
        # for. Change this to (position, momentum) if you want the momentum
        # trajectory too -- just remember to update the `exact_positions` handling
        # below to match.
        return (position, momentum), position
 
    def one_overall_step(position, momentum, current_rng):
        initial_position = position
        initial_momentum_key, accept_key, next_rng = jax.random.split(current_rng, 3)
        momentum = mass_matrix_chol @ jax.random.normal(initial_momentum_key, shape=(num_dimensions,))
        initial_momentum = momentum
 
        initial_H = negative_logdensity(initial_position) + 0.5 * initial_momentum @ inv_mass_matrix @ initial_momentum
 
        half = num_integration_steps // 2
        scales = jnp.where(jnp.arange(num_integration_steps) < half, alpha, 1.0 / alpha)
 
        (position, momentum), leapfrog_positions = jax.lax.scan(leapfrog_scan_body, (position, momentum), scales)
 
        final_H = negative_logdensity(position) + 0.5 * momentum @ inv_mass_matrix @ momentum
        acceptance_probability = jnp.minimum(1.0, jnp.exp(initial_H - final_H))
        real_acceptance_probability = jnp.where(jnp.isnan(acceptance_probability), 0.0, acceptance_probability)
 
        accept = jax.random.uniform(accept_key, shape=()) <= real_acceptance_probability
        position = jnp.where(accept, position, initial_position)
        momentum = jnp.where(accept, momentum, initial_momentum)
 
        if track_leapfrog_positions:
            # If rejected, the chain never actually visited these intermediate
            # points -- it stayed at initial_position. Swap in that behavior so
            # this array is consistent with `positions` below. To instead see the
            # raw *proposed* trajectory regardless of accept/reject, just use
            # `leapfrog_positions` directly instead of this jnp.where.
            initial_position_tiled = jnp.broadcast_to(initial_position, leapfrog_positions.shape)
            exact_positions = jnp.where(accept, leapfrog_positions, initial_position_tiled)
        else:
            exact_positions = None
 
        return position, momentum, exact_positions, real_acceptance_probability, next_rng
 
    def sampling_scan_body(carry, _):
        position, momentum, current_rng = carry
        position, momentum, exact_positions, accept_prob, next_rng = one_overall_step(position, momentum, current_rng)
        return (position, momentum, next_rng), (position, momentum, exact_positions, accept_prob)
 
    start_momentum = jnp.zeros_like(start_position)
    (_, _, final_rng), (positions, momenta, overall_exact_position_arr, accept_probs) = jax.lax.scan(
        sampling_scan_body, (start_position, start_momentum, rng), None, length=num_samples
    )
    # Note: when track_leapfrog_positions=False, overall_exact_position_arr comes
    # back as None (a valid, zero-cost JAX pytree leaf) -- no extra compute or
    # memory is spent scanning it in that case.
    return final_rng, positions, momenta, accept_probs, overall_exact_position_arr
 
 
class HMCSampler:
    def __init__(
        self,
        negative_logdensity,
        num_integration_steps=10,
        step_size=0.1,
        alpha=1.0,
        inv_mass_matrix=None,
        track_leapfrog_positions=False,
    ):
        self.negative_logdensity = negative_logdensity
        self.num_integration_steps = num_integration_steps
        self.step_size = step_size
        self.alpha = alpha
        self.inv_mass_matrix = inv_mass_matrix
        self.track_leapfrog_positions = track_leapfrog_positions
 
    def sample(self, start_position, num_samples, burn_in, rng_key):
        inv_matrix = jnp.eye(start_position.shape[0]) if self.inv_mass_matrix is None else self.inv_mass_matrix
        _, positions, momenta, accept_probs, overall_exact_position_arr = sample_hmc(
            self.negative_logdensity,
            start_position,
            num_samples,
            self.step_size,
            self.num_integration_steps,
            inv_matrix,
            self.alpha,
            rng_key,
            self.track_leapfrog_positions,
        )
        if self.track_leapfrog_positions:
            return (
                positions[burn_in:, :],
                momenta[burn_in:, :],
                accept_probs[burn_in:],
                overall_exact_position_arr[burn_in:, :, :],
            )
        return positions[burn_in:, :], momenta[burn_in:, :], accept_probs[burn_in:]