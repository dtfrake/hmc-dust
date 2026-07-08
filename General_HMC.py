import jax
import jax.numpy as jnp
from functools import partial
from typing import Callable, Tuple, Optional

@partial(jax.jit, static_argnames=('negative_logdensity', 'num_samples', 'num_integration_steps'))
def sample_hmc(
    negative_logdensity: Callable[[jnp.ndarray], float],
    start_position: jnp.ndarray,
    num_samples: int,
    step_size: float,
    num_integration_steps: int,
    inv_mass_matrix: jnp.ndarray,
    alpha: float,
    rng: jax.random.PRNGKey
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Production-grade generalized HMC using JAX Scan Operations."""
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
        return (position, momentum), (position, momentum)

    def one_overall_step(position, momentum, current_rng):
        initial_position = position
        initial_momentum_key, accept_key, next_rng = jax.random.split(current_rng, 3)
        momentum = mass_matrix_chol @ jax.random.normal(initial_momentum_key, shape=(num_dimensions,))
        initial_momentum = momentum
        
        initial_H = negative_logdensity(initial_position) + 0.5 * initial_momentum @ inv_mass_matrix @ initial_momentum
        
        half = num_integration_steps // 2
        scales = jnp.where(jnp.arange(num_integration_steps) < half, alpha, 1.0 / alpha)
        
        (position, momentum), _ = jax.lax.scan(leapfrog_scan_body, (position, momentum), scales)
        
        final_H = negative_logdensity(position) + 0.5 * momentum @ inv_mass_matrix @ momentum
        acceptance_probability = jnp.minimum(1.0, jnp.exp(initial_H - final_H))
        real_acceptance_probability = jnp.where(jnp.isnan(acceptance_probability), 0.0, acceptance_probability)
        
        accept = jax.random.uniform(accept_key, shape=()) <= real_acceptance_probability
        position = jnp.where(accept, position, initial_position)
        momentum = jnp.where(accept, momentum, initial_momentum)
        
        return position, momentum, None, None, real_acceptance_probability, next_rng

    def sampling_scan_body(carry, _):
        position, momentum, current_rng = carry
        position, momentum, _, _, accept_prob, next_rng = one_overall_step(position, momentum, current_rng)
        return (position, momentum, next_rng), (position, momentum, accept_prob)

    start_momentum = jnp.zeros_like(start_position)
    (_, _, _), (positions, momenta, accept_probs) = jax.lax.scan(
        sampling_scan_body, (start_position, start_momentum, rng), None, length=num_samples
    )
    return _, positions, momenta, accept_probs


class HMCSamplerOld:
    def __init__(self, negative_logdensity, num_integration_steps=10, step_size=0.1, alpha=1.0, inv_mass_matrix=None):
        self.negative_logdensity = negative_logdensity
        self.num_integration_steps = num_integration_steps
        self.step_size = step_size
        self.alpha = alpha
        self.inv_mass_matrix = inv_mass_matrix

    def sample(self, start_position, num_samples, burn_in, rng_key):
        inv_matrix = jnp.eye(start_position.shape[0]) if self.inv_mass_matrix is None else self.inv_mass_matrix
        _, positions, momenta, accept_probs = sample_hmc(
            self.negative_logdensity, start_position, num_samples, 
            self.step_size, self.num_integration_steps, inv_matrix, self.alpha, rng_key
        )
        return positions[burn_in:, :], momenta[burn_in:, :], accept_probs[burn_in:]