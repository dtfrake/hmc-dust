import jax
import jax.numpy as jnp
from functools import partial
from typing import Callable, Tuple, Optional

# ---------------- SoftAbs / RMHMC building blocks (validated in earlier turns) ----------------

def softabs(l, alpha):
    return jnp.where(jnp.abs(l) < 1e-6, 1.0 / alpha, l / jnp.tanh(l * alpha))

def J_matrix(eigvals, alpha):
    f = softabs(eigvals, alpha)
    li, lj = eigvals[:, None], eigvals[None, :]
    fi, fj = f[:, None], f[None, :]
    denom = li - lj
    safe_denom = jnp.where(jnp.abs(denom) < 1e-10, 1.0, denom)
    J = (fi - fj) / safe_denom
    al = alpha * eigvals
    deriv = jnp.where(jnp.abs(al) < 1e-8, 0.0, 1.0/jnp.tanh(al) - al/jnp.sinh(al)**2)
    mask = jnp.abs(li - lj) < 1e-10
    J = jnp.where(mask, deriv[:, None], J)
    return J

def H_tilde(H, alpha):
    eigvals, Q = jnp.linalg.eigh(H)
    neweigvals = softabs(eigvals, alpha)
    return Q @ jnp.diag(neweigvals) @ Q.T

def dlog_softabs_det(eigvals, Q, dH_dtheta, alpha):
    f = softabs(eigvals, alpha)
    R = 1.0 / f
    J = J_matrix(eigvals, alpha)
    weight = R * jnp.diag(J)
    C = (Q * weight) @ Q.T
    return jnp.einsum('ij,ijk->k', C, dH_dtheta)

def grad_of_momentum_hamiltonian(eigvals, Q, p, dH_dtheta, alpha):
    mul = Q.T @ p
    D = jnp.diag(mul / softabs(eigvals, alpha))
    C = Q @ D @ J_matrix(eigvals, alpha) @ D @ Q.T
    return -jnp.einsum('ij,ijk->k', C, dH_dtheta)

def Hamiltonian(position, momentum, negative_logdensity, hess_at_position, alpha):
    H_smooth = H_tilde(hess_at_position, alpha)
    _, logdet = jnp.linalg.slogdet(H_smooth)
    return 0.5 * momentum @ jnp.linalg.solve(H_smooth, momentum) + 0.5 * logdet + negative_logdensity(position)


# ---------------- fully jit/scan-compatible one_leapfrog_step ----------------

def one_leapfrog_step(position, momentum, negative_logdensity, dt, alpha, key, tolerance, max_iterations):
    old_position = position
    old_momentum = momentum
    hess = jax.hessian(negative_logdensity)
    H_initial = hess(position)
    old_hamiltonian = Hamiltonian(position, momentum, negative_logdensity, H_initial, alpha)
    initial_eigvals, initial_Q = jnp.linalg.eigh(H_initial)
    initial_dH_dtheta = jax.jacfwd(hess)(position)
    grad_phi_initial = 0.5 * dlog_softabs_det(initial_eigvals, initial_Q, initial_dH_dtheta, alpha) + jax.grad(negative_logdensity)(position)
    momentum = momentum - dt * 0.5 * grad_phi_initial

    rho = momentum

    def tau_cond(state):
        _, error, it = state
        return jnp.logical_and(error > tolerance, it < max_iterations)

    def tau_body(state):
        mom, _, it = state
        partial_tau_partial_q = 0.5 * grad_of_momentum_hamiltonian(initial_eigvals, initial_Q, mom, initial_dH_dtheta, alpha)
        new_mom = rho - dt * 0.5 * partial_tau_partial_q
        new_error = jnp.max(jnp.abs(mom - new_mom))
        return (new_mom, new_error, it + 1)

    init_state_tau = (rho, jnp.asarray(jnp.inf, dtype=rho.dtype), jnp.asarray(0, dtype=jnp.int32))
    momentum, tau_error, _ = jax.lax.while_loop(tau_cond, tau_body, init_state_tau)
    diverged_tau = tau_error > tolerance

    sigma = position
    H_tilde_old = H_tilde(H_initial, alpha)
    partial_tau_partial_p_old = jnp.linalg.solve(H_tilde_old, momentum)

    def T_cond(state):
        _, error, it = state
        return jnp.logical_and(error > tolerance, it < max_iterations)

    def T_body(state):
        pos, _, it = state
        H_new = hess(pos)
        H_tilde_new = H_tilde(H_new, alpha)
        partial_tau_partial_p_new = jnp.linalg.solve(H_tilde_new, momentum)
        new_pos = sigma + dt * 0.5 * (partial_tau_partial_p_old + partial_tau_partial_p_new)
        new_error = jnp.max(jnp.abs(pos - new_pos))
        return (new_pos, new_error, it + 1)

    init_state_T = (sigma, jnp.asarray(jnp.inf, dtype=sigma.dtype), jnp.asarray(0, dtype=jnp.int32))
    position, T_error, _ = jax.lax.while_loop(T_cond, T_body, init_state_T)
    diverged_T = T_error > tolerance

    H_final = hess(position)
    final_eigvals, final_Q = jnp.linalg.eigh(H_final)
    final_dH_dtheta = jax.jacfwd(hess)(position)
    partial_tau_partial_q_final = 0.5 * grad_of_momentum_hamiltonian(final_eigvals, final_Q, momentum, final_dH_dtheta, alpha)
    momentum = momentum - dt * 0.5 * partial_tau_partial_q_final

    grad_phi_final = 0.5 * dlog_softabs_det(final_eigvals, final_Q, final_dH_dtheta, alpha) + jax.grad(negative_logdensity)(position)
    momentum = momentum - dt * 0.5 * grad_phi_final

    new_hamiltonian = Hamiltonian(position, momentum, negative_logdensity, H_final, alpha)
    accept_prob = jnp.minimum(1.0, jnp.exp(old_hamiltonian - new_hamiltonian))
    accept_prob = jnp.where(jnp.isnan(accept_prob), 0.0, accept_prob)

    diverged = jnp.logical_or(diverged_tau, diverged_T)
    accept_prob = jnp.where(diverged, 0.0, accept_prob)

    was_rejected = jnp.logical_or(diverged, jax.random.uniform(key) >= accept_prob)

    position = jnp.where(was_rejected, old_position, position)
    momentum = jnp.where(was_rejected, old_momentum, momentum)

    return position, momentum, was_rejected, accept_prob


def sample_momentum(position, negative_logdensity, alpha, key):
    """momentum ~ N(0, Sigma(q)) where Sigma(q) is the SoftAbs metric at the current position."""
    hess = jax.hessian(negative_logdensity)
    H = hess(position)
    eigvals, Q = jnp.linalg.eigh(H)
    f = softabs(eigvals, alpha)
    z = jax.random.normal(key, shape=eigvals.shape)
    return Q @ (jnp.sqrt(f) * z)


@partial(
    jax.jit,
    static_argnames=('negative_logdensity', 'num_samples', 'num_integration_steps'),
)
def sample_rmhmc(
    negative_logdensity: Callable[[jnp.ndarray], float],
    start_position: jnp.ndarray,
    num_samples: int,
    step_size: float,
    num_integration_steps: int,
    alpha: float,
    rng: jax.random.PRNGKey,
    tolerance: float = 1e-8,
    max_iterations: int = 100,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    num_dimensions = start_position.shape[0]

    def one_overall_step(position, momentum, current_rng):
        momentum_key, leapfrog_key, next_rng = jax.random.split(current_rng, 3)
        momentum = sample_momentum(position, negative_logdensity, alpha, momentum_key)

        def leapfrog_scan_body(carry, key):
            position, momentum = carry
            position, momentum, _, accept_prob = one_leapfrog_step(
                position, momentum, negative_logdensity, step_size, alpha, key, tolerance, max_iterations
            )
            return (position, momentum), accept_prob

        keys = jax.random.split(leapfrog_key, num_integration_steps)
        (position, momentum), accept_prob_arr = jax.lax.scan(leapfrog_scan_body, (position, momentum), keys)
        return position, momentum, accept_prob_arr, next_rng

    def sampling_scan_body(carry, _):
        position, momentum, current_rng = carry
        position, momentum, accept_prob_arr, next_rng = one_overall_step(position, momentum, current_rng)
        return (position, momentum, next_rng), (position, momentum, accept_prob_arr)

    start_momentum = jnp.zeros_like(start_position)
    (_, _, final_rng), (positions, momenta, accept_prob_arr) = jax.lax.scan(
        sampling_scan_body, (start_position, start_momentum, rng), None, length=num_samples
    )
    return final_rng, positions, momenta, accept_prob_arr


class RMHMCSampler:
    def __init__(
        self,
        negative_logdensity,
        num_integration_steps=10,
        step_size=0.1,
        alpha=1.0,
        tolerance=1e-8,
        max_iterations=100,
    ):
        self.negative_logdensity = negative_logdensity
        self.num_integration_steps = num_integration_steps
        self.step_size = step_size
        self.alpha = alpha
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def sample(self, start_position, num_samples, burn_in, rng_key):
        _, positions, momenta, accept_prob_arr = sample_rmhmc(
            self.negative_logdensity,
            start_position,
            num_samples,
            self.step_size,
            self.num_integration_steps,
            self.alpha,
            rng_key,
            self.tolerance,
            self.max_iterations,
        )
        return (
            positions[burn_in:, :],
            momenta[burn_in:, :],
            accept_prob_arr[burn_in:, :],
        )