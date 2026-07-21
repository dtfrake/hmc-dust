import jax
import jax.numpy as jnp
from functools import partial
from typing import Callable, Tuple, Optional

# ---------------- SoftAbs / RMHMC building blocks ----------------

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


# ---------------- one_leapfrog_step: purely deterministic, no MH correction ----------------
# The only way a step's proposal is discarded now is if the implicit fixed-point solves
# fail to converge within max_iterations -- a numerical-failure check, not a random one.

def one_leapfrog_step(position, momentum, negative_logdensity, dt, alpha, tolerance, max_iterations):
    old_position = position
    old_momentum = momentum
    hess = jax.hessian(negative_logdensity)
    H_initial = hess(position)
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

    diverged = jnp.logical_or(diverged_tau, diverged_T)

    position = jnp.where(diverged, old_position, position)
    momentum = jnp.where(diverged, old_momentum, momentum)

    return position, momentum, diverged


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
    static_argnames=('negative_logdensity', 'num_samples', 'num_integration_steps', 'track_leapfrog_positions'),
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
    track_leapfrog_positions: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray]]:
    """RMHMC with a single Metropolis correction per overall step (standard generalized-leapfrog
    structure), rather than per leapfrog sub-step.

    If ANY individual leapfrog sub-step within an overall step fails to converge (its implicit
    fixed-point solves don't reach `tolerance` within `max_iterations`), the ENTIRE overall step
    is rejected outright and the chain stays at its pre-step state -- there is no partial/frozen-
    tail trajectory that still gets an energy-based accept/reject. This is what actually preserves
    detailed balance: a converged trajectory composed of individually-reversible steps is itself
    reversible, but a trajectory with a frozen tail generally is not, so it's never allowed to be
    counted as a real move.

    `accept_prob_arr` has shape (num_samples,). Values are the usual Metropolis acceptance
    probability in [0, 1], EXCEPT that a value of exactly -0.5 is a sentinel meaning "this overall
    step was rejected because a leapfrog sub-step diverged" -- distinct from a genuine low
    acceptance probability, which is always in [0, 1].

    When track_leapfrog_positions=True, also returns overall_exact_position_arr of shape
    (num_samples, num_integration_steps, num_dimensions). If the overall step is rejected (for
    either reason above), those intermediate positions are reported flat at the pre-step position,
    since the chain never actually visited them.
    """
    num_dimensions = start_position.shape[0]
    hess = jax.hessian(negative_logdensity)

    def one_overall_step(position, momentum, current_rng):
        initial_position = position
        momentum_key, accept_key, next_rng = jax.random.split(current_rng, 3)
        momentum = sample_momentum(position, negative_logdensity, alpha, momentum_key)
        initial_momentum = momentum

        initial_H = Hamiltonian(initial_position, initial_momentum, negative_logdensity, hess(initial_position), alpha)

        def leapfrog_scan_body(carry, _):
            position, momentum = carry
            position, momentum, diverged = one_leapfrog_step(
                position, momentum, negative_logdensity, step_size, alpha, tolerance, max_iterations
            )
            return (position, momentum), (position, diverged)

        (position, momentum), (leapfrog_positions, diverged_per_substep) = jax.lax.scan(
            leapfrog_scan_body, (position, momentum), None, length=num_integration_steps
        )
        any_diverged = jnp.any(diverged_per_substep)

        final_H = Hamiltonian(position, momentum, negative_logdensity, hess(position), alpha)
        accept_prob = jnp.minimum(1.0, jnp.exp(initial_H - final_H))
        accept_prob = jnp.where(jnp.isnan(accept_prob), 0.0, accept_prob)

        # A converged trajectory is accepted/rejected on the usual energy-based coin flip.
        # A trajectory with ANY diverged sub-step is rejected outright, full stop -- there is
        # no partial credit, since a trajectory with a frozen tail isn't provably reversible.
        accept = jnp.logical_and(jnp.logical_not(any_diverged), jax.random.uniform(accept_key) <= accept_prob)

        # -0.5 is a sentinel: "rejected because of divergence", not a real acceptance probability.
        accept_prob_reported = jnp.where(any_diverged, -0.5, accept_prob)

        position = jnp.where(accept, position, initial_position)
        momentum = jnp.where(accept, momentum, initial_momentum)

        if track_leapfrog_positions:
            initial_position_tiled = jnp.broadcast_to(initial_position, leapfrog_positions.shape)
            exact_positions = jnp.where(accept, leapfrog_positions, initial_position_tiled)
        else:
            exact_positions = None

        return position, momentum, accept_prob_reported, exact_positions, next_rng

    def sampling_scan_body(carry, _):
        position, momentum, current_rng = carry
        position, momentum, accept_prob, exact_positions, next_rng = one_overall_step(
            position, momentum, current_rng
        )
        return (position, momentum, next_rng), (position, momentum, accept_prob, exact_positions)

    start_momentum = jnp.zeros_like(start_position)
    (_, _, final_rng), (positions, momenta, accept_probs, overall_exact_position_arr) = jax.lax.scan(
        sampling_scan_body, (start_position, start_momentum, rng), None, length=num_samples
    )
    return final_rng, positions, momenta, accept_probs, overall_exact_position_arr


class RMHMCSampler2:
    def __init__(
        self,
        negative_logdensity,
        num_integration_steps=10,
        step_size=0.1,
        alpha=1.0,
        tolerance=1e-8,
        max_iterations=100,
        track_leapfrog_positions=False,
    ):
        self.negative_logdensity = negative_logdensity
        self.num_integration_steps = num_integration_steps
        self.step_size = step_size
        self.alpha = alpha
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.track_leapfrog_positions = track_leapfrog_positions

    def sample(self, start_position, num_samples, burn_in, rng_key):
        _, positions, momenta, accept_probs, overall_exact_position_arr = sample_rmhmc(
            self.negative_logdensity,
            start_position,
            num_samples,
            self.step_size,
            self.num_integration_steps,
            self.alpha,
            rng_key,
            self.tolerance,
            self.max_iterations,
            self.track_leapfrog_positions,
        )
        if self.track_leapfrog_positions:
            return (
                positions[burn_in:, :],
                momenta[burn_in:, :],
                accept_probs[burn_in:],
                overall_exact_position_arr[burn_in:, :, :],
            )
        return (
            positions[burn_in:, :],
            momenta[burn_in:, :],
            accept_probs[burn_in:],
        )