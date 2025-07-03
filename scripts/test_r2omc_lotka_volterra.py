import sbibm
import numpy as np
from scipy.integrate import solve_ivp


def simulator(params):
    """
    Pure NumPy implementation of Lotka-Volterra simulator

    Args:
        params: numpy array of shape (num_samples, 4) containing parameters
                [alpha, beta, gamma, delta] for each sample

    Returns:
        numpy array of shape (num_samples, 20) containing subsampled log-normal observations
    """
    # Constants from original implementation
    days = 20.0
    saveat = 0.1
    u0 = np.array([30.0, 1.0])  # Initial conditions [prey, predator]
    tspan = (0.0, days)
    dim_data = 20  # For subsample summary

    # Time points where we want the solution
    t_eval = np.arange(0.0, days + saveat, saveat) # 0 to 20 (inclusive) with step 0.1

    num_samples = params.shape[0]
    results = np.full((num_samples, dim_data), np.nan)

    breakpoint()

    def lotka_volterra_ode(t, u, alpha, beta, gamma, delta):
        """
        Lotka-Volterra differential equation system
        du/dt = [alpha*x - beta*x*y, -gamma*y + delta*x*y]
        where u = [x, y] = [prey, predator]
        """
        x, y = u
        dxdt = alpha * x - beta * x * y
        dydt = -gamma * y + delta * x * y
        return np.array([dxdt, dydt])

    for i in range(num_samples):
        alpha, beta, gamma, delta = params[i]

        try:
            # Solve the ODE system
            sol = solve_ivp(
                fun=lambda t, u: lotka_volterra_ode(t, u, alpha, beta, gamma, delta),
                t_span=tspan,
                y0=u0,
                t_eval=t_eval,
                method='RK45',
                rtol=1e-8,
                atol=1e-10
            )

            if sol.success and sol.y.shape[1] == len(t_eval):
                # Reshape solution: sol.y is (2, n_timepoints) = (2, 201) for 20 days with 0.1 step
                # We want to subsample every 21st point (::21) like in original
                u_solution = sol.y  # shape: (2, n_timepoints) = (2, 201)

                # Subsample every 21st point (equivalent to ::21 in original)
                # Original had int(dim_data_raw/2) = int(2*(20/0.1+1)/2) = int(202) = 201 timepoints
                # Subsampling ::21 gives us 10 points per species = 20 total points
                n_timepoints = u_solution.shape[1]
                subsample_indices = np.arange(0, n_timepoints, 21)[:10]  # Take first 10 subsampled points

                if len(subsample_indices) >= 10:
                    u_subsampled = u_solution[:, subsample_indices[:10]]  # shape: (2, 10)
                    u_flat = u_subsampled.flatten()  # shape: (20,) - interleaved [x0,y0,x1,y1,...]

                    # Apply log-normal noise like in original
                    # Original: LogNormal(loc=log(clamp(u, 1e-10, 10000)), scale=0.1)
                    u_clamped = np.clip(u_flat, 1e-10, 10000.0)
                    log_loc = np.log(u_clamped)

                    # Sample from log-normal distribution
                    # LogNormal(loc, scale) in PyTorch corresponds to
                    # exp(Normal(loc, scale)) which is lognormal with log-scale parameterization
                    normal_samples = np.random.normal(loc=log_loc, scale=0.1)
                    results[i] = np.exp(normal_samples)

        except Exception:
            # If integration fails, leave as NaN (already initialized)
            continue

    return results


# Example usage and testing
if __name__ == "__main__":
    # test_params = np.exp(normal_samples)
    test_params = np.array([[0.6859, 0.1076, 0.8879, 0.1168]])

    print("Test parameters shape:", test_params.shape)
    print("Sample parameters:", test_params[0])

    # Run simulator
    results = simulator(test_params)

    print("Results shape:", results.shape)
    print("Sample result (first 5 values):", results[0])
    print("Number of successful simulations:", np.sum(~np.isnan(results).any(axis=1)))

    task = sbibm.get_task("lotka_volterra")
    task.get_observation(1)
    # task.get_true_parameters(1)

import numpy as np


def simulator(params):
    """
    Pure NumPy implementation of Lotka-Volterra simulator

    Args:
        params: numpy array of shape (num_samples, 4) containing parameters
                [alpha, beta, gamma, delta] for each sample

    Returns:
        numpy array of shape (num_samples, 20) containing subsampled log-normal observations
    """
    # Constants
    dt = 0.1
    n_steps = 200  # 20 days / 0.1 dt
    u0 = np.array([30.0, 1.0])  # Initial conditions [prey, predator]

    num_samples = params.shape[0]
    results = np.zeros((num_samples, 20))

    def rk4_step(u, dt, alpha, beta, gamma, delta):
        """Single RK4 integration step"""

        def f(u):
            x, y = u
            return np.array([alpha * x - beta * x * y, -gamma * y + delta * x * y])

        k1 = f(u)
        k2 = f(u + 0.5 * dt * k1)
        k3 = f(u + 0.5 * dt * k2)
        k4 = f(u + dt * k3)
        return u + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    for i in range(num_samples):
        alpha, beta, gamma, delta = params[i]

        # Integrate ODE using RK4
        u = u0.copy()
        trajectory = [u.copy()]

        for _ in range(n_steps):
            u = rk4_step(u, dt, alpha, beta, gamma, delta)
            trajectory.append(u.copy())

        # Convert to array and subsample every 21st point
        trajectory = np.array(trajectory)  # shape: (201, 2)
        u_subsampled = trajectory[::21][:10]  # shape: (10, 2)
        u_flat = u_subsampled.flatten()  # shape: (20,)

        # Apply log-normal noise
        u_clamped = np.clip(u_flat, 1e-10, 10000.0)
        log_loc = np.log(u_clamped)
        normal_samples = np.random.normal(loc=log_loc, scale=0.1)
        results[i] = np.exp(normal_samples)

    return results


# Example usage and testing
if __name__ == "__main__":
    # test_params = np.exp(normal_samples)
    test_params = np.array([[0.6859, 0.1076, 0.8879, 0.1168]])

    print("Test parameters shape:", test_params.shape)
    print("Sample parameters:", test_params[0])

    # Run simulator
    results = simulator(test_params)

    print("Results shape:", results.shape)
    print("Sample result (first 5 values):", results[0])
    print("Number of successful simulations:", np.sum(~np.isnan(results).any(axis=1)))




# --------------------
    import jax
    import jax.numpy as jnp
    from jax import vmap


    def lotka_volterra_dynamics(u, params):
        """Lotka-Volterra dynamics function"""
        x, y = u
        alpha, beta, gamma, delta = params
        return jnp.array([alpha * x - beta * x * y, -gamma * y + delta * x * y])


    def rk4_step(u, dt, params):
        """Single RK4 integration step"""
        k1 = lotka_volterra_dynamics(u, params)
        k2 = lotka_volterra_dynamics(u + 0.5 * dt * k1, params)
        k3 = lotka_volterra_dynamics(u + 0.5 * dt * k2, params)
        k4 = lotka_volterra_dynamics(u + dt * k3, params)
        return u + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


    def integrate_ode(params, key):
        """
        Integrate single parameter set

        Args:
            params: array of shape (4,) containing [alpha, beta, gamma, delta]
            key: JAX random key for noise generation

        Returns:
            array of shape (20,) containing log-normal observations
        """
        # Constants
        dt = 0.1
        n_steps = 200
        u0 = jnp.array([30.0, 1.0])

        # Integration using scan for efficiency
        def scan_fn(u, _):
            u_next = rk4_step(u, dt, params)
            return u_next, u_next

        # Integrate and collect trajectory
        _, trajectory = jax.lax.scan(scan_fn, u0, None, length=n_steps)

        # Add initial condition to trajectory
        full_trajectory = jnp.concatenate([u0[None, :], trajectory], axis=0)  # shape: (201, 2)

        # Subsample every 21st point
        u_subsampled = full_trajectory[::21][:10]  # shape: (10, 2)
        u_flat = u_subsampled.flatten()  # shape: (20,)

        # Apply log-normal noise
        u_clamped = jnp.clip(u_flat, 1e-10, 10000.0)
        log_loc = jnp.log(u_clamped)
        normal_samples = jax.random.normal(key, shape=(20,)) * 0.1 + log_loc

        return jnp.exp(normal_samples)


    def simulator(params, key=None):
        """
        JAX implementation of Lotka-Volterra simulator with vmap support

        Args:
            params: array of shape (num_samples, 4) containing parameters
                    [alpha, beta, gamma, delta] for each sample
            key: JAX random key (optional, will create one if None)

        Returns:
            array of shape (num_samples, 20) containing log-normal observations
        """
        if key is None:
            key = jax.random.PRNGKey(42)

        num_samples = params.shape[0]
        keys = jax.random.split(key, num_samples)

        # Vectorize over parameter sets and random keys
        vectorized_integrate = vmap(integrate_ode, in_axes=(0, 0))

        return vectorized_integrate(params, keys)
