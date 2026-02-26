from __future__ import annotations

import typing
import numpy as np
import jax
import jax.numpy as jnp
from scipy.integrate import solve_ivp
from jax.experimental.ode import odeint

from .base import BaseSimulator


class SIR(BaseSimulator):
    def __init__(
            self,
            dim: int,
            dim_y: int,
            t_span: typing.Tuple[float, float] = (0.0, 160.0),
            saveat: float = 1.0,
            N: float = 1_000_000.0,
            I0: float = 1.0,
            R0: float = 0.0,
            total_count: int = 1000,
            summary: str = "subsample",
    ):
        """SIR simulator (JAX).


        Parameters
        ----------
        dim: number of parameters (should be 2: beta, gamma)
        dim_y: number of output channels (3 when returning raw S,I,R timeseries)
        t_span: (start, end) in days
        saveat: time resolution (days)
        N, I0, R0: initial population numbers
        total_count: number of trials for Binomial observation model
        summary: currently supports only "subsample" (returns 10 values) or None
        """

        super().__init__("sir", dim, dim_y)
        self.t_span = t_span
        self.saveat = saveat
        self.N = float(N)
        self.I0 = float(I0)
        self.R0 = float(R0)
        self.total_count = int(total_count)
        self.summary = summary

        # initial state S, I, R
        S0 = float(self.N - self.I0 - self.R0)
        self.u0 = jnp.array([S0, self.I0, self.R0], dtype=jnp.float32)

        # time points: inclusive of endpoint (matching PyTorch's arange(0, days+saveat, saveat))
        start, end = self.t_span
        # use jnp.arange so the array is JAX-friendly
        self.time_points = jnp.arange(start, end + self.saveat, self.saveat, dtype=jnp.float32)

    def sample_jax(self, theta: jnp.ndarray, seed: int) -> jnp.ndarray:
        """Simulate SIR for a single parameter set and return the observation summary.

        Parameters
        ----------
        theta: jnp.array of shape (2,) -> [beta, gamma]
        seed: integer PRNG seed

        Returns
        -------
        jnp.ndarray
        - if summary is "subsample": shape (10,) float32 (binomial counts)
        - if summary is None: shape (3, T) float32 (full S,I,R timeseries)
        """

        # define RHS inside like in Lotka-Volterra
        def sir_ode_jax(u, t, params):
            S, I, R = u
            beta, gamma = params
            dS = -beta * S * I / self.N
            dI = beta * S * I / self.N - gamma * I
            dR = gamma * I
            return jnp.array([dS, dI, dR], dtype=jnp.float32)

        # Ensure correct dtype
        theta = jnp.asarray(theta, dtype=jnp.float32)

        # Solve ODE: odeint(func, y0, t, *args)
        try:
            sol = odeint(sir_ode_jax, self.u0, self.time_points, theta)
            # odeint returns shape (T, 3) -- transpose to (3, T) to match PyTorch layout
            us = sol.T  # shape (3, T)
        except Exception:
            # If solver fails for any reason, return nan-filled observation
            if self.summary == "subsample":
                return jnp.full((10,), jnp.nan, dtype=jnp.float32)
            else:
                return jnp.full((3, self.time_points.shape[0]), jnp.nan, dtype=jnp.float32)

        # detect NaNs in solution
        if jnp.isnan(us).any():
            if self.summary == "subsample":
                return jnp.full((10,), jnp.nan, dtype=jnp.float32)
            else:
                return jnp.full((3, self.time_points.shape[0]), jnp.nan, dtype=jnp.float32)

        if self.summary is None:
            return us.astype(jnp.float32)

        if self.summary == "subsample":
            # follow the PyTorch implementation: use only I (index 1) every 17 timesteps
            indices = jnp.arange(0, us.shape[1], 17)[:10]
            I_sub = us[1, indices]  # infected counts at subsampled times

            # convert to probabilities for Binomial observation
            probs = jnp.clip(I_sub / self.N, 0.0, 1.0).astype(jnp.float32)

            # sample Binomial(total_count, probs) using JAX PRNG
            key = jax.random.PRNGKey(int(seed))
            key, subkey = jax.random.split(key)
            # jax.random.binomial(key, shape, p, n) -> returns counts
            counts = jax.random.binomial(subkey, shape=probs.shape, p=probs, n=self.total_count)

            return counts.astype(jnp.float32)

        # unsupported summary option
        raise NotImplementedError(f"Unsupported summary: {self.summary}")

    def sample_numpy(self, theta: np.ndarray, rng: np.random.Generator = None) -> np.ndarray:
        """NumPy implementation for a batch of parameter sets or a single set.


        Parameters
        ----------
        theta: shape (2,) or (B,2) array-like of [beta, gamma]
        rng: optional np.random.Generator for reproducibility


        Returns
        -------
        - if summary == 'subsample': array of shape (B, 10) or (10,) with binomial counts
        - if summary is None: array of shape (B, 3, T) or (3, T) with full timeseries
        """
        if rng is None:
            rng = np.random.default_rng()


        theta = np.asarray(theta, dtype=float)
        single = theta.ndim == 1
        if single:
            theta = theta[None, :]

        start, end = self.t_span
        t_eval = np.arange(start, end + self.saveat, self.saveat)
        T = len(t_eval)

        B = theta.shape[0]

        if self.summary is None:
            outs = np.full((B, 3, T), np.nan, dtype=np.float32)
        else:
            outs = np.full((B, 10), np.nan, dtype=np.float32)


        def sir_ode_np(t, u, beta, gamma):
            S, I, R = u
            dS = -beta * S * I / self.N
            dI = beta * S * I / self.N - gamma * I
            dR = gamma * I
            return np.array([dS, dI, dR], dtype=float)


        for i in range(B):
            beta, gamma = theta[i]
            u0 = np.array([self.N - self.I0 - self.R0, self.I0, self.R0], dtype=float)
            try:
                sol = solve_ivp(
                    fun=lambda t, u: sir_ode_np(t, u, beta, gamma),
                    t_span=(start, end),
                    y0=u0,
                    t_eval=t_eval,
                    method='RK45',
                    rtol=1e-8,
                    atol=1e-10,
                )
                if not sol.success or sol.y.shape[1] != T:
                    # leave NaNs
                    continue

                us = sol.y # shape (3, T)

                if self.summary is None:
                    outs[i] = us.astype(np.float32)
                    continue

                # subsample I every 17 timesteps (as in PyTorch)
                indices = np.arange(0, T, 17)[:10]
                I_sub = us[1, indices]

                probs = np.clip(I_sub / self.N, 0.0, 1.0)
                counts = rng.binomial(self.total_count, probs)
                outs[i] = counts.astype(np.float32)

            except Exception:
                # leave NaNs on failure
                continue

        if single:
            return outs[0]
        return outs


class LotkaVolterra(BaseSimulator):
    def __init__(self,
                 dim,
                 dim_y,
                 t_span=(0, 30),
                 n_steps=100,
                 initial_conditions=(30.0, 5.0)
                 ):
        '''
        Lotka-Volterra simulator.
        Parameters:
        - t_span: simulation time range (start, end)
        - n_steps: number of time points
        - initial_conditions: fixed initial [x0, y0] (default: [30, 5])
        '''
        super().__init__("lotka_volterra", dim, dim_y) # dim = 4 (alpha, beta delta, gamma), dim_y = 2

        self.t_span = t_span
        self.n_steps = n_steps
        self.initial_conditions = np.array(initial_conditions, dtype=np.float32)
        self.time_points = np.linspace(t_span[0], t_span[1], n_steps)

    def sample_jax(self, theta, seed):
        """
        Pure JAX Lotka-Volterra simulator for *one* parameter set.

        Args:
            theta: jnp.array of shape (4,) -> [alpha, beta, gamma, delta]

        Returns:
            jnp.array of shape (20,) -> subsampled log-normal observations
        """

        def lotka_volterra_ode_jax(u, t, params):
            """
            Lotka-Volterra differential equations.
            u: [prey, predator]
            t: time (not used because system is autonomous)
            params: [alpha, beta, gamma, delta]
            """
            x, y = u
            alpha, beta, gamma, delta = params
            dxdt = alpha * x - beta * x * y
            dydt = -gamma * y + delta * x * y
            return jnp.array([dxdt, dydt])

        # seed to key_i
        key, subkey = jax.random.split(jax.random.PRNGKey(seed))

        days = 20.0
        saveat = 0.1
        u0 = jnp.array([30.0, 1.0])  # initial [prey, predator]

        # Full time grid: [0, 0.1, ..., 20.0]
        t_eval = jnp.arange(0.0, days + saveat, saveat)  # shape (201,)

        # Solve ODE
        u_solution = odeint(lotka_volterra_ode_jax, u0, t_eval, theta)  # shape: (201, 2)

        # Transpose to match your shape: (2, timepoints)
        u_solution = u_solution.T  # shape: (2, 201)

        # Subsample every 21st point, take first 10
        subsample_indices = jnp.arange(0, u_solution.shape[1], 21)[:10]
        u_subsampled = u_solution[:, subsample_indices]  # shape: (2, 10)

        # Flatten interleaved [x0,y0,x1,y1,...]
        u_flat = u_subsampled.flatten()  # shape: (20,)

        # Clamp, log, sample from log-normal
        u_clamped = jnp.clip(u_flat, 1e-10, 1e4)
        log_loc = jnp.log(u_clamped)

        # Use PRNG for reproducible noise — pass key externally in practice
        # Here we generate a dummy key for illustration
        key, subkey = jax.random.split(key)
        normal_noise = jax.random.normal(subkey, shape=log_loc.shape) * 0.1
        log_normal_samples = log_loc + normal_noise

        return jnp.exp(log_normal_samples)

    def sample_numpy(self, theta):
        """
        Pure NumPy implementation of Lotka-Volterra simulator

        Args:
            theta: numpy array of shape (num_samples, 4) containing parameters
                    [alpha, beta, gamma, delta] for each sample

        Returns:
            numpy array of shape (num_samples, 20) containing subsampled log-normal observations
        """
        # Constants from original implementation
        days = 20.0 # total time span in days
        saveat = 0.1 # time resolution: save every 0.1 days
        u0 = np.array([30.0, 1.0])  # Initial conditions [prey, predator]
        tspan = (0.0, days) # Time span for integration
        dim_data = 20  # For subsample summary

        # Time points where we want the solution
        t_eval = np.arange(0.0, days + saveat, saveat) # [0.0, 0.1, ..., 20.0]

        num_samples = theta.shape[0]
        results = np.full((num_samples, dim_data), np.nan) # prefils results with NaN

        def lotka_volterra_ode_np(t, u, alpha, beta, gamma, delta):
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
            alpha, beta, gamma, delta = theta[i]

            try:
                # Solve the ODE system
                sol = solve_ivp(
                    fun=lambda t, u: lotka_volterra_ode_np(t, u, alpha, beta, gamma, delta),
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
