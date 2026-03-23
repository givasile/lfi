from unittest import result
import numpy as np
import jax
import jax.numpy as jnp
from jax import random

class SMC:
    def __init__(self, x, obs):
        '''
        x: Input variables for the simulator function
        obs: True observation
        '''

        self.x = x
        self.obs = obs

    def infer_numpy(self, eps, Nsamples, prior, simulator_func, prev_samples=None, prev_weights=None):
        '''
        Perform one iteration of ABC with resampling.
        
        Parameters:
        - eps: Tolerance for acceptance 
        - target_samples: Number of accepted samples to accept
        - prior: Callable to generate prior samples
        - simulator_func: Callable to generate prior samples
        - Nsamples: number of samples to generate in this iteration
        - prev_samples: Samples from the previous iteration (optional)
        - prev_weights: Weights for the previous samples (optional)
        
        Returns:
        - accepted_samples: Accepted samples for this iteration
        - weights: weights corresponding to the accepted samples
        '''

        # if there are previous samples, resample based on the weights
        if prev_samples is not None and prev_weights is not None:
            resample_indices = np.random.choice(len(prev_samples), size=(Nsamples), p=prev_weights)
            thetas = prev_samples[resample_indices] 
            # Perturb particles
            perturbation_std = np.std(prev_samples)* 0.1
            thetas += np.random.normal(0, perturbation_std, size=thetas.shape)
        else:
            # Otherwise generate new samples from the prior
            thetas = prior(Nsamples)

        # Simulate data for each theta
        sim_data = simulator_func(thetas, self.x)

        # Compute distances from observed data
        distances = np.abs(self.obs - sim_data)

        # Accept samples where distance is within the tolerance(eps)
        accepted_indices = np.where(distances<eps)[0]
        accepted_samples = thetas[accepted_indices]

        # Calculate uniform weights for accepted samples
        weights = np.exp(-distances[accepted_indices] / eps) 
        #weights = np.ones(len(accepted_indices))

        # Normalize weights 
        weights /= np.sum(weights)

        # Return the accepted_samples and weights for this iteration 
        return accepted_samples, weights
    
    def smc_algorithm(self, tolerance_sequence, target_samples, Nsamples, prior, simulator_func):
        '''
        Perform Sequential Monte Carlo Algorithm for ABC.
        
        Parameters:
        - tolerance_sequence: List of tolerance values (eps) for each iteration
        - target_samples: Number of accepted samples to collect for the final posterior
        - prior: Callable to generate prior samples
        - simulator_func: Function to generate simulated data
        - Nsamples: Number of samples to generate each iteration
        
        Returns:
        - results: List of dictionaries containing posterior samples and weights
        '''

        results = []

        # Print the start of the process
        print(f"Starting SMC for epsilon: {tolerance_sequence[0]}")

        # Initial prior samples (first iteration)
        accepted_samples, weights = self.infer_numpy(tolerance_sequence[0], Nsamples, prior, simulator_func)

        if len(accepted_samples) < target_samples:
            raise ValueError(f"Accepted samples: {len(accepted_samples)}")


        # Store the results of the first iteration
        results.append({
            'epsilon': tolerance_sequence[0],
            'samples_pos': accepted_samples[:target_samples],
            'weights': weights[:target_samples],
        })

        # Iterative process based on tolerance sequence
        for eps in tolerance_sequence[1:]:
            print(f"Running SMC for epsilon={eps}")

            # For subsequent iterations, resample from accepted samples fo thetas
            accepted_samples, weights = self.infer_numpy(eps, Nsamples, prior, simulator_func,
                                                         prev_samples=accepted_samples, prev_weights=weights)
            
            if len(accepted_samples) < target_samples:
                raise ValueError(f"Accepted samples: {len(accepted_samples)}")
            
            # Store the results for the current epsilon
            results.append({
                'epsilon': eps,
                'samples_pos': accepted_samples[:target_samples],
                'weights': weights[:target_samples],
            })

    
        return results




class SMC_JAX:
    def __init__(self, x, obs):
        self.x = x
        self.obs = obs

    def infer_jax(self, eps, Nsamples, prior, simulator_func, prev_samples=None, prev_weights=None, key=None):

        '''
        Perform one iteration of ABC with resampling using JAX
        
        Parameters:
        - key: Random key to be used for this iteration

        Returns:
        - accepted_samples: Accepted samples for this iteration
        - weights: Weights corresponding to the accepted samples
        - keys: Subkeys used for this iteration 
        '''

        # Spli the key into Nsamples subkeys
        keys = random.split(key, Nsamples) 

        # if there are previous samples, resample based on the weights
        if prev_samples is not None and prev_weights is not None:
            resample_indices = random.choice(keys[0], len(prev_samples), shape=(Nsamples,), p=prev_weights)
            thetas = prev_samples[resample_indices]
            # Add perturbation to the resamples thetas
            perturbation_std = jnp.std(prev_samples)*0.1
            thetas += random.normal(keys[1], shape=thetas.shape) * perturbation_std
        else:
            # Otherwise generate new samples from the prior
            thetas = prior(Nsamples, keys=keys)

        # Simulate data for each theta using the simulator function
        sim_data = simulator_func(thetas, self.x, keys)

        # Compute distances from observed data
        distances = jnp.abs(self.obs - sim_data)

        # Accept samples where distances is within the tolerance (eps)
        accepted_indices = jnp.where(distances < eps)[0]
        # Check if any samples are accepted
        # if len(accepted_indices) == 0:
        #     print(f"Aceepted_samples: len(accepted_indices)")
        #     return jnp.array([]), jnp.array([]), keys
        accepted_samples = thetas[accepted_indices]

        # Calculate weights for accepted samples
        weights = jnp.exp(-distances[accepted_indices] / eps)

        # Normalize weights
        weights /= jnp.sum(weights)

        # Return the accepted samples, weights, and the updated keys for this iteration
        return accepted_samples, weights, keys[-1]
        
    def smc_algorithm(self, tolerance_sequence, target_samples, Nsamples, prior, simulator_func, *, key):
        '''
        Perform Sequential Monte Carlo Algorithm for ABC using JAX
        
        Parameters:
        - key: Initial random key for randomness
        
        Returns:
        - results: List of dictionaries containing posterior samples and weights
        '''

        results=[]

        # Print the start of the process
        print(f"Starting SMC for epsilon: {tolerance_sequence[0]}")

        # First iteration: pass the initial key
        accepted_samples, weights, key = self.infer_jax(tolerance_sequence[0], Nsamples, prior, simulator_func, key=key)

        if len(accepted_samples) < target_samples:
            raise ValueError(f"Insufficient samples: {len(accepted_samples)}")
        
        # Store the results of the first iteration
        results.append({
            'epsilon': tolerance_sequence[0],
            'samples_pos': accepted_samples[:target_samples],
            'weights': weights[:target_samples]
        })

        # Iterate process based on tolerance sequence
        for eps in tolerance_sequence[1:]:
            print(f"Running SMC for epsilon={eps}")

            # From subsequent iterations, resample from accepted samples
            accepted_samples, weights, key = self.infer_jax(eps, Nsamples, prior, simulator_func,
                                                            prev_samples=accepted_samples, prev_weights=weights,
                                                            key=key)
            
            if len(accepted_samples) < target_samples:
                raise ValueError(f"Accepted samples: {len(accepted_sampels)}")
            
            # Store the results for the current epsilon
            results.append({
                'epsilon': eps, 
                'samples_pos': accepted_samples[:target_samples],
                'weights': weights[:target_samples]
            })

        return results




