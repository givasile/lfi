import numpy as np
import jax
import jax.numpy as jnp
from jax import random
#from simulators import simulator, simulator2


class ABCInference:
    def __init__(self, x, obs, eps=None):
        '''
        Initialize the ABCInference with observed data and fixed variable
        
        Parameters:
        - x: ndarray, fixed variable for simulation
        - obs: ndarray, observed data for comparison
        - eps: float, optional tolerance threshold for Method 2
        '''

        self.x = x
        self.obs = obs
        self.eps = eps

    def infer(self, target_samples, Nsamples, prior, simulator_func, batch_size=None, key=None):
        '''
        Choose Method 1 or Method 2 based on the presence of eps.
        
        Parameters:
        - target_samples: int, number of accepted samples to collect,
        - Nsamples: int, number of samples to generate in total,
        - prior: callable, generates samples from the prior
        - simulator_func: function, generates simulated data,
        - batch_size: int, number of samples per batch (Method 2),
        - key: PRNGKey (only for JAX)
        
        Returns:
        samples_pos: ndarray, accepted_samples
        '''

        # Check the type of simulator_func to decide whether to use Numpy or Jax
        if 'jax' in simulator_func.__name__:
            # Use JAX methods if the simulator function is JAX-based
            if self.eps is None:
                # Method 1
                return self.infer_method1_jax(target_samples, Nsamples, prior, simulator_func, key)
            else:
                # Method 2
                return self.infer_method2_jax(target_samples, Nsamples, batch_size, prior, simulator_func, key)
        else:
            # Use Numpy method if the simulator function is Numpy-based
            if self.eps is None:
                # Method 1(Numpy)
                return self.infer_method1_numpy(target_samples, Nsamples, prior, simulator_func)
            else:
                if batch_size is None:
                    raise ValueError(":Method 2 requires batch_size parameter.")
                return self.infer_method2_numpy(target_samples, Nsamples, batch_size, prior, simulator_func)
            
        # if self.eps is None:
        #     # Use method 1 if eps is None 
        #     return self.infer_method1(target_samples, Nsamples, prior, simulator_func, key)
        # else:
        #     # Use method 2
        #     if batch_size is None:
        #         raise ValueError(":Method 2 requires batch_size paramater.")
        #     return self.infer_method2(target_samples, Nsamples, batch_size, self.eps, prior, simulator_func, key)
        
    # Method 1 - Numpy Implementation 
    def infer_method1_numpy(self, target_samples, Nsamples, prior, simulator_func):
        '''
        Approximate Bayesian Computation Method 1 using Numpy 

        Returns:
        samples_pos: ndarray, best target_samples accepted sampels
        '''

        # Generate Nsamples thetas
        thetas = prior(Nsamples)

        # Generate Nsamples simulated data
        sim_data = simulator_func(thetas, self.x)

        # Compute distances
        distances = np.abs(self.obs-sim_data)
        
        # Select the target_samples smallest distances
        best_indices = np.argsort(distances)[:target_samples]
        
        # Select the top target_samples samples
        samples_pos = thetas[best_indices]
        return samples_pos
    
    # Method 1 - JAX implementation
    def infer_method1_jax(self, target_samples, Nsamples, prior, simulator_func, key):
        '''
        Approximate Bayesian Computation Method 1 using JAX implementation
        
        Returns:
        samples_pos: ndarray, best target_samples accepted samples
        '''
        # Generate Nsamples keys
        keys = random.split(key, Nsamples)

        # Generate Nsamples thetas
        thetas = prior(Nsamples, keys)

        # Generated Nsamples sim_data
        sim_data = simulator_func(thetas, self.x, keys)

        # Compute the distances
        distances = jnp.abs(self.obs - sim_data)

        # Select the target_samples smallest distances
        best_indices = jnp.argsort(distances)[:target_samples]
        
        # Select the top target_samples samples
        samples_pos = thetas[best_indices]
        return samples_pos
    
    # Method 2 - Numpy Implementation 
    def infer_method2_numpy(self, target_samples, Nsamples, batch_size, prior, simulator_func):
        '''
        Approximate Bayesian Computation Method 2
        
        Returns:
        samples_pos: ndarray, accepted samples
        '''
        samples_pos = []
        step=0

        while len(samples_pos) < target_samples and step < Nsamples:

            # Generate a batch of candidate samples
            thetas = prior(batch_size)

            # Simulated data for all parameters in the batch
            sim_data = simulator_func(thetas, self.x) # Shape: (batch_size,)

            # Compute the distances
            distances = np.abs(self.obs-sim_data) # Shape: (batch_size, )

            # Identify indices of accepted samples
            accepted_indices = np.where(distances < self.eps)[0]

            # Extact accepted samples
            accepted_samples = thetas[accepted_indices]

            # Add accepted samples to samples_pos
            samples_pos.extend(accepted_samples)
            
            # Stop if we have collected enough samples
            if len(samples_pos) >= target_samples:
                break

            # Increment the step count
            #step += 1
            step += batch_size
            
        print(f"The final step is: {step}")

        if len(samples_pos)<target_samples:
            raise ValueError(f"Only {len(samples_pos)} accepted samples")
        
        return  np.array(samples_pos[:target_samples])
    
    # Method 2 - JAX Implementation 
    def infer_method2_jax(self, target_samples, Nsamples, batch_size, prior, simulator_func, key):
        '''
        Approximate Bayesian Computation Method 2
        
        Returns:
        samples_pos: ndarray, accepted samples
        '''
        samples_pos= []
        step=0

        while len(samples_pos)<target_samples and step < Nsamples:

            # Split the key once, and then update if for the next iteration
            keys, key = random.split(key)

            # Now split into batch_size subkeys
            keys = random.split(keys, batch_size)

            # Generate a batch of candidate samples
            thetas = prior(batch_size, keys) # shape (batch_size, dim)

            # Generate simulated data
            sim_data = simulator_func(thetas, self.x, keys) # shape (batch_size,)

            # Compute the distances
            distances=jnp.abs(self.obs-sim_data) # Shape: (batch_size,)

            # Identify accepted samples (satisfy the distance criterion)
            accepted_indices = jnp.where(distances<self.eps)[0]

            # Extrac accepted samples from the batch
            accepted_samples_batch = thetas[accepted_indices]

            # Add accepted samples to the list
            samples_pos.extend(accepted_samples_batch)

            # Stop if we have collected enough accepted samples
            if len(samples_pos) >= target_samples:
                break

            # Increment the step count
            step += batch_size
            #step+=1

        print(f"The final step is: {step}")

        # if we can't collect enough accepted samples, raise an error
        if len(samples_pos) < target_samples:
            raise ValueError(f"Only {len(samples_pos)} accepted sampes after {step} batches")
        
        # Return only the firs target_samples
        return jnp.array(samples_pos[:target_samples])
    
    
    
    
 
        

 # type: ignore