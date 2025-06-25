# JAX helper functions for shapeful library

def hello_jax():
    """Simple function to test JAX helper import"""
    return "Hello from JAX helper!"

def array_info(arr):
    """Get information about a JAX array"""
    return {
        'shape': arr.shape,
        'dtype': str(arr.dtype),
        'size': arr.size
    }

def create_identity_matrix(n):
    """Create an identity matrix of size n x n"""
    import jax.numpy as jnp
    return jnp.eye(n)
