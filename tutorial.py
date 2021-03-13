import timeit
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

key = random.PRNGKey(0)
x = random.normal(key, (10,))
print(x)

'''
size = 3000
x = random.normal(key, (size, size), dtype=jnp.float32)

mysetup = """
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import numpy as np
"""
mycode = """
key = random.PRNGKey(0)
size = 3000
x = random.normal(key, (size, size), dtype=jnp.float32)
jnp.dot(x, x.T).block_until_ready()
"""
print (timeit.timeit(setup = mysetup,
                     stmt = mycode,
                     number = 10))


mycode = """
size = 3000
x = np.random.normal(size=(size, size)).astype(np.float32)
jnp.dot(x, x.T).block_until_ready()
"""
print (timeit.timeit(setup = mysetup,
                     stmt = mycode,
                     number = 10))
'''

x = random.normal(key, (100000,))

