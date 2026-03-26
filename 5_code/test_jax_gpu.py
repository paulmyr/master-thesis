"""Quick JAX GPU check."""
import jax
import jax.numpy as jnp

print("JAX version:", jax.__version__)
print("Devices:", jax.devices())
print("Default backend:", jax.default_backend())

x = jnp.ones((4096, 4096))
y = (x @ x).block_until_ready()
print(f"Matmul 4096x4096 OK, sum={y[0, 0]:.0f}")

if jax.default_backend() == "gpu":
    print("GPU is active!")
else:
    print("WARNING: running on", jax.default_backend(), "- GPU not detected")
