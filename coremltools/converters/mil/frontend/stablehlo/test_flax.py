from flax import nnx
import jax.numpy as jnp

from coremltools.converters.mil.frontend.stablehlo.test.test_jax import run_and_compare

def test_flax_nnx_linear():
    class TestLinear(nnx.Module):
        def __init__(self, rngs=nnx.Rngs):
            self.layer = nnx.Linear(in_features=2, out_features=4, rngs=rngs)
        
        def __call__(self, x):
            return self.layer(x)
    
    model = TestLinear(nnx.Rngs(0))
    run_and_compare(nnx.jit(model), (jnp.zeros((4, 2)), ))
