from flax import nnx
import jax.numpy as jnp

from functools import partial

from coremltools.converters.mil.frontend.stablehlo.test.test_jax import run_and_compare

def test_flax_nnx_linear():
    class TestLinear(nnx.Module):
        def __init__(self, rngs=nnx.Rngs):
            self.layer = nnx.Linear(in_features=2, out_features=4, rngs=rngs)
        
        def __call__(self, x):
            return self.layer(x)
    
    model = TestLinear(nnx.Rngs(0))
    run_and_compare(nnx.jit(model), (jnp.zeros((4, 2)), ))

def test_flax_stacked_linear():
    class TestStackedLinear(nnx.Module):
        def __init__(self, rngs=nnx.Rngs):
            self.upscale_layer = nnx.Linear(in_features=2, out_features=4, bias_init=nnx.initializers.ones, rngs=rngs)

            self.hidden_layers = []
            for _ in range(3): # 3 hidden layers
                self.hidden_layers.append(nnx.Linear(in_features=4, out_features=4, bias_init=nnx.initializers.ones, rngs=rngs))
            self.downscale_layer = nnx.Linear(in_features=4, out_features=2, bias_init=nnx.initializers.ones, rngs=rngs)

        def __call__(self, x):
            out = self.upscale_layer(x)
            for layer in self.hidden_layers:
                out = layer(out)
            out = self.downscale_layer(out)
            return out
    
    model = TestStackedLinear(nnx.Rngs(0))
    run_and_compare(nnx.jit(model), (jnp.zeros((2, 2)), ))
