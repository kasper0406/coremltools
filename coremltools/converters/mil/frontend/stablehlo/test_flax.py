import jax
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

def test_flax_stacked_lax_scan():
    class TestStackedLaxScanLinear(nnx.Module):
        def __init__(self, rngs=nnx.Rngs):
            @partial(nnx.vmap, axis_size=3) # 3 hidden layers
            def create_hidden_layers(rngs: nnx.Rngs):
                return nnx.Linear(in_features=4, out_features=4, bias_init=nnx.initializers.ones, rngs=rngs)
            self.hidden_layers = create_hidden_layers(rngs)

            self.upscale_layer = nnx.Linear(in_features=2, out_features=4, bias_init=nnx.initializers.ones, rngs=rngs)
            self.downscale_layer = nnx.Linear(in_features=4, out_features=2, bias_init=nnx.initializers.ones, rngs=rngs)

        def __call__(self, x):
            out = self.upscale_layer(x)

            layer_def, layer_states = nnx.split(self.hidden_layers)
            def forward(x, layer_state):
                layer = nnx.merge(layer_def, layer_state)
                x = layer(x)
                return x, None
            out, _ = jax.lax.scan(forward, out, layer_states)

            out = self.downscale_layer(out)
            return out

    model = TestStackedLaxScanLinear(nnx.Rngs(0))
    run_and_compare(nnx.jit(model), (jnp.zeros((2, 2)), ))
