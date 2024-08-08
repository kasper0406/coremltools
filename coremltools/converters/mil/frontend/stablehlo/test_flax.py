import jax
from flax import nnx
import jax.numpy as jnp

from functools import partial
from typing import Optional, List

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

def test_flax_convolution():
    class TestConvolution(nnx.Module):
        def __init__(self, rngs=nnx.Rngs):
            self.conv = nnx.Conv(in_features=2, out_features=1, kernel_size=3, rngs=rngs)

        def __call__(self, x):
            return self.conv(x)

    model = TestConvolution(nnx.Rngs(0))
    run_and_compare(nnx.jit(model), (jnp.zeros((2, 8, 2)), ))

def test_flax_stacked_convolution():
    class TestStackedConvolution(nnx.Module):
        def __init__(self, rngs=nnx.Rngs):
            @partial(nnx.vmap, axis_size=3) # 3 hidden layers
            def create_convs(rngs: nnx.Rngs):
                return nnx.Conv(in_features=2, out_features=2, kernel_size=3, rngs=rngs)
            self.conv_layers = create_convs(rngs)

        def __call__(self, x):
            layer_def, layer_states = nnx.split(self.conv_layers)
            def forward(x, layer_state):
                layer = nnx.merge(layer_def, layer_state)
                x = layer(x)
                x = nnx.relu(x)
                return x, None
            out, _ = jax.lax.scan(forward, x, layer_states)
            return out

    model = TestStackedConvolution(nnx.Rngs(0))
    run_and_compare(nnx.jit(model), (jnp.zeros((3, 8, 2)), ))

def test_flax_transposed_convolution():
    class TestTransposedConvolution(nnx.Module):
        def __init__(self, rngs=nnx.Rngs):
            self.conv = nnx.Conv(in_features=2, out_features=3, kernel_size=4, rngs=rngs)
            self.conv_transpose = nnx.ConvTranspose(in_features=3, out_features=2, kernel_size=3, rngs=rngs)

        def __call__(self, x):
            x = self.conv(x)
            x = self.conv_transpose(x)
            return x

    model = TestTransposedConvolution(nnx.Rngs(0))
    run_and_compare(nnx.jit(model), (jnp.zeros((4, 8, 2)), ))

def test_kernel_dilated_conv():
    class DilatedConvolution(nnx.Module):
        def __init__(self, rngs=nnx.Rngs):
            self.conv = nnx.Conv(in_features=4, out_features=2, kernel_size=4, kernel_dilation=2, rngs=rngs)

        def __call__(self, x):
            return self.conv(x)

    model = DilatedConvolution(nnx.Rngs(0))
    run_and_compare(nnx.jit(model), (jnp.zeros((4, 4, 4)), ))

def test_strided_conv_transpose():
    class StridedConvTranspose(nnx.Module):
        def __init__(self, rngs=nnx.Rngs):
            self.conv = nnx.ConvTranspose(in_features=4, out_features=2, kernel_size=3, strides=2, rngs=rngs)

        def __call__(self, x):
            return self.conv(x)

    model = StridedConvTranspose(nnx.Rngs(0))
    run_and_compare(nnx.jit(model), (jnp.zeros((4, 4, 4)), ))

def test_convolution_ranges():
    class ConvModel(nnx.Module):
        def __init__(self, conv_type, in_features: int, out_features: int, kernel_size: int, strides: int, dilation: int, rngs=nnx.Rngs):
            self.conv = conv_type(
                in_features=in_features,
                out_features=out_features,
                kernel_size=kernel_size,
                strides=strides,
                kernel_dilation=dilation,
                rngs=rngs
            )

        def __call__(self, x):
            return self.conv(x)

    for conv_type in [nnx.Conv, nnx.ConvTranspose]:
        for in_features in [1, 3]:
            for out_features in [1, 3]:
                for kernel_size in [2, 3]:
                    for strides in [2, 3]:
                        for dilation in [2, 3]:
                            model = ConvModel(
                                conv_type=conv_type,
                                in_features=in_features,
                                out_features=out_features,
                                kernel_size=kernel_size,
                                strides=strides,
                                dilation=dilation,
                                rngs=nnx.Rngs(0)
                            )
                            run_and_compare(nnx.jit(model), (jnp.zeros((2, 8, in_features)), ))

class ResidualConv(nnx.Module):
    scale_conv: nnx.Conv
    conv: nnx.Conv
    normalization_1: nnx.Module
    normalization_2: nnx.Module
    normalization_3: nnx.Module
    shortcut: nnx.Conv

    def __init__(self, in_channels: int, out_channels: int, rngs: nnx.Rngs, stride: int = 2):
        conv_type = nnx.Conv if in_channels <= out_channels else nnx.ConvTranspose

        kernel_size = 4
        self.scale_conv = conv_type(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=(kernel_size,),
            strides=(stride,),
            rngs=rngs
        )
        self.conv = nnx.Conv(
            in_features=out_channels,
            out_features=out_channels,
            kernel_size=kernel_size,
            rngs=rngs,
        )

        self.normalization_1 = nnx.BatchNorm(num_features=out_channels, rngs=rngs)
        self.normalization_2 = nnx.BatchNorm(num_features=out_channels, rngs=rngs)

        self.shortcut = conv_type(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=(stride,),
            strides=(stride,),
            rngs=rngs
        )

    def __call__(self, x):
        out = self.scale_conv(x)
        out = self.normalization_1(out)
        out = nnx.silu(out)

        out = self.conv(out)
        out = nnx.silu(out)

        # Residual
        out = out + self.shortcut(x)
        out = self.normalization_2(out)

        return out

def test_flax_residual_conv_module():
    model_upscale = ResidualConv(in_channels=2, out_channels=4, rngs=nnx.Rngs(0))
    model_upscale.eval()
    run_and_compare(nnx.jit(model_upscale), (jnp.zeros((4, 8, 2)), ))

    model_downscale = ResidualConv(in_channels=4, out_channels=2, rngs=nnx.Rngs(0))
    model_downscale.eval()
    run_and_compare(nnx.jit(model_downscale), (jnp.zeros((4, 4, 4)), ))

class Encoder(nnx.Module):
    cnn_layers: List[ResidualConv]
    normalization: nnx.Module

    def __init__(self, num_layers: int, rngs: nnx.Rngs):
        self.cnn_layers = []

        for i in range(num_layers):
            in_channels = (2 ** i)
            out_channels = 2 ** (i + 1)

            self.cnn_layers.append(ResidualConv(
                in_channels=in_channels,
                out_channels=out_channels,
                rngs=rngs,
            ))

        last_layer_features = 2 ** num_layers
        self.normalization = nnx.BatchNorm(num_features=last_layer_features, rngs=rngs)

    def __call__(self, x):
        out = x
        skip_connections = []
        for layer in self.cnn_layers:
            out = layer(out)
            skip_connections.append(out)

        out = self.normalization(out)
        out = nnx.tanh(out)

        return out, skip_connections

def test_encoder():
    model = Encoder(num_layers=3, rngs=nnx.Rngs(0))
    model.eval()
    run_and_compare(nnx.jit(model), (jnp.zeros((4, 8, 1)), ))

class Decoder(nnx.Module):
    cnn_layers: List[ResidualConv]
    residual_norm_layers: List[nnx.Module]
    output_polling: nnx.Conv

    def __init__(self, num_layers: int, rngs: nnx.Rngs):
        self.cnn_layers = []
        self.residual_norm_layers = []

        input_features = 2 ** num_layers
        for i in range(num_layers):
            # Times two to handle residual connections
            in_channels = 2 * (input_features // (2 ** i))
            out_channels = input_features // (2 ** (i + 1))

            self.residual_norm_layers.append(nnx.BatchNorm(in_channels, rngs=rngs))
            self.cnn_layers.append(ResidualConv(
                in_channels=in_channels,
                out_channels=out_channels,
                rngs=rngs,
            ))

        last_layer_features = input_features // (2 ** num_layers)
        self.output_polling = nnx.Conv(
            in_features=last_layer_features,
            out_features=1,
            kernel_size=3,
            rngs=rngs,
        )

    def __call__(self, x, skip_connections):
        skip_connections = list(reversed(skip_connections))

        out = x
        for i, (cnn_layer, residual_norm) in enumerate(zip(self.cnn_layers, self.residual_norm_layers)):
            residual = skip_connections[i]
            out = residual_norm(jnp.concatenate([out, residual], axis=-1))
            out = cnn_layer(out)

        out = self.output_polling(out)
        return out

class UNet(nnx.Module):
    encoder: Encoder
    decoder: Decoder

    def __init__(self, num_layers: int, rngs: nnx.Rngs):
        self.audio_encoding = Encoder(num_layers=num_layers, rngs=rngs)
        self.audio_decoding = Decoder(num_layers=num_layers, rngs=rngs)

    def __call__(self, x):
        def compress_dynamic_range(samples):
            mu = jnp.array(255.0, dtype=jnp.float16)
            return jnp.sign(samples) * jnp.log1p(mu * jnp.abs(samples)) / jnp.log1p(mu)
        x = compress_dynamic_range(x)

        hidden, skip_connections = self.audio_encoding(x)
        out = self.audio_decoding(hidden, skip_connections)

        return out

def test_unet():
    model = UNet(num_layers=3, rngs=nnx.Rngs(0))
    model.eval()
    run_and_compare(nnx.jit(model), (jnp.zeros((4, 8, 1)), ))
